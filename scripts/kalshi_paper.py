"""Paper-trade Kalshi's short-dated crypto contracts against a spot-price model. No Kalshi account needed.

Kalshi's hourly above/below, hourly range and 15-minute up/down markets settle on the 60-second
average of CF Benchmarks' real-time index. We estimate the fair probability of each contract from
live Kraken prices + recent realized volatility, and paper-buy whichever side the market misprices
after Kalshi's taker fee. Fills only happen at the real ask, up to the size actually offered.

Several variants run side by side (different tail models, vol windows, edge thresholds, plus a
model-free longshot fade), each with its own $1,000 paper bankroll. Every quote is also snapshotted
so we can score whether the model is better calibrated than the market at all.

  python scripts/kalshi_paper.py           run the paper trader (fleet.py starts it for you)
  python scripts/kalshi_paper.py report    leaderboard + calibration
"""
import logging
import math
import sqlite3
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import numpy as np
import requests
from scipy.stats import norm, t as student_t

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "runs" / "kalshi"
DB = DIR / "kalshi.sqlite"
API = "https://api.elections.kalshi.com/trade-api/v2"
PROXY = "http://127.0.0.1:8899"
COINS = {"BTC": "BTC/USD", "ETH": "ETH/USD", "SOL": "SOL/USD", "XRP": "XRP/USD", "DOGE": "DOGE/USD", "BNB": "BNB/USD"}
# Hourly above/below, hourly range, 15-minute up/down — only series that are actually listed get polled.
SERIES = {f"KX{c}{suffix}": c for c in COINS for suffix in ("D", "", "15M")}
BANKROLL = 1000.0
MAX_COST = 20.0          # dollars risked per trade
MAX_OPEN_FRAC = 0.5      # never have more than half the bankroll tied up
HORIZON_MIN = 6 * 60     # only trade contracts closing within 6h
MIN_TAU_MIN = 3          # stop trading 3 minutes before close (thin books, settlement average about to start)
MAX_SPREAD = 0.10        # skip markets whose yes bid/ask spread is wider than this (no real book)
POLL_SECS = 30
T_DF = 4                 # degrees of freedom for the fat-tailed model

log = logging.getLogger("kalshi")


@dataclass(frozen=True)
class Variant:
    name: str
    dist: str = "n"          # "n" normal, "t" Student-t (fat tails), "longshot" model-free
    half_life: int = 30      # minutes, EWMA volatility half-life
    edge: float = 0.03       # minimum expected profit per contract after fees, in dollars
    stop: float | None = None  # sell early if the side we hold drops this far (dollars) below entry


VARIANTS = [
    Variant("normal_fastvol_3c", "n", 30, 0.03),
    Variant("normal_fastvol_8c", "n", 30, 0.08),
    Variant("normal_slowvol_3c", "n", 240, 0.03),
    Variant("normal_slowvol_8c", "n", 240, 0.08),
    Variant("fattail_fastvol_3c", "t", 30, 0.03),
    Variant("fattail_fastvol_8c", "t", 30, 0.08),
    Variant("fattail_slowvol_3c", "t", 240, 0.03),
    Variant("fattail_slowvol_8c", "t", 240, 0.08),
    Variant("fattail_fastvol_5c_stop", "t", 30, 0.05, stop=0.15),
    Variant("longshot_fade", "longshot"),  # buy NO on <=8c YES contracts in the final hour, no model
]
MODELS = {"n30": ("n", 30), "n240": ("n", 240), "t30": ("t", 30), "t240": ("t", 240)}


def kalshi_fee(contracts: float, price: float) -> float:
    """Kalshi taker fee: 7% x C x P x (1-P), rounded up to the next cent."""
    return math.ceil(0.07 * contracts * price * (1 - price) * 100 - 1e-9) / 100


def prob(dist: str, sigma_min: float, tau_min: float, spot: float, lo: float | None, hi: float | None) -> float:
    """P(lo <= settlement <= hi) for a driftless log-price over tau minutes."""
    # The settlement is a 60s average, which removes ~40s worth of variance from the horizon.
    scale = sigma_min * math.sqrt(max(tau_min - 40 / 60, 1 / 60))
    if dist == "t":
        scale *= math.sqrt((T_DF - 2) / T_DF)  # same variance, fatter tails
        cdf = lambda z: student_t.cdf(z, T_DF)
    else:
        cdf = norm.cdf
    up = 1.0 if hi is None else cdf(math.log(hi / spot) / scale)
    down = 0.0 if lo is None else cdf(math.log(lo / spot) / scale)
    return float(min(max(up - down, 0.0), 1.0))


def bounds(m: dict) -> tuple[float | None, float | None]:
    st, fl, cap = m.get("strike_type"), m.get("floor_strike"), m.get("cap_strike")
    if st in ("greater", "greater_or_equal"):
        return fl, None
    if st in ("less", "less_or_equal"):
        return None, cap if cap is not None else fl
    if st == "between":
        return fl, cap
    return None, None


def f(x) -> float:
    try:
        return float(x or 0)
    except (TypeError, ValueError):
        return 0.0


def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


class Spot:
    """Kraken mid prices and EWMA 1-minute volatility, plus a running Kraken -> CF index basis."""

    def __init__(self):
        self.ex = ccxt.kraken({"urls": {"api": {"public": PROXY}}, "timeout": 15000})
        self.mid: dict[str, float] = {}
        self.sigma: dict[tuple[str, int], float] = {}
        self.history = {c: deque(maxlen=200) for c in COINS}  # (ts, mid) for basis estimation
        self.basis = {c: 0.0 for c in COINS}
        self.basis_known: set[str] = set()  # don't trade a coin until we've measured its basis once
        self._vol_at = 0.0

    def refresh(self) -> None:
        tick = self.ex.fetch_tickers(list(COINS.values()))
        now = time.time()
        for c, sym in COINS.items():
            t = tick.get(sym)
            if t and t.get("bid") and t.get("ask"):
                self.mid[c] = (t["bid"] + t["ask"]) / 2
                self.history[c].append((now, self.mid[c]))
        if now - self._vol_at > 60:
            for c, sym in COINS.items():
                closes = np.array([r[4] for r in self.ex.fetch_ohlcv(sym, "1m", limit=720)], dtype=float)
                r = np.diff(np.log(closes[closes > 0]))
                for hl in {v.half_life for v in VARIANTS}:
                    w = 0.5 ** (np.arange(len(r))[::-1] / hl)
                    self.sigma[(c, hl)] = max(math.sqrt(float(np.sum(w * r**2) / np.sum(w))), 1e-4)
            self._vol_at = now

    def observe_index(self, coin: str, open_ts: float, index_value: float) -> None:
        """A 15-minute market's strike is the CF index averaged over the 60s before it opened,
        so comparing it with Kraken over that minute measures the Kraken -> index basis."""
        # Kraken's 1-minute candle for that same minute approximates the 60s average better than our polls.
        bars = self.ex.fetch_ohlcv(COINS[coin], "1m", since=int(open_ts - 60) * 1000, limit=3)
        bar = next((r for r in bars if r[0] == int(open_ts - 60) * 1000), None)
        mids = [sum(bar[1:5]) / 4] if bar else [m for t, m in self.history[coin] if open_ts - 60 <= t <= open_ts + 5]
        if mids and index_value:
            b = index_value - sum(mids) / len(mids)
            if abs(b) < 0.01 * index_value:
                self.basis[coin] = 0.7 * self.basis[coin] + 0.3 * b if coin in self.basis_known else b
                self.basis_known.add(coin)

    def price(self, coin: str) -> float | None:
        return self.mid[coin] + self.basis[coin] if coin in self.mid else None


def db() -> sqlite3.Connection:
    DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY, variant TEXT, ticker TEXT, series TEXT, side TEXT,
            contracts REAL, price REAL, fee REAL, p_model REAL, edge REAL,
            opened_at TEXT, close_time TEXT, status TEXT DEFAULT 'open', result TEXT, payout REAL);
        CREATE TABLE IF NOT EXISTS snapshots (
            ticker TEXT, series TEXT, ts REAL, tau_min REAL, spot REAL,
            p_n30 REAL, p_n240 REAL, p_t30 REAL, p_t240 REAL, yes_bid REAL, yes_ask REAL, result TEXT,
            PRIMARY KEY (ticker, ts));
        CREATE INDEX IF NOT EXISTS snap_open ON snapshots(result);
    """)
    return con


class Trader:
    def __init__(self):
        self.http = requests.Session()
        self.spot = Spot()
        self.con = db()
        self.seen_15m: set[str] = set()
        self.last_snap: dict[str, float] = {}
        self.series = self._listed_series()

    def get(self, path: str, **params) -> dict:
        for attempt in range(4):
            r = self.http.get(f"{API}{path}", params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return {}

    def _listed_series(self) -> dict[str, str]:
        listed = {}
        for s, coin in SERIES.items():
            try:
                self.get(f"/series/{s}")
                listed[s] = coin
            except requests.HTTPError:
                pass
        log.info("watching %d series: %s", len(listed), " ".join(listed))
        return listed

    def markets(self) -> list[tuple[str, dict]]:
        now = int(time.time())
        out = []
        for s, coin in self.series.items():
            cursor = None
            while True:
                params = {"series_ticker": s, "status": "open", "limit": 1000,
                          "min_close_ts": now, "max_close_ts": now + HORIZON_MIN * 60}
                if cursor:
                    params["cursor"] = cursor
                page = self.get("/markets", **params)
                out += [(coin, m) for m in page.get("markets", [])]
                cursor = page.get("cursor")
                if not cursor:
                    break
        return out

    def cash(self, variant: str) -> tuple[float, float]:
        spent, payout, open_cost = self.con.execute(
            "SELECT COALESCE(SUM(contracts*price+fee),0), COALESCE(SUM(payout),0),"
            " COALESCE(SUM(CASE WHEN status='open' THEN contracts*price+fee END),0)"
            " FROM positions WHERE variant=?", (variant,)).fetchone()
        return BANKROLL - spent + payout, open_cost

    def holding(self, variant: str, ticker: str) -> bool:
        return self.con.execute("SELECT 1 FROM positions WHERE variant=? AND ticker=?", (variant, ticker)).fetchone() is not None

    def step(self) -> None:
        self.spot.refresh()
        now = time.time()
        quotes = {}
        for coin, m in self.markets():
            tkr = m["ticker"]
            if tkr.split("-")[0].endswith("15M") and tkr not in self.seen_15m:
                self.seen_15m.add(tkr)
                if now - ts(m["open_time"]) < 120:
                    self.spot.observe_index(coin, ts(m["open_time"]), f(m.get("floor_strike")))
            S = self.spot.price(coin)
            lo, hi = bounds(m)
            if S is None or (lo is None and hi is None):
                continue
            tau = (ts(m["close_time"]) - now) / 60
            yes_bid, yes_ask = f(m.get("yes_bid_dollars")), f(m.get("yes_ask_dollars"))
            quotes[tkr] = (yes_bid, f(m.get("yes_bid_size_fp")), yes_ask, f(m.get("yes_ask_size_fp")))
            if tau < MIN_TAU_MIN or coin not in self.spot.basis_known or yes_ask - yes_bid > MAX_SPREAD:
                continue
            if any((coin, hl) not in self.spot.sigma for _, hl in MODELS.values()):
                continue
            p = {k: prob(d, self.spot.sigma[(coin, hl)], tau, S, lo, hi) for k, (d, hl) in MODELS.items()}
            self.snapshot(coin, m, now, tau, S, p, yes_bid, yes_ask)
            for v in VARIANTS:
                self.consider(v, coin, m, tau, p, yes_bid, yes_ask)
        self.check_stops(quotes)
        self.settle()
        self.con.commit()

    def snapshot(self, coin, m, now, tau, S, p, yes_bid, yes_ask) -> None:
        tkr = m["ticker"]
        # One snapshot per market every 5 min, only for live two-sided quotes in the final hour.
        if tau > 60 or not (0.02 < yes_bid and yes_ask < 0.98) or now - self.last_snap.get(tkr, 0) < 300:
            return
        self.last_snap[tkr] = now
        self.con.execute("INSERT OR IGNORE INTO snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL)",
                         (tkr, tkr.split("-")[0], now, tau, S, p["n30"], p["n240"], p["t30"], p["t240"], yes_bid, yes_ask))

    def consider(self, v: Variant, coin, m, tau, p, yes_bid, yes_ask) -> None:
        tkr = m["ticker"]
        if self.holding(v.name, tkr):
            return
        # Buying YES pays the yes ask; buying NO pays 1 - yes bid, against the size resting on the yes bid.
        sides = [("yes", yes_ask, f(m.get("yes_ask_size_fp"))), ("no", 1 - yes_bid, f(m.get("yes_bid_size_fp")))]
        best = None
        for side, price, size in sides:
            if not (0.01 <= price <= 0.99) or size < 1:
                continue
            if v.dist == "longshot":
                if side != "no" or tau > 60 or not 0.03 <= yes_bid <= 0.08:
                    continue
                p_win, edge = None, 0.0
            else:
                p_yes = p[f"{v.dist}{v.half_life}"]
                p_win = p_yes if side == "yes" else 1 - p_yes
                edge = p_win - price - kalshi_fee(1, price)
                if edge < v.edge:
                    continue
            if best is None or edge > best[4]:
                best = (side, price, size, p_win, edge)
        if not best:
            return
        side, price, size, p_win, edge = best
        cash, open_cost = self.cash(v.name)
        budget = min(MAX_COST, cash, BANKROLL * MAX_OPEN_FRAC - open_cost)
        contracts = min(math.floor(budget / (price * 1.02)), math.floor(size))
        if contracts < 1:
            return
        fee = kalshi_fee(contracts, price)
        self.con.execute(
            "INSERT INTO positions (variant,ticker,series,side,contracts,price,fee,p_model,edge,opened_at,close_time)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (v.name, tkr, tkr.split("-")[0], side, contracts, price, fee, p_win, edge,
             datetime.now(timezone.utc).isoformat(), m["close_time"]))
        log.info("%-24s BUY %-3s %4d x %-34s @ %.2f  model=%s edge=%+.3f", v.name, side.upper(), contracts, tkr,
                 price, f"{p_win:.2f}" if p_win is not None else "-", edge)

    def check_stops(self, quotes: dict) -> None:
        stops = {v.name: v.stop for v in VARIANTS if v.stop}
        rows = self.con.execute(
            f"SELECT id, variant, ticker, side, contracts, price, opened_at FROM positions WHERE status='open'"
            f" AND variant IN ({','.join('?' * len(stops))})", list(stops)).fetchall()
        for pid, variant, tkr, side, n, entry, opened in rows:
            if tkr not in quotes or time.time() - ts(opened) < 120:
                continue
            yes_bid, yes_bid_size, yes_ask, yes_ask_size = quotes[tkr]
            # Selling YES hits the yes bid; selling NO means buying YES back at the yes ask.
            exit_px, depth = (yes_bid, yes_bid_size) if side == "yes" else (1 - yes_ask, yes_ask_size)
            if depth < n or yes_ask - yes_bid > MAX_SPREAD or not 0 < exit_px <= entry - stops[variant]:
                continue
            payout = n * exit_px - kalshi_fee(n, exit_px)
            self.con.execute("UPDATE positions SET status='stopped', payout=? WHERE id=?", (payout, pid))
            log.info("%-24s STOP %s @ %.2f (entry %.2f)", variant, tkr, exit_px, entry)

    def settle(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        tickers = [r[0] for r in self.con.execute(
            "SELECT DISTINCT ticker FROM positions WHERE status='open' AND close_time < ?"
            " UNION SELECT DISTINCT ticker FROM snapshots WHERE result IS NULL AND ts < ?",
            (now, time.time() - 120)).fetchall()]
        for i in range(0, len(tickers), 50):
            for m in self.get("/markets", tickers=",".join(tickers[i:i + 50]), limit=50).get("markets", []):
                res = m.get("result")
                if res not in ("yes", "no"):
                    continue
                self.con.execute("UPDATE snapshots SET result=? WHERE ticker=?", (res, m["ticker"]))
                for pid, side, n in self.con.execute(
                        "SELECT id, side, contracts FROM positions WHERE ticker=? AND status='open'", (m["ticker"],)).fetchall():
                    self.con.execute("UPDATE positions SET status='settled', result=?, payout=? WHERE id=?",
                                     (res, n if side == res else 0.0, pid))

    def run(self) -> None:
        log.info("kalshi paper trader started with %d variants", len(VARIANTS))
        while True:
            t0 = time.time()
            try:
                self.step()
            except Exception:
                log.exception("step failed")
            time.sleep(max(1, POLL_SECS - (time.time() - t0)))


def report() -> None:
    if not DB.exists():
        print("Kalshi paper trader has no data yet.")
        return
    con = sqlite3.connect(DB)
    print(f"\nKALSHI paper trading ({len(VARIANTS)} variants, ${BANKROLL:,.0f} each)\n")
    print(f"{'variant':26} {'P&L':>9} {'ret':>7} {'settled':>7} {'win%':>5} {'open':>5} {'avg edge':>8}")
    rows = []
    for v in VARIANTS:
        realized, n, wins, open_n, avg_edge = con.execute(
            "SELECT COALESCE(SUM(CASE WHEN status!='open' THEN payout-contracts*price-fee END),0),"
            " SUM(status!='open'), SUM(status!='open' AND payout>contracts*price+fee), SUM(status='open'),"
            " AVG(edge) FROM positions WHERE variant=?", (v.name,)).fetchone()
        rows.append((v.name, realized, n or 0, wins or 0, open_n or 0, avg_edge))
    for name, pnl, n, w, o, e in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:26} {pnl:+9.2f} {pnl / BANKROLL * 100:+6.1f}% {n:7} {(f'{w / n * 100:.0f}' if n else '-'):>5} "
              f"{o:5} {(f'{e:+.3f}' if e is not None else '-'):>8}")
    snaps = con.execute("SELECT p_n30,p_n240,p_t30,p_t240,(yes_bid+yes_ask)/2, result='yes' FROM snapshots"
                        " WHERE result IS NOT NULL").fetchall()
    if snaps:
        a = np.array(snaps, dtype=float)
        y = a[:, 5]
        brier = lambda col: float(np.mean((a[:, col] - y) ** 2))
        print(f"\nCalibration on {len(a)} settled quotes (Brier score, lower = better; model must beat market):")
        print("  market mid  {:.4f}".format(brier(4)))
        for i, k in enumerate(MODELS):
            print(f"  model {k:5} {brier(i):.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        DIR.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=DIR / "bot.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        Trader().run()
