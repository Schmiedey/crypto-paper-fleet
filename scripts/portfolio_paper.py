"""Live paper trading for the portfolio strategies that survived research/systematic.py.

Once a day, just after the 00:00 UTC daily close, each sleeve computes target weights from completed
Binance daily candles (exactly the research code) and rebalances its own $10k paper account at
live Kraken bid/ask prices with Kraken's 0.38% taker fee (Pro tier 3). Equity is marked to market hourly.

  python scripts/portfolio_paper.py          run (fleet.py starts it)
  python scripts/portfolio_paper.py report   equity of each sleeve vs. BTC buy & hold
  python scripts/portfolio_paper.py once     rebalance now (for testing)
"""
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
import systematic as s  # noqa: E402  (signal functions shared with the research)

DIR = ROOT / "runs" / "portfolio"
DB = DIR / "portfolio.sqlite"
PROXY = "http://127.0.0.1:8899"
START_CASH = 10000.0
FEE = 0.0038  # Kraken Pro tier 3 taker ($10k+ 30-day volume or $20k held); smaller accounts pay 0.60-0.80%
MIN_TRADE = 25.0  # skip rebalancing trades smaller than this many dollars
COINS = ("BTC ETH BNB SOL XRP ADA DOGE AVAX LINK DOT LTC TRX ATOM ETC XLM BCH FIL NEAR UNI AAVE ALGO "
         "ICP APT ARB OP SUI SHIB PEPE INJ FET HBAR SEI TIA WIF BONK FLOKI SAND MANA AXS XTZ CRV VET").split()
log = logging.getLogger("portfolio")


def momentum7(close, uni, k=5, look=14, btc_sma=100):
    """Momentum rotation split into 7 weekday sleeves (removes rebalance-day luck)."""
    score = close.pct_change(look).where(uni)
    top = score.rank(axis=1, ascending=False) <= k
    w = top.astype(float).div(top.sum(axis=1).clip(lower=1), axis=0)
    w = w.mul(close["BTC"] > close["BTC"].rolling(btc_sma).mean(), axis=0)
    sleeves = []
    for p in range(7):
        keep = (np.arange(len(w)) + p) % 7 == 0
        sleeves.append(w.where(pd.Series(keep, index=w.index), np.nan).ffill().fillna(0))
    return sum(sleeves) / 7


def btc_sma(close, n=50):
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    w["BTC"] = (close["BTC"] > close["BTC"].rolling(n).mean()).astype(float)
    return w


def vol_trend_btc(close, target=0.4, win=60, n=50):
    """BTC above its 50-day average, sized so annualized volatility is ~40%."""
    vol = close["BTC"].pct_change().rolling(win).std() * np.sqrt(365)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    w["BTC"] = (target / vol).clip(upper=1) * (close["BTC"] > close["BTC"].rolling(n).mean())
    return w


SLEEVES = {
    "btc_voltarget_trend": lambda c, u10, u20: vol_trend_btc(c),
    "btc_trend_sma50": lambda c, u10, u20: btc_sma(c, 50),
    "donchian_20_10_top10": lambda c, u10, u20: s.donchian(c, u10, 20, 10),
    "momentum7_top5": lambda c, u10, u20: momentum7(c, u20, k=5),
    "momentum7_top3": lambda c, u10, u20: momentum7(c, u20, k=3),
    "tsmom_voltarget": lambda c, u10, u20: s.tsmom_vol(c, u10, 30, 0.8),
    "blend_mom_don_btc": lambda c, u10, u20: (momentum7(c, u20) + s.donchian(c, u10, 20, 10) + btc_sma(c, 50)) / 3,
}


def daily_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Completed daily candles (close, dollar volume) from Binance's public market-data API."""
    closes, qvols = {}, {}
    for c in COINS:
        r = requests.get("https://data-api.binance.vision/api/v3/klines",
                         params={"symbol": f"{c}USDT", "interval": "1d", "limit": 400}, timeout=20)
        if r.status_code != 200:
            continue
        k = pd.DataFrame(r.json()).iloc[:-1]  # drop today's unfinished candle
        idx = pd.to_datetime(k[0], unit="ms", utc=True)
        closes[c] = pd.Series(k[4].astype(float).values, index=idx)
        qvols[c] = pd.Series(k[7].astype(float).values, index=idx)
    return pd.DataFrame(closes).sort_index(), pd.DataFrame(qvols).sort_index()


def targets() -> dict[str, dict[str, float]]:
    close, qvol = daily_panel()
    u10, u20 = s.universe(close, qvol, 10), s.universe(close, qvol, 20)
    out = {}
    for name, fn in SLEEVES.items():
        w = fn(close, u10, u20).iloc[-1].fillna(0)
        out[name] = {c: float(x) for c, x in w.items() if x > 1e-6}
    return out


class Book:
    def __init__(self):
        DIR.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(DB)
        self.con.executescript("""
            CREATE TABLE IF NOT EXISTS holdings (sleeve TEXT, coin TEXT, qty REAL, PRIMARY KEY (sleeve, coin));
            CREATE TABLE IF NOT EXISTS cash (sleeve TEXT PRIMARY KEY, usd REAL);
            CREATE TABLE IF NOT EXISTS trades (ts TEXT, sleeve TEXT, coin TEXT, side TEXT, qty REAL, price REAL, fee REAL);
            CREATE TABLE IF NOT EXISTS equity (ts TEXT, sleeve TEXT, usd REAL);
            CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
        """)
        for name in SLEEVES:
            self.con.execute("INSERT OR IGNORE INTO cash VALUES (?, ?)", (name, START_CASH))
        self.con.commit()
        self.ex = ccxt.kraken({"urls": {"api": {"public": PROXY}}, "timeout": 20000})
        self.ex.load_markets()

    def quotes(self) -> dict[str, tuple[float, float]]:
        syms = [f"{c}/USD" for c in COINS if f"{c}/USD" in self.ex.markets]
        t = self.ex.fetch_tickers(syms)
        return {s_.split("/")[0]: (v["bid"], v["ask"]) for s_, v in t.items() if v.get("bid") and v.get("ask")}

    def positions(self, sleeve: str) -> tuple[float, dict[str, float]]:
        cash = self.con.execute("SELECT usd FROM cash WHERE sleeve=?", (sleeve,)).fetchone()[0]
        hold = dict(self.con.execute("SELECT coin, qty FROM holdings WHERE sleeve=? AND qty>0", (sleeve,)).fetchall())
        return cash, hold

    def equity(self, sleeve: str, q: dict) -> float:
        cash, hold = self.positions(sleeve)
        return cash + sum(qty * q[c][0] for c, qty in hold.items() if c in q)  # value at the bid

    def trade(self, sleeve, coin, usd, q, now):
        bid, ask = q[coin]
        cash, hold = self.positions(sleeve)
        if usd > 0:
            usd = min(usd, cash / (1 + FEE))
            qty, px = usd / ask, ask
            cash -= usd * (1 + FEE)
            fee = usd * FEE
        else:
            qty = min(-usd / bid, hold.get(coin, 0))
            px = bid
            cash += qty * bid * (1 - FEE)
            fee, qty = qty * bid * FEE, -qty
        if abs(qty) * px < 1:
            return
        self.con.execute("INSERT INTO holdings VALUES (?,?,?) ON CONFLICT(sleeve,coin) DO UPDATE SET qty=qty+excluded.qty",
                         (sleeve, coin, qty))
        self.con.execute("UPDATE cash SET usd=? WHERE sleeve=?", (cash, sleeve))
        self.con.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?)",
                         (now, sleeve, coin, "buy" if qty > 0 else "sell", abs(qty), px, fee))

    def rebalance(self) -> None:
        tgt, q = targets(), self.quotes()
        now = datetime.now(timezone.utc).isoformat()
        for sleeve, weights in tgt.items():
            eq = self.equity(sleeve, q)
            _, hold = self.positions(sleeve)
            want = {c: w * eq for c, w in weights.items() if c in q}
            have = {c: qty * q[c][0] for c, qty in hold.items() if c in q}
            deltas = {c: want.get(c, 0) - have.get(c, 0) for c in set(want) | set(have)}
            for c, d in sorted(deltas.items(), key=lambda kv: kv[1]):  # sells first to free cash
                if abs(d) >= MIN_TRADE:
                    self.trade(sleeve, c, d, q, now)
            log.info("%-22s equity $%9.2f  targets %s", sleeve, eq,
                     " ".join(f"{c}:{w:.0%}" for c, w in sorted(weights.items(), key=lambda kv: -kv[1])) or "cash")
        self.con.execute("INSERT OR REPLACE INTO meta VALUES ('last_rebalance', ?)", (now[:10],))
        if not self.con.execute("SELECT v FROM meta WHERE k='btc_start'").fetchone():
            self.con.execute("INSERT INTO meta VALUES ('btc_start', ?)", (json.dumps([now, q["BTC"][1]]),))
        self.con.commit()

    def mark(self) -> None:
        q = self.quotes()
        now = datetime.now(timezone.utc).isoformat()
        for sleeve in SLEEVES:
            self.con.execute("INSERT INTO equity VALUES (?,?,?)", (now, sleeve, self.equity(sleeve, q)))
        self.con.commit()

    def run(self) -> None:
        log.info("portfolio paper trader started with %d sleeves", len(SLEEVES))
        while True:
            try:
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                last = self.con.execute("SELECT v FROM meta WHERE k='last_rebalance'").fetchone()
                # Daily candle closes at 00:00 UTC; wait 5 minutes for the data feed to publish it.
                if (not last or last[0] < today) and datetime.now(timezone.utc).minute >= 5:
                    self.rebalance()
                self.mark()
            except Exception:
                log.exception("cycle failed")
            time.sleep(3600 - time.time() % 3600 + 330)  # wake at hh:05:30


def report() -> None:
    if not DB.exists():
        print("Portfolio paper trader has no data yet.")
        return
    con = sqlite3.connect(DB)
    start = con.execute("SELECT v FROM meta WHERE k='btc_start'").fetchone()
    print(f"\nPORTFOLIO strategies (research survivors, ${START_CASH:,.0f} each, daily rebalance)\n")
    rows = con.execute("SELECT sleeve, usd, ts FROM equity WHERE ts=(SELECT MAX(ts) FROM equity)").fetchall()
    if start:
        ts0, btc0 = json.loads(start[0])
        btc_now = ccxt.kraken({"urls": {"api": {"public": PROXY}}}).fetch_ticker("BTC/USD")["bid"]
        print(f"since {ts0[:16]} UTC  |  BTC buy & hold: {btc_now / btc0 - 1:+.2%}")
    trades = dict(con.execute("SELECT sleeve, COUNT(*) FROM trades GROUP BY sleeve").fetchall())
    for sleeve, usd, _ in sorted(rows, key=lambda r: -r[1]):
        print(f"  {sleeve:24} ${usd:10,.2f}  {usd / START_CASH - 1:+7.2%}  trades {trades.get(sleeve, 0)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "report":
        report()
    else:
        DIR.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=DIR / "bot.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        book = Book()
        if cmd == "once":
            book.rebalance()
            book.mark()
        else:
            book.run()
