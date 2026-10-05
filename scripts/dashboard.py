"""Local live dashboard for the paper-trading fleet.

  python scripts/dashboard.py        serve http://127.0.0.1:8787 (fleet.py starts it)

Reads the bots' SQLite files directly; nothing leaves the machine. Every 5 minutes it records a
snapshot of the headline returns so the performance chart builds up history.
"""
import asyncio
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import fleet  # noqa: E402
import kalshi_paper as kp  # noqa: E402

PORT = 8787
RUNS = fleet.RUNS
HIST = RUNS / "dashboard" / "history.sqlite"
DEPLOYED = ROOT / "research" / "bt" / "deployed.txt"
PORTFOLIO_DB = RUNS / "portfolio" / "portfolio.sqlite"
PORTFOLIO_CASH = 10000.0
CACHE_SECS = 20
SNAPSHOT_SECS = 300

_cache: dict = {"at": 0.0, "state": None}


def _q(db: Path, sql: str, args=()) -> list:
    if not db.exists():
        return []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return con.execute(sql, args).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        con.close()


_hourly: dict = {"at": 0.0, "closes": {}}


def market_since(start_ms: int, now: dict[str, float]) -> float | None:
    """Equal-weight % move of the whitelist pairs since a bot started (Kraken hourly closes, cached 10 min)."""
    if time.time() - _hourly["at"] > 600:
        import ccxt
        ex = ccxt.kraken({"urls": {"api": {"public": f"http://127.0.0.1:{fleet.PROXY_PORT}"}}})
        _hourly["closes"] = {p: ex.fetch_ohlcv(p, "1h", limit=720) for p in now}
        _hourly["at"] = time.time()
    moves = []
    for pair, bars in _hourly["closes"].items():
        start = next((b[1] for b in bars if b[0] >= start_ms - 3_600_000), None)  # open of the start hour
        if start and now.get(pair):
            moves.append(now[pair] / start - 1)
    return sum(moves) / len(moves) * 100 if moves else None


def bot_start_ms(name: str) -> int | None:
    """When the bot first started: the first timestamp in its log (local time)."""
    log = RUNS / name / "bot.log"
    if not log.exists():
        return None
    with open(log) as fh:
        first = fh.readline()[:19]
    try:
        return int(datetime.strptime(first, "%Y-%m-%d %H:%M:%S").astimezone().timestamp() * 1000)
    except ValueError:
        return None


def crypto_bots() -> tuple[list[dict], float, dict]:
    rows, bh = fleet.leaderboard()
    now = fleet.prices()
    deployed = set(DEPLOYED.read_text().split()) if DEPLOYED.exists() else set()
    bots = []
    for name, running, ret, closed, opened, wins, _ in rows:
        if (RUNS / name / "retired").exists():
            continue
        start = bot_start_ms(name)
        mkt = market_since(start, now) if start else None
        bots.append({"name": name, "group": "dip" if name in deployed else "legacy", "running": running,
                     "ret": ret, "mkt": mkt, "vs_bh": ret - mkt if mkt is not None else None,
                     "started": start, "closed": closed, "open": opened,
                     "win": wins / closed * 100 if closed else None})
    started = json.loads((RUNS / "benchmark.json").read_text())["started"]
    return bots, bh, {"started": started}


def recent_trades(names: list[str], limit: int = 25) -> list[dict]:
    events = []
    for n in names:
        for pair, od, cd, is_open, cp, stake in _q(
                RUNS / n / "trades.sqlite",
                "SELECT pair, open_date, close_date, is_open, close_profit, stake_amount FROM trades ORDER BY id DESC LIMIT 6"):
            events.append({"ts": od, "kind": "crypto", "who": n, "what": f"bought {pair}", "detail": f"${stake:,.0f}"})
            if not is_open and cd:
                events.append({"ts": cd, "kind": "crypto", "who": n, "what": f"sold {pair}",
                               "detail": f"{(cp or 0) * 100:+.2f}%", "pnl": cp})
    return events


def portfolio() -> dict:
    sleeves = []
    latest = _q(PORTFOLIO_DB, "SELECT sleeve, usd FROM equity WHERE ts=(SELECT MAX(ts) FROM equity)")
    counts = dict(_q(PORTFOLIO_DB, "SELECT sleeve, COUNT(*) FROM trades GROUP BY sleeve"))
    last_px = {(s, c): p for s, c, p in _q(PORTFOLIO_DB, "SELECT sleeve, coin, price FROM trades ORDER BY ts")}
    for sleeve, usd in sorted(latest, key=lambda r: -r[1]):
        hold = _q(PORTFOLIO_DB, "SELECT coin, qty FROM holdings WHERE sleeve=? AND qty>1e-12", (sleeve,))
        vals = sorted(((c, q * last_px.get((sleeve, c), 0)) for c, q in hold), key=lambda x: -x[1])
        total = sum(v for _, v in vals) or 1
        sleeves.append({"name": sleeve, "equity": usd, "ret": (usd / PORTFOLIO_CASH - 1) * 100,
                        "trades": counts.get(sleeve, 0),
                        "holdings": [{"coin": c, "w": v / usd * 100} for c, v in vals if v / total > 0.01],
                        "cash_w": max(0.0, (usd - sum(v for _, v in vals)) / usd * 100)})
    start = _q(PORTFOLIO_DB, "SELECT v FROM meta WHERE k='btc_start'")
    last = _q(PORTFOLIO_DB, "SELECT v FROM meta WHERE k='last_rebalance'")
    return {"sleeves": sleeves, "btc_start": json.loads(start[0][0]) if start else None,
            "last_rebalance": last[0][0] if last else None}


def kalshi() -> dict:
    db = kp.DB
    variants = []
    for v in kp.VARIANTS:
        r = _q(db, "SELECT COALESCE(SUM(CASE WHEN status!='open' THEN payout-contracts*price-fee END),0),"
                   " SUM(status!='open'), SUM(status!='open' AND payout>contracts*price+fee), SUM(status='open')"
                   " FROM positions WHERE variant=?", (v.name,))
        pnl, n, w, o = r[0] if r else (0, 0, 0, 0)
        variants.append({"name": v.name, "pnl": pnl or 0.0, "settled": n or 0, "win": (w or 0) / n * 100 if n else None,
                         "open": o or 0})
    variants.sort(key=lambda x: -x["pnl"])
    snaps = _q(db, "SELECT p_n30, p_n240, p_t30, p_t240, (yes_bid+yes_ask)/2, result='yes' FROM snapshots WHERE result IS NOT NULL")
    brier = []
    if snaps:
        cols = list(zip(*snaps))
        y = cols[5]
        score = lambda c: sum((a - b) ** 2 for a, b in zip(c, y)) / len(y)
        brier = [{"name": "market price", "score": score(cols[4]), "market": True}] + \
                [{"name": f"model {k}", "score": score(cols[i]), "market": False} for i, k in enumerate(kp.MODELS)]
    events = [{"ts": ts, "kind": "kalshi", "who": var, "what": f"bought {side.upper()} {tkr}",
               "detail": f"{int(n)} @ {price:.2f}" + (f" → {res}" if res else "")}
              for var, tkr, side, n, price, ts, res in _q(
                  db, "SELECT variant, ticker, side, contracts, price, opened_at, result FROM positions ORDER BY id DESC LIMIT 12")]
    return {"variants": variants, "bankroll": kp.BANKROLL, "brier": brier, "brier_n": len(snaps), "events": events}


def build_state() -> dict:
    bots, bh, meta = crypto_bots()
    port = portfolio()
    kal = kalshi()
    dip = [b["ret"] for b in bots if b["group"] == "dip"]
    blend = next((s for s in port["sleeves"] if s["name"] == "blend_mom_don_btc"), None)
    btc_now = fleet.prices().get("BTC/USD")
    btc_ret = (btc_now / port["btc_start"][1] - 1) * 100 if port["btc_start"] and btc_now else None
    dip_avg = sum(dip) / len(dip) if dip else None
    trend = blend["ret"] if blend else None
    combo = 0.3 * trend + 0.7 * dip_avg if trend is not None and dip_avg is not None else None
    procs = sum(1 for p in RUNS.glob("*/bot.pid") if not (p.parent / "retired").exists() and fleet._pid(p))
    proxy = None
    try:
        import urllib.request
        proxy = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{fleet.PROXY_PORT}/stats", timeout=3).read())
    except Exception:
        pass
    events = recent_trades([b["name"] for b in bots]) + kal["events"]
    events.sort(key=lambda e: e["ts"] or "", reverse=True)
    return {
        "now": datetime.now(timezone.utc).isoformat(),
        "headline": {"combo": combo, "dip": dip_avg, "trend": trend, "btc": btc_ret, "bh14": bh,
                     "kalshi_pnl": sum(v["pnl"] for v in kal["variants"]),
                     "open_positions": sum(b["open"] for b in bots)},
        "fleet": {"processes": procs, "started": meta["started"], "proxy": proxy},
        "bots": bots, "portfolio": port, "kalshi": kal, "events": events[:30],
        "history": history(),
    }


def history() -> dict:
    rows = _q(HIST, "SELECT ts, series, value FROM snap ORDER BY ts")
    out: dict[str, list] = {}
    for ts, series, value in rows:
        out.setdefault(series, []).append([ts, value])
    return out


def record(state: dict) -> None:
    HIST.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(HIST)
    con.execute("CREATE TABLE IF NOT EXISTS snap (ts TEXT, series TEXT, value REAL)")
    h = state["headline"]
    rows = [(state["now"], k, h[k]) for k in ("combo", "dip", "trend", "btc") if h[k] is not None]
    con.executemany("INSERT INTO snap VALUES (?,?,?)", rows)
    con.commit()
    con.close()


async def get_state() -> dict:
    if time.time() - _cache["at"] > CACHE_SECS or _cache["state"] is None:
        _cache["state"] = await asyncio.get_running_loop().run_in_executor(None, build_state)
        _cache["at"] = time.time()
    return _cache["state"]


async def api_state(_: web.Request) -> web.Response:
    return web.json_response(await get_state())


async def index(_: web.Request) -> web.Response:
    return web.FileResponse(ROOT / "scripts" / "dashboard.html")


async def snapshot_loop(_: web.Application):
    async def loop():
        while True:
            try:
                _cache["at"] = 0
                state = await get_state()
                await asyncio.get_running_loop().run_in_executor(None, record, state)
            except Exception as e:  # keep serving even if a snapshot fails
                print("snapshot failed:", e, flush=True)
            await asyncio.sleep(SNAPSHOT_SECS)
    task = asyncio.create_task(loop())
    yield
    task.cancel()


if __name__ == "__main__":
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/state", api_state)
    app.cleanup_ctx.append(snapshot_loop)
    web.run_app(app, host="127.0.0.1", port=PORT, print=None)
