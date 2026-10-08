"""Dump everything the web dashboard (docs/index.html) needs into one small JSON file.
Usage: python scripts/export_json.py OUT.json   (reads runs/*/*.sqlite)"""
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import os
import time
RUNS = Path(os.environ.get("RUNS_DIR") or Path(__file__).resolve().parent.parent / "runs")
STRATS = Path(__file__).resolve().parent.parent / "user_data" / "strategies"


def iso(s):
    return s.replace(" ", "T")[:19] + "Z" if s else None


def rows(db, sql, args=()):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, args)]
    finally:
        con.close()


def bots():
    out = {}
    for d in sorted(RUNS.iterdir()):
        db = d / "trades.sqlite"
        if not db.exists() or (d / "retired").exists() or not (STRATS / f"{d.name}.py").exists():
            continue  # dropped strategies keep their old database but are no longer shown
        cols = ("id,pair,is_open,open_date,close_date,open_rate,close_rate,amount,stake_amount,open_trade_value,"
                "fee_close,close_profit,close_profit_abs,exit_reason,enter_tag,stop_loss,max_rate,min_rate")
        try:
            op = rows(db, f"select {cols} from trades where is_open=1")
            cl = rows(db, f"select {cols} from trades where is_open=0 order by close_date desc limit 300")
            tot = rows(db, "select count(*) n, sum(close_profit_abs>0) w, coalesce(sum(close_profit_abs),0) p from trades where is_open=0")[0]
        except sqlite3.Error:
            continue
        for t in op + cl:
            t["open_date"], t["close_date"] = iso(t["open_date"]), iso(t["close_date"])
        out[d.name] = {"open": op, "closed": cl, "n_closed": tot["n"], "wins": tot["w"] or 0, "pnl": tot["p"]}
    return out


def portfolio():
    db = RUNS / "portfolio" / "portfolio.sqlite"
    if not db.exists():
        return {}
    return {"cash": rows(db, "select * from cash"), "holdings": rows(db, "select * from holdings where qty>0"),
            "equity": rows(db, "select * from equity order by ts desc limit 600"),
            "daily": rows(db, "select sleeve, d ts, usd from (select sleeve, substr(ts,1,10) d, usd, row_number() over "
                              "(partition by sleeve, substr(ts,1,10) order by ts desc) rn from equity) where rn=1 order by d"),
            "trades": rows(db, "select * from trades order by ts desc limit 100")}


def daytrader():
    db = RUNS / "daytrader" / "daytrader.sqlite"
    if not db.exists():
        return {}
    return {"account": rows(db, "select * from account"), "trades": rows(db, "select * from trades order by closed desc limit 300"),
            "equity": rows(db, "select * from equity order by ts")}


def agent():
    db = RUNS / "agent" / "agent.sqlite"
    if not db.exists():
        return {}
    return {"equity": rows(db, "select * from equity order by ts"), "trades": rows(db, "select * from trades order by ts desc limit 200"),
            "decisions": rows(db, "select id, ts, model, regime, view, weights, reasons, headlines, equity from decisions order by id desc limit 100")}


def multi():
    db = RUNS / "multi" / "multi.sqlite"
    if not db.exists():
        return {}
    return {"account": rows(db, "select * from account"), "log": rows(db, "select * from log order by ts desc limit 200"),
            "equity": rows(db, "select * from equity order by ts")}


def scorecard():
    """Inputs for docs/scorecard.html: the exposure-adjusted leaderboard plus BTC and QQQ buy & hold since the start.
    Prices come straight from Kraken (the fleet's proxy is down during the final save of a leg)."""
    out = {}
    try:
        import ccxt
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import fleet
        fleet.RUNS = RUNS
        bench = json.loads((RUNS / "benchmark.json").read_text())
        pairs = sorted(set(json.loads(fleet.BASE_CFG.read_text())["exchange"]["pair_whitelist"]) | set(bench["prices"]))
        now = {p: t["last"] for p, t in ccxt.kraken().fetch_tickers(pairs).items()}
        lb, bh = fleet.leaderboard(now)
        keys = ("name", "running", "ret", "closed", "open", "wins", "realized", "expo", "alpha", "t")
        out.update(started=bench["started"], basket_bh=bh, bots=[dict(zip(keys, r)) for r in lb],
                   btc={"start": bench["prices"].get("BTC/USD"), "last": now.get("BTC/USD")})
    except Exception as e:  # never let the dashboard export break a save
        out["error"] = f"leaderboard: {e}"
    cache = Path("/tmp/fleet_qqq.json")  # yfinance is slow; refresh hourly
    try:
        if not cache.exists() or time.time() - cache.stat().st_mtime > 3600:
            import yfinance as yf
            q = yf.download("QQQ", start="2026-10-01", interval="1d", progress=False, auto_adjust=False)
            q.columns = [c[0] if isinstance(c, tuple) else c for c in q.columns]
            cache.write_text(json.dumps({str(i.date()): float(v) for i, v in q.Close.dropna().items()}))
        out["qqq"] = json.loads(cache.read_text())
    except Exception as e:
        out["error"] = (out.get("error", "") + f" qqq: {e}").strip()
    return out


if __name__ == "__main__":
    data = {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "wallet": 10000,
            "bots": bots(), "portfolio": portfolio(), "daytrader": daytrader(), "multi": multi(), "agent": agent(), "scorecard": scorecard()}
    Path(sys.argv[1]).write_text(json.dumps(data, separators=(",", ":"), default=str))
