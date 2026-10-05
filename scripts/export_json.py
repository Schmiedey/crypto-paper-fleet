"""Dump everything the web dashboard (docs/index.html) needs into one small JSON file.
Usage: python scripts/export_json.py OUT.json   (reads runs/*/*.sqlite)"""
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import os
RUNS = Path(os.environ.get("RUNS_DIR") or Path(__file__).resolve().parent.parent / "runs")


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
        if not db.exists() or (d / "retired").exists():
            continue
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
            "trades": rows(db, "select * from trades order by ts desc limit 100")}


def kalshi():
    db = RUNS / "kalshi" / "kalshi.sqlite"
    if not db.exists():
        return {}
    return {"positions": rows(db, "select * from positions order by id desc limit 600")}


def daytrader():
    db = RUNS / "daytrader" / "daytrader.sqlite"
    if not db.exists():
        return {}
    return {"account": rows(db, "select * from account"), "trades": rows(db, "select * from trades order by closed desc limit 300"),
            "equity": rows(db, "select * from equity order by ts")}


if __name__ == "__main__":
    data = {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "wallet": 10000,
            "bots": bots(), "portfolio": portfolio(), "kalshi": kalshi(), "daytrader": daytrader()}
    Path(sys.argv[1]).write_text(json.dumps(data, separators=(",", ":"), default=str))
