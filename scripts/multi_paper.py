"""Paper trader for the three non-fleet strategies from the October 2026 deep search (research/deep_search.py).

  IBS_QQQ        QQQ 'internal bar strength': buy when (close-low)/(high-low) < 0.1, sell when > 0.8.
                 Decided at 15:55 ET with the day's range so far, filled at that price. 5 bp per side.
  QQQ_2x_SMA200  2x QQQ while QQQ is above its 200-day average, otherwise T-bills. Leverage costs T-bill + 0.5%.
                 Decided at 15:55 ET. 5 bp per unit of turnover.
  BTC_MTF        Bitcoin long while above its 20, 50 and 100-day averages together, otherwise cash.
                 Decided at 00:05 UTC on the last completed daily Kraken candle. 38 bp per switch (Kraken Pro tier 3 taker).

Equity is marked at every decision, so it matches the backtest accounting. $10,000 each. The NAM Nasdaq
day trader lives in daytrader_paper.py; the dashboard blends the four. State: runs/multi/multi.sqlite.
"""
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

DIR = Path(os.environ.get("MULTI_DIR") or Path(__file__).resolve().parent.parent / "runs" / "multi")
NY = ZoneInfo("America/New_York")
START = 10_000.0
SLEEVES = ("IBS_QQQ", "QQQ_2x_SMA200", "BTC_MTF")
log = logging.getLogger("multi")


def db():
    con = sqlite3.connect(DIR / "multi.sqlite")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS account (sleeve TEXT PRIMARY KEY, equity REAL, pos REAL, last_price REAL, last_key TEXT, since TEXT);
        CREATE TABLE IF NOT EXISTS log (ts TEXT, sleeve TEXT, event TEXT, price REAL, pos_from REAL, pos_to REAL, equity REAL, note TEXT);
        CREATE TABLE IF NOT EXISTS equity (ts TEXT, sleeve TEXT, equity REAL);""")
    con.commit()
    return con


def tbill():
    try:
        d = yf.download("^IRX", period="10d", interval="1d", progress=False, auto_adjust=False)
        d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
        return float(d.Close.dropna().iloc[-1]) / 100
    except Exception:
        return 0.04


def qqq_state():
    """Latest price, today's high/low, and prior daily closes."""
    m = yf.download("QQQ", period="1d", interval="1m", progress=False, auto_adjust=False, prepost=False)
    m.columns = [c[0] if isinstance(c, tuple) else c for c in m.columns]
    m = m.dropna()
    dly = yf.download("QQQ", period="2y", interval="1d", progress=False, auto_adjust=False)
    dly.columns = [c[0] if isinstance(c, tuple) else c for c in dly.columns]
    today = m.index[-1].tz_convert(NY).date()
    closes = dly.Close.dropna()
    closes = closes[[i.date() < today for i in closes.index]]
    return float(m.Close.iloc[-1]), float(m.High.max()), float(m.Low.min()), closes, today


def kraken_btc():
    r = requests.get("https://api.kraken.com/0/public/OHLC", params={"pair": "BTC/USD", "interval": 1440}, timeout=30).json()
    k = [x for x in r["result"] if x != "last"][0]
    rows = r["result"][k]
    s = pd.Series([float(x[4]) for x in rows], pd.to_datetime([x[0] for x in rows], unit="s"))
    return s.iloc[:-1]  # drop the still-forming candle


def get(con, sleeve):
    return con.execute("SELECT equity,pos,last_price,last_key FROM account WHERE sleeve=?", (sleeve,)).fetchone()


def settle(con, sleeve, key, price, new_pos, cost_per_unit, carry, gross_mult, now, note=""):
    """Mark equity from the previous decision price, apply costs, then switch to the new position."""
    row = get(con, sleeve)
    if row is None:
        con.execute("INSERT INTO account VALUES (?,?,?,?,?,?)", (sleeve, START, new_pos, price, key, now.isoformat()))
        con.execute("INSERT INTO log VALUES (?,?,?,?,?,?,?,?)", (now.isoformat(), sleeve, "START", price, 0, new_pos, START, note))
        con.execute("INSERT INTO equity VALUES (?,?,?)", (now.isoformat(), sleeve, START))
        log.info("%-14s START pos %.0f @ %.2f", sleeve, new_pos, price)
        return
    eq, pos, last, last_key = row
    if last_key == key:
        return
    r = pos * gross_mult * (price / last - 1) + (1 - min(abs(pos), 1)) * carry["cash"] + carry.get("borrow", 0) * pos
    eq *= 1 + r - abs(new_pos - pos) * cost_per_unit
    con.execute("UPDATE account SET equity=?, pos=?, last_price=?, last_key=? WHERE sleeve=?", (eq, new_pos, price, key, sleeve))
    con.execute("INSERT INTO equity VALUES (?,?,?)", (now.isoformat(), sleeve, eq))
    if new_pos != pos:
        con.execute("INSERT INTO log VALUES (?,?,?,?,?,?,?,?)",
                    (now.isoformat(), sleeve, "ENTER" if new_pos else "EXIT", price, pos, new_pos, eq, note))
        log.info("%-14s %s @ %.2f equity %.2f (%s)", sleeve, "ENTER" if new_pos else "EXIT", price, eq, note)
    else:
        log.info("%-14s hold pos %.0f equity %.2f", sleeve, new_pos, eq)


def decide_stocks(con, now, qs):
    price, hi, lo, closes, today = qs
    key = today.isoformat()
    rf = tbill()
    daily_rf = rf / 252
    # IBS: sticky signal, so the previous position matters
    ibs = (price - lo) / (hi - lo) if hi > lo else 0.5
    prev = get(con, "IBS_QQQ")
    pos = prev[1] if prev else 0.0
    new = 1.0 if ibs < 0.1 else (0.0 if ibs > 0.8 else pos)
    settle(con, "IBS_QQQ", key, price, new, 0.0005, {"cash": daily_rf}, 1, now, f"ibs {ibs:.2f}")
    # 2x QQQ above SMA200
    sma = float(np.mean(np.r_[closes.values[-199:], price]))
    new2 = 1.0 if price > sma else 0.0
    settle(con, "QQQ_2x_SMA200", key, price, new2, 0.0005 * 2, {"cash": daily_rf, "borrow": -(rf + 0.005) / 252}, 2, now,
           f"price {price:.2f} vs SMA200 {sma:.2f}")
    con.commit()


def decide_btc(con, now):
    c = kraken_btc()
    key = str(c.index[-1].date())
    ok = c.iloc[-1] > c.rolling(20).mean().iloc[-1] and c.iloc[-1] > c.rolling(50).mean().iloc[-1] \
        and c.iloc[-1] > c.rolling(100).mean().iloc[-1]
    settle(con, "BTC_MTF", key, float(c.iloc[-1]), 1.0 if ok else 0.0, 0.0038, {"cash": tbill() / 365}, 1, now,
           f"close {c.iloc[-1]:.0f}")
    con.commit()


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=DIR / "bot.log", level=logging.INFO, format="%(asctime)s %(message)s")
    con = db()
    force = bool(os.environ.get("FORCE"))
    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            ny = now_utc.astimezone(NY)
            done = {r[0]: r[1] for r in con.execute("SELECT sleeve,last_key FROM account")}
            # stocks: weekdays from 15:55 ET until 20:00 ET (late starts use the last price, like a market-on-close order)
            if force or (ny.weekday() < 5 and (ny.hour, ny.minute) >= (15, 55) and ny.hour < 20):
                qs = qqq_state()
                if force or (qs[4] == ny.date() and (done.get("IBS_QQQ") != qs[4].isoformat()
                                                       or done.get("QQQ_2x_SMA200") != qs[4].isoformat())):
                    decide_stocks(con, now_utc, qs)
            # bitcoin: after 00:05 UTC, once per UTC day
            if force or (now_utc.hour, now_utc.minute) >= (0, 5):
                last = done.get("BTC_MTF")
                if force or last != str((now_utc - timedelta(days=1)).date()):
                    decide_btc(con, now_utc)
        except Exception:
            log.exception("tick error")
        if force:
            return
        time.sleep(120)


if __name__ == "__main__":
    main()
