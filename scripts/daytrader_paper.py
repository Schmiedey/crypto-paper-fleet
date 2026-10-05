"""Paper day-trader: noise-area intraday momentum (Zarattini, Aziz & Barbon 2024) on QQQ and SPY.

Best day-trading result in research/daytrade.py: Nasdaq-100 2018-2026 +12.7%/yr, Sharpe 0.92 at 1 bp per side.
Every 30 minutes from 10:00 to 15:30 ET it compares price with a band around today's open whose width is
the average 14-day move at that time of day. Above the band: long; below: short (not in the -long variant).
Exit when price falls back through max(band, VWAP) (min for shorts), and always flat at the close.
Size = min(4, 2% / 14-day daily vol) x equity. Fills at the checkpoint price, 1 bp cost per side.
Data: Yahoo Finance (free, real time for these ETFs). State: runs/daytrader/daytrader.sqlite.
"""
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

DIR = Path(__file__).resolve().parent.parent / "runs" / "daytrader"
NY = ZoneInfo("America/New_York")
VARIANTS = {"NAM_QQQ": ("QQQ", False), "NAM_QQQ_long": ("QQQ", True)}  # SPY version stopped working after 2024
START_CASH, COST, TARGET_VOL, MAX_LEV = 10_000.0, 0.0001, 0.02, 4.0
CHECKS = [(h, m) for h in range(10, 16) for m in (0, 30)]  # 10:00 ... 15:30
log = logging.getLogger("daytrader")


def db():
    con = sqlite3.connect(DIR / "daytrader.sqlite")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS account (variant TEXT PRIMARY KEY, equity REAL, pos REAL, entry REAL, lev REAL, opened TEXT);
        CREATE TABLE IF NOT EXISTS trades (variant TEXT, symbol TEXT, side TEXT, opened TEXT, closed TEXT,
            entry REAL, exit REAL, lev REAL, pnl REAL, pnl_pct REAL, reason TEXT);
        CREATE TABLE IF NOT EXISTS equity (ts TEXT, variant TEXT, equity REAL);""")
    con.execute("DELETE FROM account WHERE variant NOT IN (%s)" % ",".join("?" * len(VARIANTS)), tuple(VARIANTS))
    for v in VARIANTS:
        con.execute("INSERT OR IGNORE INTO account VALUES (?,?,0,0,0,NULL)", (v, START_CASH))
    con.commit()
    return con


def bars(sym, interval, period):
    d = yf.download(sym, period=period, interval=interval, progress=False, auto_adjust=False, prepost=False)
    d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    d.index = d.index.tz_convert(NY) if d.index.tz else d.index.tz_localize(NY)
    return d.dropna()


def bands(sym, today):
    """Per-checkpoint sigma from the previous 14 sessions, plus the sizing leverage."""
    h = bars(sym, "30m", "60d")
    h = h[h.index.date < today]
    days = sorted(set(h.index.date))[-14:]
    sig = {}
    for hh, mm in CHECKS + [(16, 0)]:
        mv = []
        for d in days:
            day = h[h.index.date == d]
            if day.empty:
                continue
            o = day.Open.iloc[0]
            at = day[day.index + timedelta(minutes=30) <= datetime(d.year, d.month, d.day, hh, mm, tzinfo=NY)]
            if len(at):
                mv.append(abs(at.Close.iloc[-1] / o - 1))
        sig[(hh, mm)] = float(np.mean(mv)) if mv else np.nan
    daily = bars(sym, "1d", "3mo").Close
    daily = daily[daily.index.date < today]
    vol = daily.pct_change().tail(14).std()
    return sig, float(min(MAX_LEV, TARGET_VOL / vol)), float(daily.iloc[-1])


def close_pos(con, v, sym, price, now, reason):
    eq, pos, entry, lev, opened = con.execute("SELECT equity,pos,entry,lev,opened FROM account WHERE variant=?", (v,)).fetchone()
    if not pos:
        return
    r = lev * (pos * (price / entry - 1) - 2 * COST)
    pnl = eq * r
    con.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?)", (v, sym, "long" if pos > 0 else "short", opened,
                now.isoformat(), entry, price, lev, pnl, r * 100, reason))
    con.execute("UPDATE account SET equity=?, pos=0, entry=0, lev=0, opened=NULL WHERE variant=?", (eq + pnl, v))
    log.info("%-13s CLOSE %s %s @ %.2f  %+.2f%% (%s)", v, "long" if pos > 0 else "short", sym, price, r * 100, reason)


def open_pos(con, v, sym, side, price, lev, now):
    con.execute("UPDATE account SET pos=?, entry=?, lev=?, opened=? WHERE variant=?", (side, price, lev, now.isoformat(), v))
    log.info("%-13s OPEN  %s %s @ %.2f  lev %.2f", v, "long" if side > 0 else "short", sym, price, lev)


def session(con):
    now = datetime.now(NY)
    today = now.date()
    info = {s: bands(s, today) for s in {s for s, _ in VARIANTS.values()}}
    for hh, mm in CHECKS + [(15, 59)]:
        t = datetime(today.year, today.month, today.day, hh, mm, 20, tzinfo=NY)
        if datetime.now(NY) > t + timedelta(minutes=10):
            continue  # missed this checkpoint (started late)
        time.sleep(max(0, (t - datetime.now(NY)).total_seconds()))
        now = datetime.now(NY)
        for sym in info:
            m = bars(sym, "1m", "1d")
            m = m[m.index.date == today]
            if m.empty:
                continue
            sig, lev, prev = info[sym]
            o, px = m.Open.iloc[0], m.Close.iloc[-1]
            tp = (m.High + m.Low + m.Close) / 3
            vwap = float((tp * m.Volume).sum() / max(m.Volume.sum(), 1))
            for v, (vs, longonly) in VARIANTS.items():
                if vs != sym:
                    continue
                pos = con.execute("SELECT pos FROM account WHERE variant=?", (v,)).fetchone()[0]
                if (hh, mm) == (15, 59):
                    close_pos(con, v, sym, px, now, "market close")
                    continue
                s = sig[(hh, mm)]
                if np.isnan(s):
                    continue
                up, dn = max(o, prev) * (1 + s), min(o, prev) * (1 - s)
                if px > up:
                    want = 1
                elif px < dn and not longonly:
                    want = -1
                elif pos == 1 and px > max(up, vwap):
                    want = 1
                elif pos == -1 and px < min(dn, vwap):
                    want = -1
                else:
                    want = 0
                if want != pos:
                    if pos:
                        close_pos(con, v, sym, px, now, "stop" if want == 0 else "reverse")
                    if want:
                        open_pos(con, v, sym, want, px, lev, now)
        con.commit()
    for v, eq in con.execute("SELECT variant, equity FROM account").fetchall():
        con.execute("INSERT INTO equity VALUES (?,?,?)", (today.isoformat(), v, eq))
    con.commit()


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=DIR / "bot.log", level=logging.INFO, format="%(asctime)s %(message)s")
    con = db()
    while True:
        now = datetime.now(NY)
        trading_day = now.weekday() < 5
        if trading_day and now.time() < datetime.strptime("15:50", "%H:%M").time():
            opening = now.replace(hour=9, minute=59, second=0, microsecond=0)
            time.sleep(max(0, (opening - now).total_seconds()))
            try:
                if not bars("QQQ", "1m", "1d").pipe(lambda m: m[m.index.date == datetime.now(NY).date()]).empty:
                    session(con)  # market holidays have no bars and are skipped
            except Exception:
                log.exception("session error")
        nxt = (datetime.now(NY) + timedelta(days=1)).replace(hour=9, minute=50, second=0, microsecond=0)
        time.sleep(max(60, (nxt - datetime.now(NY)).total_seconds()))


if __name__ == "__main__":
    main()
