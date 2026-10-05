"""Free minute data for day-trading research, cached to research/intraday/.

Dukascopy (no account): US500, USTECH, XAUUSD, EURUSD, USDJPY 1-minute BID candles, 2018-01 onward.
Binance public archive: BTC ETH SOL XRP DOGE BNB vs USDT, 5-minute klines, 2020-01 onward.
Output: research/intraday/{SYMBOL}-{1m|5m}.feather with UTC timestamps.

Usage: python research/daytrade_data.py
"""
import io
import lzma
import struct
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent / "intraday"
DUKA = {"US500": ("USA500IDXUSD", 1e3), "USTECH": ("USATECHIDXUSD", 1e3), "XAUUSD": ("XAUUSD", 1e3),
        "EURUSD": ("EURUSD", 1e5), "USDJPY": ("USDJPY", 1e3)}
COINS = "BTC ETH SOL XRP DOGE BNB".split()
http = requests.Session()


def duka_day(inst, scale, d):
    url = f"https://datafeed.dukascopy.com/datafeed/{inst}/{d.year}/{d.month - 1:02d}/{d.day:02d}/BID_candles_min_1.bi5"
    for _ in range(12):
        try:
            r = http.get(url, timeout=30)
            if r.status_code == 429:
                time.sleep(20)
                continue
            if r.status_code == 404 or not r.content:
                return None
            raw = lzma.decompress(r.content)
            rows = [struct.unpack(">5if", raw[i:i + 24]) for i in range(0, len(raw), 24)]
            df = pd.DataFrame(rows, columns=["t", "open", "close", "low", "high", "volume"])
            df["date"] = pd.Timestamp(d, tz="UTC") + pd.to_timedelta(df.t, unit="s")
            for c in ("open", "close", "low", "high"):
                df[c] = df[c] / scale
            return df[["date", "open", "high", "low", "close", "volume"]]
        except Exception:
            continue
    return None


def duka(name, start=date(2018, 1, 1)):
    f = OUT / f"{name}-1m.feather"
    if f.exists():
        return
    inst, scale = DUKA[name]
    days = [start + timedelta(i) for i in range((date.today() - start).days) if (start + timedelta(i)).weekday() < 5]
    with ThreadPoolExecutor(3) as ex:
        parts = [p for p in ex.map(lambda d: duka_day(inst, scale, d), days) if p is not None and len(p)]
    df = pd.concat(parts).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df = df[df.volume > 0]  # flat minutes outside trading hours
    df.to_feather(f)
    print(name, len(df), df.date.min(), df.date.max(), flush=True)


def binance_month(sym, y, m):
    url = f"https://data.binance.vision/data/spot/monthly/klines/{sym}/5m/{sym}-5m-{y}-{m:02d}.zip"
    r = http.get(url, timeout=60)
    if r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=range(6))
    df.columns = ["t", "open", "high", "low", "close", "volume"]
    t = df.t.astype("int64")
    df["date"] = pd.to_datetime(t // 1000 if t.max() > 1e14 else t, unit="ms", utc=True)  # 2025+ files use microseconds
    return df[["date", "open", "high", "low", "close", "volume"]]


def binance(coin, start=(2020, 1)):
    f = OUT / f"{coin}-5m.feather"
    if f.exists():
        return
    sym = f"{coin}USDT"
    months = []
    y, m = start
    while (y, m) < (date.today().year, date.today().month):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(lambda ym: binance_month(sym, *ym), months) if p is not None]
    df = pd.concat(parts).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df.to_feather(f)
    print(coin, len(df), df.date.min(), df.date.max(), flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for c in COINS:
        binance(c)
    for n in DUKA:
        duka(n)
