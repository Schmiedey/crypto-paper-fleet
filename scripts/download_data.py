"""Seed recent candle history for the self-learning (FreqAI) bots.

Kraken's API only returns the latest 720 candles, so FreqAI's training window is seeded from
Binance's public archive (data.binance.vision, reachable from the US) and written in Freqtrade's
feather format as Kraken /USD pairs. Live bots then append fresh Kraken candles on top.

Usage: python scripts/download_data.py [DAYS]   (default 120)
"""
import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "user_data" / "data" / "kraken"
COINS = "BTC ETH SOL XRP DOGE ADA AVAX LINK BNB LTC SHIB PEPE BONK WIF".split()
BASE_TF = "5m"
RESAMPLE = {"15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}
URL = "https://data.binance.vision/data/spot/{kind}/klines/{sym}/{tf}/{sym}-{tf}-{period}.zip"


def get(sym: str, kind: str, period: str) -> bytes | None:
    r = requests.get(URL.format(kind=kind, sym=sym, tf=BASE_TF, period=period), timeout=60)
    return r.content if r.status_code == 200 else None


def blobs(sym: str, start: date):
    """Monthly files where published, falling back to daily files (monthly lags a few days after month end)."""
    today = date.today()
    d = start.replace(day=1)
    while d < today:
        nxt = (d + timedelta(days=32)).replace(day=1)
        if nxt <= today and (b := get(sym, "monthly", d.strftime("%Y-%m"))):
            yield b
        else:
            day = d
            while day < min(nxt, today):
                if b := get(sym, "daily", day.isoformat()):
                    yield b
                day += timedelta(days=1)
        d = nxt


def parse(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None, usecols=range(6))
    if not str(df.iloc[0, 0]).isdigit():  # some files have a header row
        df = df.iloc[1:]
    df = df.astype(float)
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    ts = df["date"].astype("int64")
    df["date"] = pd.to_datetime(ts, unit="us" if ts.iloc[0] > 10**14 else "ms", utc=True)  # archive moved to µs in 2025
    return df


def build(coin: str, days: int) -> str:
    start = date.today() - timedelta(days=days)
    frames = [parse(b) for b in blobs(f"{coin}USDT", start)]
    if not frames:
        return f"{coin}: no data"
    df = pd.concat(frames).drop_duplicates("date").sort_values("date")
    df = df[df["date"] >= pd.Timestamp(start, tz="UTC")].reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_feather(OUT / f"{coin}_USD-{BASE_TF}.feather")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for tf, rule in RESAMPLE.items():
        r = df.set_index("date").resample(rule).agg(agg).dropna().reset_index()
        r.to_feather(OUT / f"{coin}_USD-{tf}.feather")
    gaps = int((df["date"].diff() > pd.Timedelta(BASE_TF)).sum())
    return f"{coin}: {len(df):>6} candles {df.date.iloc[0]:%Y-%m-%d} -> {df.date.iloc[-1]:%m-%d %H:%M}  gaps={gaps}"


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    with ThreadPoolExecutor(7) as pool:
        for line in pool.map(lambda c: build(c, days), COINS):
            print(line)
