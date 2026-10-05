"""Deep search, part 2: crypto funding-rate carry, crypto basket trend, and combinations of the leaders.

  carry   long spot + short perpetual (delta-neutral), collect funding while the 3-day average is positive.
          Binance USD-M funding history (public archive). 0.1% per leg per switch. Shown at 1x and 0.5x
          capital efficiency (0.5x = the short is fully cash-collateralised).
  basket  each of ~45 Binance coins (incl. LUNA/FTT, so not just survivors) held while above its n-day SMA,
          1/N of equity per coin, 0.25% per trade.
  combo   equal-risk blends of the best daily return streams from deep_search.py.

Usage: python research/deep_search2.py
"""
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import deep_search as ds

R = Path(__file__).resolve().parent
FUND = R / "funding"
CRYPTO_COST = 0.0025


def funding(coin):
    f = FUND / f"{coin}.feather"
    if f.exists():
        return pd.read_feather(f).set_index("date").rate
    FUND.mkdir(exist_ok=True)
    months = [(y, m) for y in range(2020, 2027) for m in range(1, 13) if (y, m) <= (2026, 9)]

    def get(ym):
        y, m = ym
        r = requests.get(f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{coin}USDT/"
                         f"{coin}USDT-fundingRate-{y}-{m:02d}.zip", timeout=60)
        if r.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            return pd.read_csv(z.open(z.namelist()[0]))

    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(get, months) if p is not None]
    df = pd.concat(parts)
    df["date"] = pd.to_datetime(df.calc_time, unit="ms", utc=True).dt.tz_localize(None)
    df = df.rename(columns={"last_funding_rate": "rate"})[["date", "rate"]].sort_values("date")
    df.reset_index(drop=True).to_feather(f)
    return df.set_index("date").rate


def carry(coins=("BTC", "ETH", "SOL", "XRP", "DOGE", "BNB"), eff=1.0, leg_cost=0.001):
    daily = pd.DataFrame({c: funding(c).resample("D").sum() for c in coins}).fillna(0)
    on = (daily.rolling(3).mean() > 0).shift(1).fillna(False).astype(float)  # decide on past funding only
    switch = on.diff().abs().fillna(on.iloc[0])
    per_coin = on * daily - switch * 2 * leg_cost  # two legs each switch
    return per_coin.mean(1) * eff, switch.sum(1).mean()


def basket_trend(n=50, start="2020-01-01"):
    files = sorted((R / "data").glob("*-1d.feather"))
    P = pd.DataFrame({f.stem.split("-")[0]: pd.read_feather(f).set_index("date").close for f in files})
    P.index = P.index.tz_localize(None)
    P = P[P.index >= start]
    r = P.pct_change()
    on = (P > P.rolling(n).mean()) & P.notna()
    avail = P.notna().sum(1).replace(0, np.nan)
    w = on.astype(float).div(avail, axis=0)
    held = w.shift(1).fillna(0)
    turn = (w - w.shift(1).fillna(0)).abs().sum(1).shift(1).fillna(0)
    net = (held * r.fillna(0)).sum(1) - turn * CRYPTO_COST
    ew = r.mean(1)  # equal-weight buy & hold of whatever is listed
    return net, ew, turn


def show(name, r, turn=None):
    out = {}
    for per, sl in (("all", slice(None)), ("2020-21", slice("2020", "2021")), ("2022+", slice("2022", None))):
        s = ds.stats(r[sl], turn)
        out.update({f"{per}_{k}": round(float(v), 3) for k, v in s.items() if k in ("cagr", "sharpe", "maxdd")})
    print(f"{name:45s}", out, flush=True)
    return out


if __name__ == "__main__":
    print("== funding-rate carry")
    for eff in (1.0, 0.5):
        c, sw = carry(eff=eff)
        show(f"carry 6 coins, capital efficiency {eff}x", c)
    for coin in ("BTC", "ETH"):
        c, _ = carry((coin,))
        show(f"carry {coin} only 1x", c)
    print("== crypto basket trend (1/N per coin, held above SMA)")
    for n in (20, 50, 100, 200):
        net, ew, turn = basket_trend(n)
        show(f"basket trend SMA{n}", net, turn)
    show("equal-weight basket buy&hold", ew)
