"""Day-trading research: published intraday strategies on stock indices, gold, forex and crypto.

Every position is closed by the end of its session, so these are true day trades.
Parameters come from the papers and are NOT tuned here, so the later years act as out-of-sample:
  NAM  noise-area intraday momentum (Zarattini, Aziz & Barbon 2024, "Beat the Market"):
       bands = open * (1 +/- 14-day avg move at that time of day); check every 30 min;
       trailing stop = band or VWAP; size = min(4, 2% target daily vol / 14-day vol). Long and short.
  ORB  5-minute opening-range breakout (Zarattini, Barbon & Aziz 2023):
       trade in the direction of the first 5-min candle, stop at its other end, target 10R,
       risk 1% of equity per trade, leverage cap 4.
  IM   intraday momentum (Gao, Han, Li & Zhou 2018): sign of (prev close -> first 30 min)
       decides the direction of the last 30 minutes, unlevered.
Costs are charged per side on notional (see COST). Data: research/daytrade_data.py.

Usage: python research/daytrade.py [SYMBOL ...]
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
D = Path(__file__).resolve().parent / "intraday"

# symbol: (file, tz, session start, session end, bar minutes, cost per side)
MARKETS = {
    "US500":    ("US500-1m", "America/New_York", "09:30", "16:00", 1, 0.0001),
    "USTECH":   ("USTECH-1m", "America/New_York", "09:30", "16:00", 1, 0.0001),
    "XAUUSD":   ("XAUUSD-1m", "America/New_York", "09:30", "16:00", 1, 0.0002),
    "EURUSD":   ("EURUSD-1m", "Europe/London", "08:00", "16:00", 1, 0.0001),
    "USDJPY":   ("USDJPY-1m", "Europe/London", "08:00", "16:00", 1, 0.0001),
    **{f"{c}": (f"{c}-5m", "UTC", "00:00", "23:59", 5, 0.0005) for c in "BTC ETH SOL XRP DOGE BNB".split()},
    **{f"{c}-NY": (f"{c}-5m", "America/New_York", "09:30", "16:00", 5, 0.0005) for c in "BTC ETH SOL".split()},
}


def grid(sym):
    """Session bars as 2-D arrays (days x bars-of-session): open, high, low, close, volume."""
    f, tz, s, e, m, _ = MARKETS[sym]
    df = pd.read_feather(D / f"{f}.feather")
    t = df.date.dt.tz_convert(tz)
    sh, sm = map(int, s.split(":"))
    eh, em = map(int, e.split(":"))
    mins = t.dt.hour * 60 + t.dt.minute
    lo, hi = sh * 60 + sm, eh * 60 + em + (1 if e == "23:59" else 0)
    keep = (mins >= lo) & (mins < hi)
    df = df[keep].copy()
    df["day"] = t[keep].dt.date
    df["k"] = ((mins[keep] - lo) // m).astype(int)
    T = (hi - lo) // m
    out = {}
    for c in ("open", "high", "low", "close", "volume"):
        p = df.pivot_table(index="day", columns="k", values=c, aggfunc="first").reindex(columns=range(T))
        out[c] = p
    have = out["close"].notna().sum(axis=1)
    good = have >= 0.8 * T
    for c in out:
        out[c] = out[c][good]
    cl = out["close"].ffill(axis=1).bfill(axis=1)
    op = out["open"].fillna(cl.shift(1, axis=1)).fillna(cl)
    hi_ = out["high"].fillna(cl)
    lo_ = out["low"].fillna(cl)
    vol = out["volume"].fillna(0)
    days = pd.to_datetime(cl.index)
    return days, op.values, hi_.values, lo_.values, cl.values, vol.values, m


def nam(days, O, H, L, C, V, m, cost, longonly=False):
    nd, T = C.shape
    o = O[:, 0]
    prev = np.r_[np.nan, C[:-1, -1]]
    move = np.abs(C / o[:, None] - 1)
    sig = pd.DataFrame(move).rolling(14).mean().shift(1).values
    up = np.maximum(o, np.nan_to_num(prev, nan=o))[:, None] * (1 + sig)
    dn = np.minimum(o, np.nan_to_num(prev, nan=o))[:, None] * (1 - sig)
    tp = (H + L + C) / 3
    vw = np.cumsum(tp * V, 1) / np.maximum(np.cumsum(V, 1), 1e-12)
    vw = np.where(np.cumsum(V, 1) > 0, vw, C)
    dret = pd.Series(C[:, -1]).pct_change()
    vol14 = dret.rolling(14).std().shift(1).values
    lev = np.minimum(4, 0.02 / vol14)
    K = 30 // m
    chk = list(range(K - 1, T, K))
    if chk[-1] != T - 1:
        chk.append(T - 1)
    rets = np.zeros(nd)
    trades = np.zeros(nd)
    for d in range(nd):
        if np.isnan(sig[d, 0]) or np.isnan(lev[d]):
            continue
        pos, r, n = 0, 0.0, 0
        for i, k in enumerate(chk):
            if pos:
                r += pos * (C[d, k] / C[d, chk[i - 1]] - 1)
            if k == T - 1:
                break
            c = C[d, k]
            if c > up[d, k]:
                want = 1
            elif c < dn[d, k] and not longonly:
                want = -1
            elif pos == 1 and c > max(up[d, k], vw[d, k]):
                want = 1
            elif pos == -1 and c < min(dn[d, k], vw[d, k]):
                want = -1
            else:
                want = 0
            if want != pos:
                r -= abs(want - pos) * cost
                n += want != 0
                pos = want
        if pos:
            r -= cost
        rets[d] = lev[d] * r
        trades[d] = n
    return pd.Series(rets, days), trades


def orb(days, O, H, L, C, V, m, cost, longonly=False):
    nd, T = C.shape
    b = 5 // m  # bars in the opening range
    o, c = O[:, 0], C[:, b - 1]
    h, lo = H[:, :b].max(1), L[:, :b].min(1)
    rets = np.zeros(nd)
    trades = np.zeros(nd)
    for d in range(nd):
        if c[d] == o[d] or (longonly and c[d] < o[d]):
            continue
        side = 1 if c[d] > o[d] else -1
        entry = c[d]
        stop = lo[d] if side == 1 else h[d]
        R = abs(entry - stop)
        if R <= 0:
            continue
        target = entry + side * 10 * R
        exitp = C[d, -1]
        for k in range(b, T):
            if side == 1:
                if L[d, k] <= stop:
                    exitp = min(O[d, k], stop)
                    break
                if H[d, k] >= target:
                    exitp = target
                    break
            else:
                if H[d, k] >= stop:
                    exitp = max(O[d, k], stop)
                    break
                if L[d, k] <= target:
                    exitp = target
                    break
        lev = min(4, 0.01 / (R / entry))
        rets[d] = lev * (side * (exitp / entry - 1) - 2 * cost)
        trades[d] = 1
    return pd.Series(rets, days), trades


def im(days, O, H, L, C, V, m, cost, longonly=False):
    nd, T = C.shape
    k1 = 30 // m - 1
    prev = np.r_[np.nan, C[:-1, -1]]
    r1 = C[:, k1] / prev - 1
    side = np.sign(r1)
    if longonly:
        side = np.maximum(side, 0)
    last = C[:, -1] / C[:, T - 1 - 30 // m] - 1
    rets = np.nan_to_num(side * last - np.abs(side) * 2 * cost)
    return pd.Series(rets, days), np.abs(np.nan_to_num(side))


def stats(r, trades):
    if len(r) < 20 or r.std() == 0:
        return None
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    ann = 252 if r.index.dayofweek.max() < 5 else 365
    cagr = eq.iloc[-1] ** (1 / max(yrs, 0.1)) - 1
    sh = r.mean() / r.std() * np.sqrt(ann)
    dd = (eq / eq.cummax() - 1).min()
    act = r[trades > 0]
    return dict(cagr=cagr, sharpe=sh, maxdd=dd, trades_yr=trades.sum() / max(yrs, 0.1),
                win=(act > 0).mean() if len(act) else np.nan)


PERIODS = [("2018-01-01", "2021-12-31"), ("2022-01-01", "2023-12-31"), ("2024-01-01", "2026-12-31")]


def run(sym, longonly=False):
    days, O, H, L, C, V, m = grid(sym)
    cost = MARKETS[sym][5]
    rows = []
    bh = pd.Series(C[:, -1], days).pct_change().fillna(0)
    for name, fn in (("NAM", nam), ("ORB", orb), ("IM", im)):
        r, tr = fn(days, O, H, L, C, V, m, cost, longonly)
        tr = pd.Series(tr, days)
        for a, b in [("all", None)] + PERIODS:
            sl = slice(None) if a == "all" else slice(a, b)
            s = stats(r[sl], tr[sl].values)
            if s:
                bs = stats(bh[sl], np.ones(len(bh[sl])))
                rows.append(dict(sym=sym + ("-long" if longonly else ""), strat=name, period=a[:4] if a != "all" else "all",
                                 **{k: round(v, 3) for k, v in s.items()}, bh_cagr=round(bs["cagr"], 3) if bs else None))
    return rows


if __name__ == "__main__":
    syms = sys.argv[1:] or [s for s in MARKETS if (D / f"{MARKETS[s][0]}.feather").exists()]
    allrows = []
    for s in syms:
        try:
            allrows += run(s)
            if MARKETS[s][2] == "00:00" or s.endswith("-NY"):
                allrows += run(s, longonly=True)  # spot crypto can't short
            print("done", s, flush=True)
        except Exception as e:
            print("fail", s, e, flush=True)
    df = pd.DataFrame(allrows)
    out = Path(__file__).resolve().parent / "results" / "daytrade.csv"
    old = pd.read_csv(out) if out.exists() and sys.argv[1:] else None
    if old is not None:
        df = pd.concat([old[~old.sym.isin(df.sym)], df])
    df.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    print(df[df.period == "all"].sort_values("sharpe", ascending=False).to_string(index=False))
