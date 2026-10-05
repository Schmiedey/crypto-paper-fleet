"""Deep strategy search on daily data: one evaluator, real costs, fixed parameters, out-of-sample judgement.

Candidates come from published research and the most-starred GitHub strategy repos
(paperswithbacktest/awesome-systematic-trading, je-suis-tm/quant-trading, freqtrade, Reddit/GitHub TQQQ lore).
Rules for every strategy:
  * decide at the close of day t, earn day t+1 (no lookahead)
  * costs per unit of turnover (COST by asset class); idle cash earns the 13-week T-bill rate
  * leverage pays T-bill + 0.5% on the borrowed part; synthetic 3x ETFs pay 0.95%/yr fees
  * paper default parameters; a small neighbourhood grid checks that the result is not one lucky setting
  * in-sample = up to 2014, out-of-sample = 2015-2026 (crypto: IS to 2019, OOS 2020-2026)

Usage: python research/deep_search.py          writes research/results/deep_search.csv
"""
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parent
CACHE = R / "daily"
OUT = R / "results" / "deep_search.csv"
IS_END = {"etf": "2014-12-31", "crypto": "2019-12-31"}
COST = {"etf": 0.0005, "crypto": 0.0025, "fx": 0.0002}


# ---------------------------------------------------------------- data
def px(t, field="Close"):
    f = CACHE / f"{t.replace('^', '_').replace('=', '_')}.feather"
    if not f.exists():
        CACHE.mkdir(exist_ok=True)
        d = yf.download(t, start="1990-01-01", progress=False, auto_adjust=True, timeout=30)
        d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
        d.reset_index().to_feather(f)
    d = pd.read_feather(f).set_index("Date")
    d.index = pd.to_datetime(d.index).tz_localize(None)
    return d[field] if field else d


def tbill():
    r = px("^IRX") / 100 / 252
    return r.clip(lower=0)


# ---------------------------------------------------------------- evaluator
def evaluate(w, rets, kind, rf, lev_cost=True):
    """w: DataFrame of target weights (decided at close t). rets: DataFrame of asset daily returns."""
    w = w.reindex(rets.index).ffill().fillna(0)
    held = w.shift(1).fillna(0)
    gross = (held * rets).sum(1)
    rfd = rf.reindex(rets.index).ffill().fillna(0)
    expo = held.abs().sum(1)
    cash = (1 - expo).clip(lower=0) * rfd
    borrow = (expo - 1).clip(lower=0) * (rfd + 0.005 / 252) if lev_cost else 0
    turn = (w - w.shift(1).fillna(0)).abs().sum(1).shift(1).fillna(0)
    return gross + cash - borrow - turn * COST[kind], turn


def stats(r, turn=None):
    r = r.dropna()
    if len(r) < 120 or r.std() == 0:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / (365 if (r.index.dayofweek >= 5).any() else 252)
    ann = 365 if (r.index.dayofweek >= 5).any() else 252
    out = dict(cagr=eq.iloc[-1] ** (1 / yrs) - 1, sharpe=r.mean() / r.std() * np.sqrt(ann),
               maxdd=(eq / eq.cummax() - 1).min(), vol=r.std() * np.sqrt(ann))
    if turn is not None:
        out["turns_yr"] = turn.reindex(r.index).sum() / yrs
    return out


# ---------------------------------------------------------------- signal helpers
def sma(s, n):
    return s.rolling(n).mean()


def mom(s, n):
    return s / s.shift(n) - 1


def one(series_bool, col):
    return series_bool.astype(float).to_frame(col)


def mondays(idx):
    """Month-end rebalance mask."""
    s = pd.Series(idx, idx)
    return s.groupby([idx.year, idx.month]).transform("max") == s


def monthly(w):
    """Keep weights only on month-ends, hold in between."""
    m = mondays(w.index)
    return w.where(m, np.nan).ffill()


# ---------------------------------------------------------------- strategies (each returns (weights, rets, kind))
def S_buyhold(t, kind="etf"):
    p = px(t)
    return pd.DataFrame({t: 1.0}, p.index), p.pct_change().to_frame(t), kind


def S_sma_timing(t, n=200, lev=1.0, kind="etf"):
    """Faber 2007 / Gayed 2016 'Leverage for the long run': hold (lev x) when above the n-day SMA."""
    p = px(t)
    return (one(p > sma(p, n), t) * lev), p.pct_change().to_frame(t), kind


def S_synthetic_3x(t, n=200):
    """3x daily-reset ETF (TQQQ/UPRO style, built from the index ETF so history covers 2000-2002 and 2008),
    held only while the underlying is above its n-day SMA. n=0 means always hold."""
    p = px(t)
    rf = tbill().reindex(p.index).ffill().fillna(0)
    r3 = 3 * p.pct_change() - 2 * (rf + 0.005 / 252) - 0.0095 / 252
    w = one(p > sma(p, n), "3x") if n else pd.DataFrame({"3x": 1.0}, p.index)
    return w, r3.to_frame("3x"), "etf"


def S_gem(look=252):
    """Antonacci Global Equities Momentum: SPY vs EFA by 12m return, both vs T-bills, else bonds (IEF)."""
    P = pd.DataFrame({t: px(t) for t in ("SPY", "EFA", "IEF")}).dropna()
    m = P.apply(lambda s: mom(s, look))
    bill = tbill().reindex(P.index).ffill().rolling(look).sum()
    best = np.where(m.SPY > m.EFA, "SPY", "EFA")
    w = pd.DataFrame(0.0, P.index, P.columns)
    for i, (b, ms, mb) in enumerate(zip(best, m.SPY.values, bill.values)):
        w.iloc[i, list(P.columns).index(b if ms > mb else "IEF")] = 1
    return monthly(w), P.pct_change(), "etf"


def S_vaa(top=1):
    """Keller & Keuning 2017 VAA-G4: 13612W momentum; any risk asset negative -> best defensive asset."""
    risk, safe = ["SPY", "EFA", "EEM", "AGG"], ["LQD", "IEF", "SHY"]
    P = pd.DataFrame({t: px(t) for t in risk + safe}).dropna()
    s = 12 * mom(P, 21) + 4 * mom(P, 63) + 2 * mom(P, 126) + mom(P, 252)
    w = pd.DataFrame(0.0, P.index, P.columns)
    for i in range(len(P)):
        row = s.iloc[i]
        if row[risk].isna().any():
            continue
        if (row[risk] > 0).all():
            for c in row[risk].nlargest(top).index:
                w.iloc[i, list(P.columns).index(c)] = 1 / top
        else:
            w.iloc[i, list(P.columns).index(row[safe].idxmax())] = 1
    return monthly(w), P.pct_change(), "etf"


def S_sector_rotation(look=126, top=3, filt=True):
    """Sector momentum (Moskowitz & Grinblatt style): top-k SPDR sectors by 6m return, SPY>SMA200 filter."""
    secs = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
    P = pd.DataFrame({t: px(t) for t in secs + ["SPY"]}).dropna()
    m = mom(P[secs], look)
    rank = m.rank(axis=1, ascending=False)
    w = (rank <= top).astype(float) / top
    if filt:
        w = w.mul((P.SPY > sma(P.SPY, 200)).astype(float), axis=0)
    w["SPY"] = 0.0
    return monthly(w), P.pct_change(), "etf"


def S_tsmom(look=252, target=0.10):
    """Moskowitz, Ooi & Pedersen 2012 time-series momentum on 8 asset-class ETFs, long-only, vol-scaled."""
    assets = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]
    P = pd.DataFrame({t: px(t) for t in assets}).dropna()
    r = P.pct_change()
    vol = r.rolling(63).std() * np.sqrt(252)
    w = (mom(P, look) > 0).astype(float) * (target / vol) / len(assets)
    w = w.clip(upper=0.5)
    return monthly(w), r, "etf"


def S_risk_parity():
    """Inverse-volatility SPY / TLT / GLD ('all-weather' baseline), monthly."""
    P = pd.DataFrame({t: px(t) for t in ("SPY", "TLT", "GLD")}).dropna()
    r = P.pct_change()
    iv = 1 / r.rolling(63).std()
    return monthly(iv.div(iv.sum(1), axis=0)), r, "etf"


def S_overnight(t):
    """Overnight anomaly: buy at the close, sell at the next open (two trades a day)."""
    d = px(t, None)
    r = (d.Open / d.Close.shift(1) - 1).to_frame(t)
    w = pd.DataFrame({t: 1.0}, d.index)
    rr, _ = r, None
    # turnover is 2 per day: charge it directly
    return w, rr - 2 * COST["etf"], "etf_nocost"


def S_ibs(t, lo=0.2, hi=0.8):
    """Internal bar strength mean reversion: buy when (C-L)/(H-L) < lo, exit when > hi."""
    d = px(t, None)
    ibs = (d.Close - d.Low) / (d.High - d.Low).replace(0, np.nan)
    pos, out = 0.0, []
    for v in ibs.values:
        if v < lo:
            pos = 1.0
        elif v > hi:
            pos = 0.0
        out.append(pos)
    return pd.DataFrame({t: out}, d.index), d.Close.pct_change().to_frame(t), "etf"


def S_rsi2(t, lo=10, n=200):
    """Connors RSI(2): buy RSI2<lo above SMA200, sell when close > 5-day SMA."""
    p = px(t)
    d = p.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    ma5, trend = sma(p, 5), p > sma(p, n)
    pos, out = 0.0, []
    for r_, c, m5, tr in zip(rsi.values, p.values, ma5.values, trend.values):
        if pos == 0 and r_ < lo and tr:
            pos = 1.0
        elif pos == 1 and c > m5:
            pos = 0.0
        out.append(pos)
    return pd.DataFrame({t: out}, p.index), p.pct_change().to_frame(t), "etf"


def S_tom(t, before=4, after=3):
    """Turn of the month: hold the last `before` and first `after` trading days of each month."""
    p = px(t)
    idx = p.index
    g = pd.Series(1, idx).groupby([idx.year, idx.month])
    k_from_start = g.cumsum()
    k_from_end = g.transform("sum") - k_from_start + 1
    # weight decided at close t applies to t+1: hold t+1 if t+1 is in the window
    inwin = ((k_from_end <= before) | (k_from_start <= after)).shift(-1).fillna(False)
    return one(inwin, t), p.pct_change().to_frame(t), "etf"


# ---- classic indicator zoo (je-suis-tm/quant-trading & friends), long-only, cash otherwise
def ta_signals(t):
    d = px(t, None)
    c, h, l, o = d.Close, d.High, d.Low, d.Open
    sig = {}
    sig["sma50_200"] = sma(c, 50) > sma(c, 200)
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    sig["macd"] = macd > macd.ewm(span=9).mean()
    ha_c = (o + h + l + c) / 4
    ha_o = ha_c.copy()
    ha_o.iloc[0] = o.iloc[0]
    for i in range(1, len(ha_o)):
        ha_o.iloc[i] = (ha_o.iloc[i - 1] + ha_c.iloc[i - 1]) / 2
    sig["heikin_ashi"] = ha_c > ha_o
    ao = sma((h + l) / 2, 5) - sma((h + l) / 2, 34)
    sig["awesome_osc"] = ao > 0
    mid, sd = sma(c, 20), c.rolling(20).std()
    st, out = 0, []
    for cc, u, m_ in zip(c.values, (mid + 2 * sd).values, mid.values):
        st = 1 if cc > u else (0 if cc < m_ else st)
        out.append(st)
    sig["boll_breakout"] = pd.Series(out, c.index).astype(bool)
    st, out = 0, []
    hi20, lo10 = h.rolling(20).max().shift(1), l.rolling(10).min().shift(1)
    for cc, a, b in zip(c.values, hi20.values, lo10.values):
        st = 1 if cc > a else (0 if cc < b else st)
        out.append(st)
    sig["donchian_turtle"] = pd.Series(out, c.index).astype(bool)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(1)
    atr = tr.rolling(10).mean()
    hl2 = (h + l) / 2
    ub, lb = (hl2 + 3 * atr).values, (hl2 - 3 * atr).values
    fu, fl, trend, out = ub.copy(), lb.copy(), 1, []
    for i in range(len(c)):
        if i:
            fu[i] = ub[i] if (np.isnan(fu[i - 1]) or ub[i] < fu[i - 1] or c.iloc[i - 1] > fu[i - 1]) else fu[i - 1]
            fl[i] = lb[i] if (np.isnan(fl[i - 1]) or lb[i] > fl[i - 1] or c.iloc[i - 1] < fl[i - 1]) else fl[i - 1]
            trend = 1 if c.iloc[i] > fu[i - 1] else (-1 if c.iloc[i] < fl[i - 1] else trend)
        out.append(trend > 0)
    sig["supertrend"] = pd.Series(out, c.index)
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    sig["ichimoku_cloud"] = c > pd.concat([span_a, span_b], axis=1).max(1)
    dd = c.diff()
    rsi = 100 - 100 / (1 + dd.clip(lower=0).rolling(14).mean() / (-dd.clip(upper=0)).rolling(14).mean())
    st, out = 0, []
    for v in rsi.values:
        st = 1 if v < 30 else (0 if v > 70 else st)
        out.append(st)
    sig["rsi14_30_70"] = pd.Series(out, c.index).astype(bool)
    # parabolic SAR (0.02, 0.2)
    af, ep, sar, up, out = 0.02, h.iloc[0], l.iloc[0], True, []
    for i in range(len(c)):
        if i:
            sar = sar + af * (ep - sar)
            if up:
                if l.iloc[i] < sar:
                    up, sar, ep, af = False, ep, l.iloc[i], 0.02
                elif h.iloc[i] > ep:
                    ep, af = h.iloc[i], min(0.2, af + 0.02)
            else:
                if h.iloc[i] > sar:
                    up, sar, ep, af = True, ep, h.iloc[i], 0.02
                elif l.iloc[i] < ep:
                    ep, af = l.iloc[i], min(0.2, af + 0.02)
        out.append(up)
    sig["parabolic_sar"] = pd.Series(out, c.index)
    return sig, c.pct_change()


# ---------------------------------------------------------------- crypto specials
def S_btc_mtf():
    """Multi-timeframe BTC trend (paperswithbacktest crypto entry): long when price is above the
    20, 50 and 100-day SMAs together; otherwise cash."""
    p = px("BTC-USD")
    ok = (p > sma(p, 20)) & (p > sma(p, 50)) & (p > sma(p, 100))
    return one(ok, "BTC"), p.pct_change().to_frame("BTC"), "crypto"


def run_all():
    rf = tbill()
    rows = []

    def add(name, family, params, w, r, kind, note=""):
        if kind == "etf_nocost":
            net = (w.shift(1).fillna(0) * r).sum(1)
            turn = pd.Series(2.0, net.index)
            kind = "etf"
        else:
            net, turn = evaluate(w, r, kind, rf)
        net = net[net.index >= r.dropna(how="all").index[0]]
        first = w.reindex(r.index).dropna(how="all").index
        net = net[net.index >= (first[0] if len(first) else net.index[0])].iloc[260:]  # warm-up
        ise = IS_END["crypto" if kind == "crypto" else "etf"]
        for per, sl in (("all", slice(None)), ("IS", slice(None, ise)), ("OOS", slice(ise, None)),
                        ("2022+", slice("2022-01-01", None))):
            s = stats(net[sl], turn)
            if s:
                rows.append(dict(name=name, family=family, params=params, period=per, start=str(net[sl].index[0].date()),
                                 **{k: round(float(v), 4) for k, v in s.items()}, note=note))

    print("benchmarks...", flush=True)
    for t, k in (("SPY", "etf"), ("QQQ", "etf"), ("TLT", "etf"), ("GLD", "etf"), ("BTC-USD", "crypto"), ("ETH-USD", "crypto")):
        add(f"buy&hold {t}", "benchmark", "", *S_buyhold(t, k))
    add("buy&hold 3x QQQ (synthetic TQQQ)", "benchmark", "", *S_synthetic_3x("QQQ", 0))
    add("buy&hold 3x SPY (synthetic UPRO)", "benchmark", "", *S_synthetic_3x("SPY", 0))

    print("timing / leverage...", flush=True)
    for t in ("SPY", "QQQ"):
        for n in (150, 200, 250):
            for lev in (1, 2):
                add(f"{t} SMA{n} timing {lev}x", "sma_timing", f"n={n},lev={lev}", *S_sma_timing(t, n, lev))
        for n in (150, 200, 250):
            add(f"{t} 3x when > SMA{n}", "lev3x_sma", f"n={n}", *S_synthetic_3x(t, n))
    for t, k in (("BTC-USD", "crypto"), ("ETH-USD", "crypto")):
        for n in (20, 50, 100, 200):
            add(f"{t} SMA{n} timing", "crypto_sma", f"n={n}", *S_sma_timing(t, n, 1, k))
    add("BTC multi-timeframe trend (20/50/100)", "crypto_mtf", "", *S_btc_mtf())

    print("allocation...", flush=True)
    for look in (189, 252, 315):
        add(f"GEM dual momentum {look}d", "gem", f"look={look}", *S_gem(look))
    for top in (1, 2):
        add(f"VAA-G4 top{top}", "vaa", f"top={top}", *S_vaa(top))
    for look, top in itertools.product((63, 126, 252), (2, 3)):
        add(f"sector rotation {look}d top{top}", "sector_rot", f"look={look},top={top}", *S_sector_rotation(look, top))
    for look in (126, 252):
        add(f"TSMOM 8 ETFs {look}d", "tsmom", f"look={look}", *S_tsmom(look))
    add("risk parity SPY/TLT/GLD", "risk_parity", "", *S_risk_parity())

    print("anomalies...", flush=True)
    for t in ("SPY", "QQQ"):
        add(f"{t} overnight (close->open)", "overnight", "", *S_overnight(t))
        for lo, hi in ((0.1, 0.8), (0.2, 0.8), (0.25, 0.75)):
            add(f"{t} IBS {lo}/{hi}", "ibs", f"lo={lo},hi={hi}", *S_ibs(t, lo, hi))
        for lo in (5, 10, 15):
            add(f"{t} RSI2<{lo}", "rsi2", f"lo={lo}", *S_rsi2(t, lo))
        for b, a in ((3, 2), (4, 3), (5, 4)):
            add(f"{t} turn-of-month -{b}/+{a}", "tom", f"b={b},a={a}", *S_tom(t, b, a))

    print("indicator zoo...", flush=True)
    for t, k in (("SPY", "etf"), ("QQQ", "etf"), ("GLD", "etf"), ("TLT", "etf"), ("BTC-USD", "crypto"),
                 ("ETH-USD", "crypto"), ("EURUSD=X", "fx")):
        sig, r = ta_signals(t)
        for nm, s in sig.items():
            add(f"{t} {nm}", f"ta_{nm}", "", s.astype(float).to_frame(t), r.to_frame(t), k)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    return df


if __name__ == "__main__":
    df = run_all()
    pd.set_option("display.width", 250)
    o = df[df.period == "OOS"].sort_values("sharpe", ascending=False)
    print(o[["name", "start", "cagr", "sharpe", "maxdd", "turns_yr"]].head(40).to_string(index=False))
