"""Research-backed systematic crypto strategies, tested 2020-2026 with strict in/out-of-sample split.

Every family has a small parameter grid. Parameters are chosen on the in-sample period only
(2020-01 .. 2023-12); the out-of-sample period (2024-01 .. today) is reported untouched, together
with how many of the grid's variants were profitable out-of-sample (a robustness check against luck).

Costs: 0.30% per unit of turnover (Kraken 0.26% taker + slippage). Long-only, no leverage.
Signals use data up to the close of bar t and earn the return of bar t+1 (no lookahead).

Usage: python research/systematic.py
"""
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "results"
COST = 0.003
IS_END = "2023-12-31"
OOS_START = "2024-01-01"


def load(tf: str) -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(DATA.glob(f"*-{tf}.feather")):
        df = pd.read_feather(f).set_index("date")
        out[f.stem.split("-")[0]] = df
    return out


def panel(frames: dict[str, pd.DataFrame], col: str) -> pd.DataFrame:
    return pd.DataFrame({c: f[col] for c, f in frames.items()}).sort_index()


def stats(r: pd.Series, periods: int) -> dict:
    r = r.dropna()
    if len(r) < 10 or r.std() == 0:
        return {"cagr": np.nan, "sharpe": np.nan, "maxdd": np.nan, "calmar": np.nan}
    eq = (1 + r).cumprod()
    years = len(r) / periods
    cagr = eq.iloc[-1] ** (1 / years) - 1
    dd = (eq / eq.cummax() - 1).min()
    return {"cagr": cagr, "sharpe": r.mean() / r.std() * np.sqrt(periods), "maxdd": dd,
            "calmar": cagr / abs(dd) if dd < 0 else np.nan}


def run(weights: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    """Portfolio return: weights decided at bar t earn bar t+1's return, minus turnover cost."""
    w = weights.reindex(rets.index).fillna(0)
    gross = (w.shift(1) * rets).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1)
    return gross - turnover.shift(1).fillna(0) * COST


# ---------- strategy families (each returns a weights DataFrame) ----------

def universe(close: pd.DataFrame, qvol: pd.DataFrame, top: int, min_days: int = 60) -> pd.DataFrame:
    """Top-N coins by trailing 30-day dollar volume that have enough history (known at bar t)."""
    hist = close.notna().cumsum() >= min_days
    liq = qvol.rolling(30, min_periods=20).mean().where(hist)
    return liq.rank(axis=1, ascending=False) <= top


def trend_sma(close, uni, n):
    sig = (close > close.rolling(n).mean()) & uni
    return sig.div(uni.sum(axis=1).clip(lower=1), axis=0)  # equal slot per universe coin; cash if not in trend


def trend_cross(close, uni, fast, slow):
    sig = (close.ewm(span=fast).mean() > close.ewm(span=slow).mean()) & uni
    return sig.div(uni.sum(axis=1).clip(lower=1), axis=0)


def donchian(close, uni, entry, exit_):
    hi = close.rolling(entry).max().shift(1)
    lo = close.rolling(exit_).min().shift(1)
    state = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    state[close > hi] = 1
    state[close < lo] = 0
    sig = state.ffill().fillna(0).astype(bool) & uni
    return sig.div(uni.sum(axis=1).clip(lower=1), axis=0)


def tsmom_vol(close, uni, look, target):
    r = close.pct_change()
    vol = r.rolling(30).std() * np.sqrt(365)
    sig = (close.pct_change(look) > 0) & uni
    w = (target / vol).clip(upper=1).where(sig, 0)
    return w.div(uni.sum(axis=1).clip(lower=1), axis=0)


def xs_momentum(close, uni, look, k, btc_filter, rebalance=7):
    score = close.pct_change(look).where(uni)
    top = score.rank(axis=1, ascending=False) <= k
    w = top.astype(float).div(top.sum(axis=1).clip(lower=1), axis=0)
    if btc_filter:
        w = w.mul(close["BTC"] > close["BTC"].rolling(btc_filter).mean(), axis=0)
    keep = np.arange(len(w)) % rebalance == 0
    return w.where(pd.Series(keep, index=w.index), np.nan).ffill()


def reversal(close, uni, k, btc_filter):
    score = close.pct_change(1).where(uni)
    bottom = score.rank(axis=1, ascending=True) <= k
    w = bottom.astype(float).div(bottom.sum(axis=1).clip(lower=1), axis=0)
    if btc_filter:
        w = w.mul(close["BTC"] > close["BTC"].rolling(btc_filter).mean(), axis=0)
    return w


def ml_rank(close, qvol, uni, k, horizon, seed=0):
    """Self-learning: LightGBM retrained monthly on all past data, ranks coins by predicted forward return."""
    import lightgbm as lgb
    r = close.pct_change()
    feats = {}
    for n in (1, 3, 7, 14, 30, 60):
        feats[f"ret{n}"] = close.pct_change(n)
    for n in (7, 30):
        feats[f"vol{n}"] = r.rolling(n).std()
    for n in (20, 50, 200):
        feats[f"dist{n}"] = close / close.rolling(n).mean() - 1
    feats["vratio"] = qvol / qvol.rolling(30).mean()
    btc = {f"btc_ret{n}": close["BTC"].pct_change(n) for n in (7, 30)}
    btc["btc_trend"] = close["BTC"] / close["BTC"].rolling(100).mean() - 1
    target = close.shift(-horizon) / close - 1
    target_rank = target.rank(axis=1, pct=True)

    long = pd.concat({k_: v.where(uni).stack() for k_, v in feats.items()}, axis=1)
    for k_, v in btc.items():
        long[k_] = long.index.get_level_values(0).map(v)
    long["y"] = target_rank.stack().reindex(long.index)
    long = long.dropna(subset=list(feats))

    dates = long.index.get_level_values(0)
    months = pd.date_range(pd.Timestamp("2021-01-01", tz="UTC"), close.index[-1], freq="MS")
    preds = []
    for m0, m1 in zip(months, list(months[1:]) + [close.index[-1] + pd.Timedelta(days=1)]):
        train = long[(dates < m0 - pd.Timedelta(days=horizon))].dropna(subset=["y"])
        test = long[(dates >= m0) & (dates < m1)]
        if len(train) < 2000 or test.empty:
            continue
        model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
        X = train.drop(columns="y")
        model.fit(X, train["y"])
        preds.append(pd.Series(model.predict(test.drop(columns="y")), index=test.index))
    score = pd.concat(preds).unstack()
    top = score.rank(axis=1, ascending=False) <= k
    w = top.astype(float).div(top.sum(axis=1).clip(lower=1), axis=0)
    w = w.mul(close["BTC"].reindex(w.index) > close["BTC"].rolling(100).mean().reindex(w.index), axis=0)
    keep = np.arange(len(w)) % horizon == 0
    return w.where(pd.Series(keep, index=w.index), np.nan).ffill().reindex(close.index).fillna(0)


def seasonality_btc(close_h, start_hour, end_hour):
    """Quantpedia: long BTC from start_hour to end_hour UTC each day."""
    hours = close_h.index.hour
    hold = (hours >= start_hour) & (hours < end_hour)  # weight set at bar t earns bar t+1
    w = pd.DataFrame(0.0, index=close_h.index, columns=close_h.columns)
    w.loc[hold, "BTC"] = 1.0
    return w


def bb_reversion(close_h, coin, n, k, exit_mid=True):
    c = close_h[coin]
    mid = c.rolling(n).mean()
    sd = c.rolling(n).std()
    state = pd.Series(np.nan, index=c.index)
    state[c < mid - k * sd] = 1
    state[c > (mid if exit_mid else mid + k * sd)] = 0
    w = pd.DataFrame(0.0, index=c.index, columns=close_h.columns)
    w[coin] = state.ffill().fillna(0)
    return w


def main() -> None:
    OUT.mkdir(exist_ok=True)
    d = load("1d")
    close, qvol = panel(d, "close"), panel(d, "close") * panel(d, "volume")
    rets = close.pct_change().fillna(0)
    uni10, uni20 = universe(close, qvol, 10), universe(close, qvol, 20)

    families = {
        "trend_sma (top10)": (lambda n: trend_sma(close, uni10, n), {"n": [20, 50, 100, 150, 200]}),
        "trend_ema_cross (top10)": (lambda f, s: trend_cross(close, uni10, f, s), {"f": [10, 20, 50], "s": [50, 100, 200]}),
        "donchian (top10)": (lambda e, x: donchian(close, uni10, e, x), {"e": [20, 50, 100], "x": [10, 20, 50]}),
        "tsmom_voltarget (top10)": (lambda l, t: tsmom_vol(close, uni10, l, t), {"l": [14, 30, 60, 90], "t": [0.4, 0.6, 0.8]}),
        "btc_only_sma": (lambda n: trend_sma(close[["BTC"]], uni10[["BTC"]] | True, n).reindex(columns=close.columns).fillna(0), {"n": [20, 50, 100, 150, 200]}),
        "xs_momentum (top20)": (lambda l, k, b: xs_momentum(close, uni20, l, k, b), {"l": [7, 14, 30, 60], "k": [3, 5], "b": [0, 100]}),
        "reversal_1d (top20)": (lambda k, b: reversal(close, uni20, k, b), {"k": [3, 5], "b": [0, 100]}),
    }
    bench = {"BTC buy&hold": pd.DataFrame({"BTC": 1.0}, index=close.index).reindex(columns=close.columns).fillna(0),
             "equal-weight top10 buy&hold": uni10.astype(float).div(uni10.sum(axis=1).clip(lower=1), axis=0)}

    rows = []

    def record(family, params, w, r_, periods):
        r = run(w, r_)
        is_, oos = stats(r[:IS_END], periods), stats(r[OOS_START:], periods)
        rows.append({"family": family, "params": params, **{f"is_{k}": v for k, v in is_.items()},
                     **{f"oos_{k}": v for k, v in oos.items()}, "exposure": w.sum(axis=1).mean(),
                     "turnover_yr": w.diff().abs().sum(axis=1).mean() * periods})

    for name, w in bench.items():
        record(name, "", w, rets, 365)
    for fam, (fn, grid) in families.items():
        for combo in itertools.product(*grid.values()):
            record(fam, dict(zip(grid, combo)), fn(*combo), rets, 365)
        print(f"done {fam}", flush=True)
    for k, h in itertools.product([3, 5], [1, 3, 7]):
        record("ml_lightgbm_rank (top20)", {"k": k, "horizon": h}, ml_rank(close, qvol, uni20, k, h), rets, 365)
    print("done ml", flush=True)

    h = load("1h")
    close_h = panel({c: h[c] for c in ("BTC", "ETH")}, "close")
    rets_h = close_h.pct_change().fillna(0)
    for s, e in [(21, 23), (20, 23), (22, 24)]:
        record("btc_hour_seasonality", {"from": s, "to": e}, seasonality_btc(close_h, s, e), rets_h, 24 * 365)
    for coin, n, k in itertools.product(["BTC", "ETH"], [20, 50], [2.0, 2.5]):
        record("bb_reversion_1h", {"coin": coin, "n": n, "k": k}, bb_reversion(close_h, coin, n, k), rets_h, 24 * 365)
    print("done hourly", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "systematic_all.csv", index=False)

    # Honest summary: per family, the variant with the best IN-SAMPLE Sharpe, judged out-of-sample.
    print(f"\nIn-sample 2020-01..{IS_END}  |  Out-of-sample {OOS_START}..today   (0.30% cost per trade)\n")
    hdr = f"{'family':28} {'chosen on IS':26} {'IS sharpe':>9} {'OOS cagr':>9} {'OOS sharpe':>10} {'OOS maxDD':>9} {'grid +OOS':>9}"
    print(hdr)
    for fam, g in res.groupby("family", sort=False):
        best = g.loc[g.is_sharpe.idxmax()] if g.is_sharpe.notna().any() else g.iloc[0]
        frac = f"{(g.oos_cagr > 0).sum()}/{len(g)}"
        print(f"{fam:28} {str(best.params)[:26]:26} {best.is_sharpe:9.2f} {best.oos_cagr:+8.1%} {best.oos_sharpe:10.2f} "
              f"{best.oos_maxdd:+8.1%} {frac:>9}")


if __name__ == "__main__":
    main()
