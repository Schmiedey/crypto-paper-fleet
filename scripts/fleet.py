"""Run a fleet of Freqtrade paper bots in parallel on live Kraken prices.

  python scripts/fleet.py start [all | NAME ...]  start proxy + the ACTIVE wave, every bot, or only the named ones
  python scripts/fleet.py stop               stop everything (crypto bots, Kalshi trader, proxy)
  python scripts/fleet.py status             leaderboard vs. equal-weight buy & hold (live dashboard: http://127.0.0.1:8787)
  python scripts/fleet.py retire NAME ...    stop bots for good
  python scripts/fleet.py rotate             retire the 3 worst judged bots, start 3 from the bench
"""
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
BIN = ROOT / ".venv" / "bin"
BASE_CFG = ROOT / "user_data" / "config_base.json"
PROXY_PORT = 8899
WALLET = 10000

FREQAI_BASE = {
    "enabled": True,
    "purge_old_models": 2,
    "train_period_days": 15,
    "backtest_period_days": 7,
    "live_retrain_hours": 1,
    "expired_hours": 3,
    "override_exchange_checks": True,
    "feature_parameters": {
        "include_timeframes": ["5m", "15m", "1h"],
        "include_corr_pairlist": ["BTC/USD", "ETH/USD"],
        "label_period_candles": 24,
        "include_shifted_candles": 2,
        "DI_threshold": 0.9,
        "weight_factor": 0.9,
        "principal_component_analysis": False,
        "use_SVM_to_remove_outliers": True,
        "indicator_periods_candles": [10, 20],
    },
    "data_split_parameters": {"test_size": 0.33, "random_state": 1},
    "model_training_parameters": {"n_estimators": 400},
}


def _freqai(**overrides) -> dict:
    cfg = json.loads(json.dumps(FREQAI_BASE))
    cfg["feature_parameters"].update(overrides.pop("feature_parameters", {}))
    cfg.update(overrides)
    return cfg


# Self-learning bots: FreqAI retrains these models on fresh candles every hour while trading.
FREQAI_BOTS = {
    "AI_LightGBM_5m": ("LightGBMRegressor", {"timeframe": "5m", "freqai": _freqai()}),
    "AI_XGBoost_5m": ("XGBoostRegressor", {"timeframe": "5m", "freqai": _freqai()}),
    "AI_LightGBM_1h": ("LightGBMRegressor", {"timeframe": "1h", "freqai": _freqai(
        train_period_days=60, feature_parameters={"include_timeframes": ["1h", "4h"], "label_period_candles": 12})}),
}


# Bots launched by a bare `start`. RAM allows ~28 at once (~165MB each, FreqAI more); the other
# strategies in user_data/strategies are the bench, rotated in as losers get cut.
ACTIVE = [
    # 5m
    "Strategy001", "Strategy002", "Strategy003", "Strategy004", "Strategy005", "Diamond", "PowerTower",
    "UniversalMACD", "BinHV27", "ClucMay72018", "CofiBitStrategy", "CombinedBinHAndCluc", "MultiRSI",
    "SmoothOperator",
    # 15m / 1h / 4h
    "SwingHighToSky", "Supertrend", "TrendRiderStrategy", "BbandRsi", "ADXMomentum",
    "DoubleEMACrossoverWithTrend", "MACDCrossoverWithTrend", "RSIDirectionalWithTrend",
    "EMAPriceCrossoverWithThreshold", "hlhb", "MultiMa",
    *FREQAI_BOTS,
]


def bots() -> dict[str, dict]:
    out = {}
    for f in sorted((ROOT / "user_data" / "strategies").glob("*.py")):
        if f.stem != "FreqaiSpot":
            out[f.stem] = {"strategy": f.stem, "model": None, "extra": {}}
    for name, (model, extra) in FREQAI_BOTS.items():
        out[name] = {"strategy": "FreqaiSpot", "model": model, "extra": extra}
    return out


def _seed_datadir(d: Path, freqai: dict) -> str:
    """Give each FreqAI bot a private, trimmed copy of the seed history so bots don't write the same files."""
    import pandas as pd
    out = d / "data"
    if not out.exists():
        out.mkdir()
        days = freqai["train_period_days"] + 30
        for tf in freqai["feature_parameters"]["include_timeframes"]:
            for f in (ROOT / "user_data" / "data" / "kraken").glob(f"*-{tf}.feather"):
                df = pd.read_feather(f)
                df[df["date"] >= df["date"].max() - pd.Timedelta(days=days)].reset_index(drop=True).to_feather(out / f.name)
    return str(out)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _pid(path: Path) -> int | None:
    if path.exists() and _alive(pid := int(path.read_text())):
        return pid
    return None


def _spawn(cmd: list[str], log: Path, pidfile: Path) -> int:
    with open(log, "ab") as fh:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    pidfile.write_text(str(p.pid))
    return p.pid


def start(names: list[str]) -> None:
    RUNS.mkdir(exist_ok=True)
    proxy_pid = RUNS / "proxy.pid"
    if not _pid(proxy_pid):
        pid = _spawn([str(BIN / "python"), "scripts/kraken_proxy.py", str(PROXY_PORT)], RUNS / "proxy.log", proxy_pid)
        # Keep the Mac from idle-sleeping while the proxy (and so the fleet) is running.
        if shutil.which("caffeinate"):
            subprocess.Popen(["caffeinate", "-i", "-w", str(pid)], start_new_session=True)
        time.sleep(2)
    bench = RUNS / "benchmark.json"
    if not bench.exists():
        bench.write_text(json.dumps({"started": datetime.now(timezone.utc).isoformat(), "prices": prices()}))
    for name, script in (("kalshi", "scripts/kalshi_paper.py"), ("portfolio", "scripts/portfolio_paper.py"),
                         ("dashboard", "scripts/dashboard.py"), ("daytrader", "scripts/daytrader_paper.py"),
                         ("multi", "scripts/multi_paper.py")):
        d = RUNS / name
        d.mkdir(exist_ok=True)
        if not _pid(d / "bot.pid"):
            _spawn([str(BIN / "python"), script], d / "stdout.log", d / "bot.pid")
            print(f"started {name}")

    fleet = bots()
    if names == ["all"]:  # every strategy that hasn't been retired
        names = [n for n in fleet if not (RUNS / n / "retired").exists()]
    for name in names or ACTIVE:
        b, d = fleet[name], RUNS / name
        d.mkdir(exist_ok=True)
        if _pid(d / "bot.pid"):
            continue
        cfg = {"bot_name": name, **b["extra"]}
        if b["model"]:
            cfg["freqai"] = {**b["extra"]["freqai"], "identifier": name}  # separate model folder per bot
            cfg["datadir"] = _seed_datadir(d, cfg["freqai"])
        (d / "config.json").write_text(json.dumps(cfg, indent=2))
        cmd = [str(BIN / "freqtrade"), "trade", "--userdir", "user_data",
               "-c", str(BASE_CFG), "-c", str(d / "config.json"),
               "--strategy", b["strategy"], "--db-url", f"sqlite:///{d / 'trades.sqlite'}",
               "--logfile", str(d / "bot.log")]
        if b["model"]:
            cmd += ["--freqaimodel", b["model"]]
        _spawn(cmd, d / "stdout.log", d / "bot.pid")
        print(f"started {name}")
        time.sleep(1.5)  # stagger startup so bots don't all warm up at once


def stop() -> None:
    for pidfile in list(RUNS.glob("*/bot.pid")) + [RUNS / "proxy.pid"]:
        if pid := _pid(pidfile):
            os.killpg(pid, signal.SIGTERM)
            print(f"stopped {pidfile.parent.name if pidfile.name == 'bot.pid' else 'proxy'}")


def prices() -> dict[str, float]:
    import ccxt
    ex = ccxt.kraken({"urls": {"api": {"public": f"http://127.0.0.1:{PROXY_PORT}"}}})
    pairs = json.loads(BASE_CFG.read_text())["exchange"]["pair_whitelist"]
    return {p: t["last"] for p, t in ex.fetch_tickers(pairs).items()}


def retire(names: list[str]) -> None:
    """Stop bots for good; `start all` skips retired ones."""
    for name in names:
        if pid := _pid(RUNS / name / "bot.pid"):
            os.killpg(pid, signal.SIGTERM)
        (RUNS / name / "retired").touch()
        print(f"retired {name}")


def rotate(k: int = 3, min_closed: int = 10) -> None:
    """Tournament step: retire the k worst running bots that have closed enough trades to judge
    and trail buy & hold, and start k strategies from the bench in their place."""
    rows, bh = leaderboard()
    judged = [r for r in rows if r[1] and r[3] >= min_closed and r[2] < bh and r[0] not in FREQAI_BOTS]
    losers = sorted(judged, key=lambda r: r[2])[:k]
    bench = [n for n in bots() if not (RUNS / n).exists()][: len(losers)]
    for name, *_ in losers:
        os.killpg(int((RUNS / name / "bot.pid").read_text()), signal.SIGTERM)
        (RUNS / name / "retired").touch()
        print(f"retired {name}")
    if bench:
        start(bench)
    if not losers:
        print(f"nobody to retire yet (need >= {min_closed} closed trades and trailing buy & hold)")


def leaderboard() -> tuple[list, float]:
    now = prices()
    bench = json.loads((RUNS / "benchmark.json").read_text())
    bh = sum(now[p] / bench["prices"][p] - 1 for p in bench["prices"]) / len(bench["prices"]) * 100  # original 14-coin basket
    rows = []
    for name in (n for n in bots() if (RUNS / n).exists()):
        db = RUNS / name / "trades.sqlite"
        running = bool(_pid(RUNS / name / "bot.pid"))
        if not db.exists():
            rows.append((name, running, 0.0, 0, 0, 0, None))
            continue
        con = sqlite3.connect(db)
        try:
            closed = con.execute("SELECT close_profit_abs FROM trades WHERE is_open=0").fetchall()
            opened = con.execute("SELECT pair, amount, stake_amount, fee_open FROM trades WHERE is_open=1").fetchall()
        except sqlite3.OperationalError:  # bot hasn't created its tables yet
            closed, opened = [], []
        con.close()
        realized = sum(r[0] or 0 for r in closed)
        # Value open positions at the last price, net of a taker exit fee.
        unreal = sum(amt * now.get(pair, 0) * (1 - fee) - stake for pair, amt, stake, fee in opened if amt)
        wins = sum(1 for r in closed if (r[0] or 0) > 0)
        ret = (realized + unreal) / WALLET * 100
        rows.append((name, running, ret, len(closed), len(opened), wins, realized))
    rows.sort(key=lambda r: -r[2])
    return rows, bh


def status() -> None:
    rows, bh = leaderboard()
    started = json.loads((RUNS / "benchmark.json").read_text())["started"]
    hours = (datetime.now(timezone.utc) - datetime.fromisoformat(started)).total_seconds() / 3600
    print(f"Running {hours:.1f}h | equal-weight buy&hold of all 14 coins: {bh:+.2f}%\n")
    print(f"{'bot':31} {'up':>3} {'return':>8} {'vs B&H':>8} {'closed':>6} {'open':>4} {'win%':>5}")
    for name, running, ret, n, o, w, _ in rows:
        win = f"{w / n * 100:.0f}" if n else "-"
        print(f"{name:31} {'✓' if running else '✗':>3} {ret:+7.2f}% {ret - bh:+7.2f}% {n:6} {o:4} {win:>5}")
    # The combination that tested best: 30% trend portfolio + 70% equal-weight basket of research finalists.
    deployed = ROOT / "research" / "bt" / "deployed.txt"
    pdb = RUNS / "portfolio" / "portfolio.sqlite"
    if deployed.exists() and pdb.exists():
        names = set(deployed.read_text().split())
        dip = [r[2] for r in rows if r[0] in names]
        con = sqlite3.connect(pdb)
        trend = con.execute("SELECT usd FROM equity WHERE sleeve='blend_mom_don_btc' ORDER BY ts DESC LIMIT 1").fetchone()
        con.close()
        if dip and trend:
            t = (trend[0] / 10000 - 1) * 100
            d = sum(dip) / len(dip)
            print(f"\nCOMBO 30% trend + 70% dip-buyers: {0.3 * t + 0.7 * d:+.2f}%   "
                  f"(trend blend {t:+.2f}%, dip-buyer basket of {len(dip)} {d:+.2f}%)")
    sys.path.insert(0, str(ROOT / "scripts"))
    import kalshi_paper
    import portfolio_paper
    portfolio_paper.report()
    kalshi_paper.report()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"start": lambda: start(sys.argv[2:]), "stop": stop, "status": status, "rotate": rotate, "retire": lambda: retire(sys.argv[2:])}[cmd]()
