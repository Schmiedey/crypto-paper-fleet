# Crypto paper-trading fleet

27 Freqtrade dry-run bots (fake $10k each) on live Kraken prices, a trend portfolio, the Nasdaq day trader and three daily strategies (see the dashboard).
Nothing here trades real money and there are no API keys.

GitHub Actions runs it around the clock: each run lasts ~5h50m and queues the next one when it starts.
Trade databases and `STATUS.md` (leaderboard) are saved to the `state` branch every 5 minutes.

## Live dashboard
`docs/index.html` is a TradingView-style terminal (candles, entry/exit markers, open positions, trade history, leaderboard).
It is served by GitHub Pages and reads `data.json` from the `state` branch, with live prices straight from Kraken.
