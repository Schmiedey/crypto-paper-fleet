# Crypto paper-trading fleet

27 Freqtrade dry-run bots (fake $10k each) on live Kraken prices, plus a Kalshi paper trader and a trend portfolio.
Nothing here trades real money and there are no API keys.

GitHub Actions runs it around the clock: each run lasts ~5h50m and queues the next one when it starts.
Trade databases and `STATUS.md` (leaderboard) are saved to the `state` branch every 15 minutes.
