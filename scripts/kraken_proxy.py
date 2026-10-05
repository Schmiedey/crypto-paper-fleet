"""Shared caching proxy for Kraken's public REST API.

Every paper bot points ccxt at this proxy instead of api.kraken.com, so N bots asking for the
same candles/ticker produce one upstream request. Identical in-flight requests are coalesced and
upstream calls are spaced out to stay under Kraken's per-IP public rate limit.

Usage: python scripts/kraken_proxy.py [port]
"""
import asyncio
import sys
import time
from urllib.parse import urlencode

from aiohttp import ClientSession, ClientTimeout, web

UPSTREAM = "https://api.kraken.com"
MIN_SPACING = 0.35  # seconds between upstream calls
TTL = {"Ticker": 4, "Depth": 3, "Spread": 3, "Trades": 5, "AssetPairs": 900, "Assets": 900, "Time": 1, "SystemStatus": 30}

cache: dict[str, tuple[float, int, bytes, str]] = {}
inflight: dict[str, asyncio.Future] = {}
stats = {"requests": 0, "upstream": 0, "errors": 0}
_last_call = 0.0
_lock = asyncio.Lock()


def cache_key(method: str, query: dict) -> tuple[str, dict]:
    if method == "OHLC":
        # Drop `since`: Kraken always serves the latest 720 candles and ccxt filters client-side,
        # so every bot can share one response per (pair, interval).
        query = {k: v for k, v in query.items() if k != "since"}
    return f"{method}?{urlencode(sorted(query.items()))}", query


def is_fresh(method: str, query: dict, fetched: float, now: float) -> bool:
    if method == "OHLC":
        period = int(query.get("interval", 1)) * 60
        # Stale once a new candle period has started, and refresh the open candle every 20s.
        return fetched // period == now // period and now - fetched < 20
    return now - fetched < TTL.get(method, 0)


async def upstream(session: ClientSession, method: str, query: dict) -> tuple[int, bytes, str]:
    global _last_call
    async with _lock:
        wait = _last_call + MIN_SPACING - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call = time.monotonic()
    stats["upstream"] += 1
    async with session.get(f"{UPSTREAM}/0/public/{method}", params=query) as r:
        return r.status, await r.read(), r.headers.get("Content-Type", "application/json")


async def handle(request: web.Request) -> web.Response:
    stats["requests"] += 1
    method = request.match_info["method"]
    key, query = cache_key(method, dict(request.query))
    now = time.time()
    hit = cache.get(key)
    if hit and is_fresh(method, query, hit[0], now):
        return web.Response(status=hit[1], body=hit[2], content_type=hit[3].split(";")[0])
    if key in inflight:
        status, body, ctype = await asyncio.shield(inflight[key])
    else:
        fut = asyncio.get_running_loop().create_future()
        inflight[key] = fut
        try:
            status, body, ctype = await upstream(request.app["session"], method, query)
            fut.set_result((status, body, ctype))
        except Exception as e:  # surface as a 502 so ccxt retries
            stats["errors"] += 1
            status, body, ctype = 502, str(e).encode(), "text/plain"
            fut.set_result((status, body, ctype))
        finally:
            inflight.pop(key, None)
        # Only cache clean responses; Kraken reports rate limits as 200 + {"error": [...]}
        if status == 200 and b'"error":[]' in body.replace(b" ", b""):
            cache[key] = (time.time(), status, body, ctype)
        else:
            stats["errors"] += 1
    return web.Response(status=status, body=body, content_type=ctype.split(";")[0])


async def stats_view(_: web.Request) -> web.Response:
    return web.json_response({**stats, "cached_keys": len(cache)})


async def on_startup(app: web.Application) -> None:
    app["session"] = ClientSession(timeout=ClientTimeout(total=20))


async def on_cleanup(app: web.Application) -> None:
    await app["session"].close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8899
    app = web.Application()
    app.router.add_get("/0/public/{method}", handle)
    app.router.add_get("/stats", stats_view)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    web.run_app(app, host="127.0.0.1", port=port, print=None)
