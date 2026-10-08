# Fleet status
updated 2026-10-08 22:37 UTC
```
Traceback (most recent call last):
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/urllib3/connection.py", line 239, in _new_conn
    sock = connection.create_connection(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/urllib3/util/connection.py", line 85, in create_connection
    raise err
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/urllib3/util/connection.py", line 73, in create_connection
    sock.connect(sa)
ConnectionRefusedError: [Errno 111] Connection refused

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/urllib3/connectionpool.py", line 793, in urlopen
    response = self._make_request(
               ^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/urllib3/connectionpool.py", line 499, in _make_request
    conn.request(
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/urllib3/connection.py", line 567, in request
    self.endheaders()
  File "/opt/hostedtoolcache/Python/3.12.15/x64/lib/python3.12/http/client.py", line 1381, in endheaders
    self._send_output(message_body, encode_chunked=encode_chunked)
  File "/opt/hostedtoolcache/Python/3.12.15/x64/lib/python3.12/http/client.py", line 1141, in _send_output
    self.send(msg)
  File "/opt/hostedtoolcache/Python/3.12.15/x64/lib/python3.12/http/client.py", line 1085, in send
    self.connect()
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/urllib3/connection.py", line 398, in connect
    self.sock = self._new_conn()
                ^^^^^^^^^^^^^^^^
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/urllib3/connection.py", line 254, in _new_conn
    raise NewConnectionError(
urllib3.exceptions.NewConnectionError: HTTPConnection(host='127.0.0.1', port=8899): Failed to establish a new connection: [Errno 111] Connection refused

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/requests/adapters.py", line 696, in send
    resp = conn.urlopen(
           ^^^^^^^^^^^^^
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/urllib3/connectionpool.py", line 847, in urlopen
    retries = retries.increment(
              ^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/urllib3/util/retry.py", line 555, in increment
    raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
urllib3.exceptions.MaxRetryError: HTTPConnectionPool(host='127.0.0.1', port=8899): Max retries exceeded with url: /0/public/Assets (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8899): Failed to establish a new connection: [Errno 111] Connection refused"))

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/ccxt/base/exchange.py", line 607, in fetch
    response = self.session.request(
               ^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/requests/sessions.py", line 651, in request
    resp = self.send(prep, **send_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/requests/sessions.py", line 784, in send
    r = adapter.send(request, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/requests/adapters.py", line 729, in send
    raise ConnectionError(e, request=request)
requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8899): Max retries exceeded with url: /0/public/Assets (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8899): Failed to establish a new connection: [Errno 111] Connection refused"))

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/scripts/fleet.py", line 312, in <module>
    {"start": lambda: start(sys.argv[2:]), "stop": stop, "status": status, "rotate": rotate, "retire": lambda: retire(sys.argv[2:])}[cmd]()
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/scripts/fleet.py", line 277, in status
    rows, bh = leaderboard()
               ^^^^^^^^^^^^^
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/scripts/fleet.py", line 219, in leaderboard
    now = now or prices()
                 ^^^^^^^^
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/scripts/fleet.py", line 182, in prices
    return {p: t["last"] for p, t in ex.fetch_tickers(pairs).items()}
                                     ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/ccxt/kraken.py", line 1110, in fetch_tickers
    self.load_markets()
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/ccxt/base/exchange.py", line 1914, in load_markets
    currencies = self.fetch_currencies()
                 ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/ccxt/kraken.py", line 801, in fetch_currencies
    response = self.publicGetAssets(params)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/ccxt/base/types.py", line 43, in unbound_method
    return _self.request(self.path, self.api, self.method, params, config=self.config)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/ccxt/base/exchange.py", line 5823, in request
    return self.fetch2(path, api, method, params, headers, body, config)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/ccxt/base/exchange.py", line 5817, in fetch2
    raise e
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/ccxt/base/exchange.py", line 5800, in fetch2
    response = self.fetch(request['url'], request['method'], request['headers'], request['body'])
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/crypto-paper-fleet/crypto-paper-fleet/.venv/lib/python3.12/site-packages/ccxt/base/exchange.py", line 663, in fetch
    raise NetworkError(details) from e
ccxt.base.errors.NetworkError: kraken GET http://127.0.0.1:8899/0/public/Assets
```
