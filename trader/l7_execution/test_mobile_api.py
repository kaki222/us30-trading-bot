"""
test_mobile_api.py — offline smoke test for mobile_api.py, same spirit as
the other test_*.py files in this package: mocks every MT5-touching call
(no real terminal needed, runs on Linux) and drives the actual Flask app
through its test client, so this is a real check of routing/auth/response
shape, not just an import check.

Run:  PYTHONPATH=. python -m trader.l7_execution.test_mobile_api
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Point manual_overrides / journal at a throwaway temp data dir BEFORE
# importing mobile_api, since load_or_create_token() and the overrides/
# journal modules resolve their paths at import time relative to the real
# repo's data/ dir otherwise - this test shouldn't touch the user's real
# manual_overrides.json or journal.jsonl.
_tmp_data = Path(tempfile.mkdtemp()) / "data"
_tmp_data.mkdir(parents=True)

from trader.l7_execution import mobile_api as api
from trader.l7_execution import live_monitor as lm
from trader.l7_execution import manual_overrides as mo
from trader.l7_execution import journal as jr

api.DATA_DIR = _tmp_data
api.TOKEN_PATH = _tmp_data / "mobile_token.txt"
mo.DATA_DIR = _tmp_data
mo.OVERRIDES_PATH = _tmp_data / "manual_overrides.json"
mo.OVERRIDES_LOG_PATH = _tmp_data / "manual_overrides_log.jsonl"

# journal.py's append_entry/read_entries default path= is bound at
# function-definition time (module import), so reassigning jr.JOURNAL_PATH
# after the fact does NOT change what those functions actually use unless
# path= is passed explicitly each call - patch both call sites (the test's
# own jr.append_entry() calls below, and mobile_api's read_entries()) to
# always pass the temp path instead, so this test never touches the real
# repo's data/journal.jsonl.
_journal_path = _tmp_data / "journal.jsonl"
jr.JOURNAL_PATH = _journal_path
api.read_entries = lambda: jr.read_entries(path=_journal_path)

api.TOKEN = api._load_or_create_token()

N = 60
idx = pd.date_range("2026-07-01", periods=N, freq="4h")
rng = np.random.default_rng(7)
close = 4000 + np.cumsum(rng.normal(0, 5, N))


def fake_build_live_features(symbol, er_length=20, timeframe=None, count=800):
    return pd.DataFrame({
        "Open": close, "High": close + 2, "Low": close - 2, "Close": close,
        "Volume": np.full(N, 1000.0),
        "ma_360": pd.Series(close).rolling(10, min_periods=1).mean(),
        "ma_200": pd.Series(close).rolling(10, min_periods=1).mean(),
        "ma_89": pd.Series(close).rolling(10, min_periods=1).mean(),
        "ema_21": pd.Series(close).ewm(span=21).mean(),
        "ema_8": pd.Series(close).ewm(span=8).mean(),
        "macd_hist": rng.normal(0, 2, N),
        "atr_14": np.full(N, 5.0),
        "er": rng.uniform(0, 1, N),
    }, index=idx)


def fake_account_summary():
    return {"login": 1, "server": "TEST", "balance": 500000.0, "equity": 500123.45,
            "margin": 0.0, "margin_free": 500123.45, "leverage": 30, "currency": "USD"}


def fake_get_position_info(symbol, magic):
    return None if symbol != "GOLD" else {
        "ticket": 1, "direction": "long", "volume": 0.1, "price_open": 4000.0,
        "price_current": 4010.0, "sl": 3950.0, "tp": 4100.0, "profit": 100.0, "pnl_pct": 0.0002,
    }


class FakeBreaker:
    def __init__(self, *a, **kw): pass
    def in_cooldown(self): return False


api.connect = lambda path=None: None
api.shutdown = lambda: None
api.build_live_features = fake_build_live_features
api.account_summary = fake_account_summary
api.get_position_info = fake_get_position_info
api.LiveCircuitBreaker = FakeBreaker

# /api/chart/<key>.png goes through lm._redraw_column(), which calls
# live_monitor.py's OWN build_live_features/LiveCircuitBreaker bindings
# (`from . import build_live_features, LiveCircuitBreaker` inside
# live_monitor.py) - a separate name binding from mobile_api's, so
# patching api.* above does not reach code running inside lm.*. Same
# pattern render_layout_test.py uses for live_monitor.py directly.
lm.connect = lambda path=None: None
lm.shutdown = lambda: None
lm.build_live_features = fake_build_live_features
lm.LiveCircuitBreaker = FakeBreaker

client = api.app.test_client()
failures = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failures.append(name)


# --- status ---
r = client.get("/api/status")
check("GET /api/status 200", r.status_code == 200)
body = r.get_json()
check("status has account.equity", body["account"]["equity"] == 500123.45)
check("status has both symbols", set(body["symbols"].keys()) == {"US30", "GOLD"})
check("US30 flat", body["symbols"]["US30"]["position"] is None)
check("GOLD has position", body["symbols"]["GOLD"]["position"]["direction"] == "long")
check("ER is a float 0-1", 0.0 <= body["symbols"]["US30"]["er"] <= 1.0)
check("regime is TRENDING or CHOP", body["symbols"]["US30"]["regime"] in ("TRENDING", "CHOP"))
check("bias defaults to None (neutral)", body["symbols"]["US30"]["bias"] is None)

# --- chart PNG (reuses live_monitor.py's own drawing code, Agg backend) ---
r = client.get("/api/chart/US30.png")
check("GET /api/chart/US30.png 200", r.status_code == 200)
check("chart is a real PNG (magic bytes)", r.data[:8] == b"\x89PNG\r\n\x1a\n")
check("chart content-type is image/png", r.content_type == "image/png")
r = client.get("/api/chart/NOPE.png")
check("unknown symbol chart -> 404", r.status_code == 404)

# --- journal (empty at first) ---
r = client.get("/api/journal")
check("GET /api/journal 200 empty", r.status_code == 200 and r.get_json() == [])

jr.append_entry("US30", "H4", 100001, {"action": "skip", "reason": "no signal"}, path=_journal_path)
jr.append_entry("GOLD", "H4", 100001, {"action": "trade", "signal": {"signal": "long"}, "trade": {}}, path=_journal_path)
r = client.get("/api/journal")
entries = r.get_json()
check("journal has 2 entries", len(entries) == 2)
check("journal most-recent-first", entries[0]["symbol_key"] == "GOLD")

# --- auth: POST without token rejected ---
r = client.post("/api/bias", json={"symbol_key": "US30", "value": "long"})
check("POST /api/bias no token -> 401", r.status_code == 401)
check("bias unchanged after rejected POST", mo.load_overrides()["bias"]["US30"] is None)

# --- auth: POST with wrong token rejected ---
r = client.post("/api/bias?token=wrong", json={"symbol_key": "US30", "value": "long"})
check("POST /api/bias wrong token -> 401", r.status_code == 401)

# --- POST with correct token actually writes through manual_overrides ---
r = client.post(f"/api/bias?token={api.TOKEN}", json={"symbol_key": "US30", "value": "long"})
check("POST /api/bias correct token -> 200", r.status_code == 200)
check("bias actually set in manual_overrides.json", mo.load_overrides()["bias"]["US30"] == "long")

r = client.post("/api/pause", json={"symbol_key": "GOLD", "value": True},
                 headers={"X-Auth-Token": api.TOKEN})
check("POST /api/pause via header token -> 200", r.status_code == 200)
check("paused_now actually set", mo.load_overrides()["paused_now"]["GOLD"] is True)

r = client.post(f"/api/key_level?token={api.TOKEN}",
                 json={"symbol_key": "GOLD", "which": "invalidation_up", "value": 4147.0})
check("POST /api/key_level -> 200", r.status_code == 200)
check("key level actually set", mo.load_overrides()["key_levels"]["GOLD"]["invalidation_up"] == 4147.0)

# --- validation: bad symbol/field rejected with 400, not silently accepted ---
r = client.post(f"/api/bias?token={api.TOKEN}", json={"symbol_key": "NOPE", "value": "long"})
check("bad symbol_key -> 400", r.status_code == 400)
r = client.post(f"/api/bias?token={api.TOKEN}", json={"symbol_key": "US30", "value": "sideways"})
check("bad bias value -> 400", r.status_code == 400)

# --- static frontend actually served ---
r = client.get("/")
check("GET / serves index.html", r.status_code == 200 and b"US30-TRADING-BOT" in r.data)
r = client.get("/manifest.json")
check("GET /manifest.json 200", r.status_code == 200)
manifest = r.get_json()
check("manifest has standalone display", manifest["display"] == "standalone")
r = client.get("/icons/icon-192.png")
check("GET /icons/icon-192.png 200", r.status_code == 200)

shutil.rmtree(_tmp_data.parent, ignore_errors=True)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("ALL PASSED")
