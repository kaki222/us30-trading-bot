"""
mobile_api.py — Layer 7: read-only status + safe discretionary controls,
served over HTTP for the Android "app" (a PWA — see mobile_app/ for the
frontend, and ARCHITECTURE.md's "Mobile app (PWA)" section for the full
story of why this exists and what it deliberately can't do).

Why this file exists at all: MetaTrader5's Python package only works on
Windows, wired to a locally-running MT5 terminal via COM/DLL calls — there
is no MT5 Python API for Android (or Linux/Mac). So the actual bot —
connect(), build_live_features(), run_scheduled.py's cycle — has to keep
living on this Windows machine no matter what. This file doesn't change
that; it's a second, independent MT5-connected process (same pattern as
live_monitor.py already is) that exposes read-only status and the exact
same safe controls the dashboard's buttons already call, over a small
Flask server on your home network — so a phone on the same WiFi can see
and touch the same manual_overrides.json that live_monitor.py and
run_scheduled.py already read/write.

Security note (2026-07-27, user's call — "home WiFi only"): this binds to
0.0.0.0 so any device on your LAN can reach it — deliberately NOT exposed
to the open internet (no port forwarding, no tunnel; if that's ever
wanted, it needs a real auth layer on top of this, not just the token
below). The three POST endpoints require a shared token (?token=... or an
X-Auth-Token header, generated once into data/mobile_token.txt, gitignored
same as the rest of data/) purely so a random device on the WiFi (guest
network, IoT gadget) can't flip your bias/pause state without at least
having that token — it is NOT meant to withstand a real attacker, matching
the "home WiFi only" scope this was built for. GET endpoints (status/
journal) are unauthenticated — read-only numbers off a $500k demo account,
the same sensitivity as what's already visible in live_monitor.py's window.

What this can NEVER do, by construction (same boundary as
live_monitor.py's buttons): no order placement, no close_position(), no
run_once() call. The only functions this calls that mutate anything are
manual_overrides.set_bias() / set_paused_now() / set_key_level() — which
can only ever mute, downsize, or skip a trade the mechanical strategy
would otherwise take, never invent or place one. See run_scheduled.py's
module docstring for exactly how those get applied.

Run alongside (not instead of) run_scheduled.py's Task Scheduler job and
live_monitor.py — independent processes, all three read/write the same
data/manual_overrides.json and data/journal.jsonl:

    (venv) PS> python -m trader.l7_execution.mobile_api "C:\\path\\to\\terminal64.exe"

Then on your phone (same WiFi): open Chrome to
http://<this-PC's-LAN-IP>:8765/?token=<printed-on-startup>, and use
Chrome's menu -> "Add to Home Screen" (or the "Install app" prompt it
offers) to get a real app icon that opens full-screen, no browser chrome.
Find your PC's LAN IP with `ipconfig` (the "IPv4 Address" under your
active adapter) — Windows Firewall will likely prompt to allow Python
through on first run; allow it for Private networks only.

Needs Flask in addition to what run_scheduled.py already needs:
    (venv) PS> pip install flask
"""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from . import (
    connect, shutdown, SYMBOL_MAP, TIMEFRAME_SECONDS,
    build_live_features, LiveCircuitBreaker, account_summary, get_position_info,
)
from .run_scheduled import TIMEFRAME, PARAMS, MAGIC
from .manual_overrides import load_overrides, set_bias, set_key_level, set_paused_now
from .journal import read_entries

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TOKEN_PATH = DATA_DIR / "mobile_token.txt"
STATIC_DIR = Path(__file__).resolve().parent / "mobile_app"

app = Flask(__name__, static_folder=None)


def _load_or_create_token() -> str:
    """One persistent random token, generated once and reused across
    restarts (data/mobile_token.txt, gitignored) — so the phone doesn't
    need re-entering it every time this server restarts. Delete the file
    to rotate it."""
    if TOKEN_PATH.exists():
        return TOKEN_PATH.read_text().strip()
    token = secrets.token_urlsafe(16)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(token)
    return token


TOKEN = _load_or_create_token()


def _check_token() -> bool:
    supplied = request.args.get("token") or request.headers.get("X-Auth-Token")
    return supplied == TOKEN


def _regime_snapshot(symbol_key: str) -> dict:
    """Same read live_monitor.py's _regime_info() does, reimplemented
    standalone here so this file doesn't need to import matplotlib/
    mplfinance just to reuse one small function — the underlying
    computation (last row of build_live_features()) is identical. Reports
    the STRATEGY's own MACD (12,26,9, from build_live_features/l2_features
    — the one RegimeConfluenceStrategy's macd_bull/macd_bear gate actually
    uses), not live_monitor.py's chart-only fast display variant (5,13,9)
    — this is a status number, not a chart, so the real gating value is
    the more useful one here."""
    symbol = SYMBOL_MAP[symbol_key]
    df = build_live_features(symbol, er_length=PARAMS.get("er_length", 20), timeframe=TIMEFRAME)
    row = df.iloc[-1]
    er = row["er"]
    trending = bool(er > PARAMS["er_threshold"]) if pd.notna(er) else False
    regime = "TRENDING" if trending else ("CHOP" if pd.notna(er) else "warming up")
    breaker = LiveCircuitBreaker(symbol=symbol, magic=MAGIC, bar_seconds=TIMEFRAME_SECONDS[TIMEFRAME.upper()])
    return {
        "close": float(row["Close"]),
        "er": None if pd.isna(er) else float(er),
        "macd_hist": None if pd.isna(row["macd_hist"]) else float(row["macd_hist"]),
        "regime": regime,
        "cooldown": breaker.in_cooldown(),
    }


@app.route("/api/status")
def api_status():
    overrides = load_overrides()
    symbols = {}
    for symbol_key, symbol in SYMBOL_MAP.items():
        symbols[symbol_key] = {
            **_regime_snapshot(symbol_key),
            "bias": overrides["bias"].get(symbol_key),
            "paused_now": overrides["paused_now"].get(symbol_key, False),
            "key_levels": overrides["key_levels"].get(symbol_key, {}),
            "position": get_position_info(symbol, MAGIC),
        }
    return jsonify({"account": account_summary(), "symbols": symbols})


@app.route("/api/journal")
def api_journal():
    n = int(request.args.get("n", 20))
    entries = read_entries()
    return jsonify(list(reversed(entries[-n:])))  # most recent first


@app.route("/api/bias", methods=["POST"])
def api_set_bias():
    if not _check_token():
        return jsonify({"error": "bad token"}), 401
    body = request.get_json(force=True) or {}
    symbol_key, value = body.get("symbol_key"), body.get("value")
    if symbol_key not in SYMBOL_MAP:
        return jsonify({"error": f"unknown symbol_key {symbol_key!r}"}), 400
    if value not in (None, "long", "short"):
        return jsonify({"error": f"value must be long/short/null, got {value!r}"}), 400
    set_bias(symbol_key, value)
    return jsonify({"ok": True})


@app.route("/api/pause", methods=["POST"])
def api_set_pause():
    if not _check_token():
        return jsonify({"error": "bad token"}), 401
    body = request.get_json(force=True) or {}
    symbol_key, value = body.get("symbol_key"), bool(body.get("value"))
    if symbol_key not in SYMBOL_MAP:
        return jsonify({"error": f"unknown symbol_key {symbol_key!r}"}), 400
    set_paused_now(symbol_key, value)
    return jsonify({"ok": True})


@app.route("/api/key_level", methods=["POST"])
def api_set_key_level():
    if not _check_token():
        return jsonify({"error": "bad token"}), 401
    body = request.get_json(force=True) or {}
    symbol_key, which, value = body.get("symbol_key"), body.get("which"), body.get("value")
    if symbol_key not in SYMBOL_MAP:
        return jsonify({"error": f"unknown symbol_key {symbol_key!r}"}), 400
    if which not in ("invalidation_up", "invalidation_down"):
        return jsonify({"error": f"which must be invalidation_up/invalidation_down, got {which!r}"}), 400
    set_key_level(symbol_key, which, None if value in (None, "") else float(value))
    return jsonify({"ok": True})


@app.route("/")
@app.route("/<path:filename>")
def static_files(filename="index.html"):
    return send_from_directory(STATIC_DIR, filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mt5_path", nargs="?", default=None,
                         help="path to terminal64.exe (optional if MT5 is already logged in)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    connect(path=args.mt5_path)
    acct = account_summary()
    print(f"Connected: login={acct['login']} server={acct['server']} equity={acct['equity']}")
    print(f"Token: {TOKEN}")
    print(f"On your phone (same WiFi as this PC): http://<this PC's LAN IP>:{args.port}/?token={TOKEN}")
    print("Find the LAN IP with `ipconfig` (IPv4 Address, active adapter). "
          "Allow Python through Windows Firewall for Private networks if prompted.")
    try:
        app.run(host="0.0.0.0", port=args.port)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
