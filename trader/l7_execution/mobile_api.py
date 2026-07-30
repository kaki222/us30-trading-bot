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

Discretionary controls (bias/pause/key-level) can NEVER place an order,
by construction: the only functions they call are manual_overrides.
set_bias()/set_paused_now()/set_key_level(), which can only ever mute,
downsize, or skip a trade the mechanical strategy would otherwise take,
never invent or place one. See run_scheduled.py's module docstring for
exactly how those get applied.

Manual orders (2026-07-30 — the one thing in this whole repo that CAN
place a real order): /api/manual_mode toggles a symbol between Auto (the
mechanical engine runs it, same as always) and Manual (run_scheduled.py
skips it entirely — see its module docstring — and these two endpoints
become usable for it):
  POST /api/manual_order/validate — always dry_run=True, a pure preview
    (lot size, SL/TP, what request WOULD be sent), reuses place_trade()
    from this package rather than reimplementing order math.
  POST /api/manual_order/send — the only call anywhere in this repo that
    can reach place_trade(dry_run=False). Requires ALL of: token auth,
    manual_mode=True already set for that symbol, body confirm=true, no
    existing open position under MANUAL_MAGIC on that symbol, AND — the
    hard safeguard for the demo-only scope the user set 2026-07-30 — the
    connected account's login must equal DEMO_LOGIN below; anything else
    (including the user's own real-money account, 330507861) is refused
    with a 403 before place_trade() is ever called, no override. Uses
    MANUAL_MAGIC (100002), a distinct magic from the mechanical engine's
    MAGIC (100001), so a manual fill can never be mistaken for a
    mechanical one by has_open_position()/LiveCircuitBreaker/journal
    filtering, and vice versa — the two can coexist on different symbols
    (or even the same symbol, though manual_mode blocks the mechanical
    engine from also trading it) without confusing each other's state.
    Every send is journaled via journal.append_entry() same as a
    mechanical trade, so journal_summary.py shows manual fills too.
This is still a demo-account tool: nothing here reaches for the real
account on its own, and the DEMO_LOGIN check is a hard stop, not a
warning, if that ever changes.

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
import io
import secrets
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, Response

# Agg (non-interactive, file/buffer-only) backend, set BEFORE live_monitor
# (or anything else) gets a chance to `import matplotlib.pyplot` — this
# process is a headless server, so the normal interactive backend
# live_monitor.py uses when run standalone (TkAgg/QtAgg, opens a real
# window) would either fail with no display attached or, worse, try to
# actually open a GUI window on whatever machine runs this. matplotlib's
# backend is a process-wide global set on first pyplot import, so this
# has to happen here, first, before the `from . import live_monitor` below.
import matplotlib
matplotlib.use("Agg")

from . import (
    connect, shutdown, SYMBOL_MAP, TIMEFRAME_SECONDS,
    build_live_features, LiveCircuitBreaker, account_summary, get_position_info,
    place_trade, has_open_position,
)
from . import live_monitor as lm
from .run_scheduled import TIMEFRAME, PARAMS, MAGIC
from .manual_overrides import load_overrides, set_bias, set_key_level, set_paused_now, set_manual_mode
from .journal import read_entries, append_entry

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TOKEN_PATH = DATA_DIR / "mobile_token.txt"
STATIC_DIR = Path(__file__).resolve().parent / "mobile_app"

# Manual orders get their own magic, distinct from run_scheduled.py's
# MAGIC (100001) — see module docstring's "Manual orders" section.
MANUAL_MAGIC = 100002

# Hard demo-only guard for /api/manual_order/send — see module docstring.
# The user's real-money account (330507861, same broker/server family,
# see l7_execution/__init__.py's SYMBOL_MAP comment) must NEVER be
# reachable through this endpoint; this is a login-number equality
# check, not a setting, so there's nothing to accidentally misconfigure.
DEMO_LOGIN = 345899957

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


def _render_chart_png(symbol_key: str, bars: int, visible_bars: int) -> bytes:
    """
    Same chart live_monitor.py's desktop window draws for one symbol —
    price + EMA8/EMA21/MA89/MA200/MA360 + swing hi/lo, ER, and MACD — but
    rendered once to a PNG buffer instead of an interactive window. Reuses
    live_monitor.py's own `_redraw_column()` (its exact drawing code,
    colors, and the display-only MACD(5,13,9)) rather than re-implementing
    any of it here, so the phone's chart always matches the desktop
    dashboard's — one drawing function, two outputs (a live window vs. a
    PNG snapshot), not two copies of the plotting logic to keep in sync.
    """
    symbol = SYMBOL_MAP[symbol_key]
    fig, (price_ax, er_ax, macd_ax) = lm.plt.subplots(
        nrows=3, ncols=1, figsize=(7.2, 8.6),
        gridspec_kw={"height_ratios": [3, 1, 1]}, sharex=True,
    )
    fig.patch.set_facecolor(lm.BG)
    bias = load_overrides()["bias"].get(symbol_key)
    lm._redraw_column(
        symbol_key, symbol, price_ax, er_ax, macd_ax, bars, TIMEFRAME, visible_bars,
        {}, bias, show_ylabels=True, n_columns=1, panel_width_in=0.0,
        # compact=True: the phone card already shows symbol/price/ER/regime/
        # bias/breaker in its own header right above this image, so skip
        # live_monitor.py's redundant on-chart title text here (desktop's
        # own window, with no such card header, still gets it).
        compact=True,
    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=lm.BG, dpi=130)
    lm.plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


@app.route("/api/chart/<symbol_key>.png")
def api_chart(symbol_key):
    if symbol_key not in SYMBOL_MAP:
        return jsonify({"error": f"unknown symbol_key {symbol_key!r}"}), 404
    bars = int(request.args.get("bars", 150))
    visible_bars = int(request.args.get("visible_bars", 40))
    png = _render_chart_png(symbol_key, bars, visible_bars)
    return Response(png, mimetype="image/png", headers={"Cache-Control": "no-store"})


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
            "manual_mode": overrides.get("manual_mode", {}).get(symbol_key, False),
            "position": get_position_info(symbol, MAGIC),
            "manual_position": get_position_info(symbol, MANUAL_MAGIC),
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


@app.route("/api/manual_mode", methods=["POST"])
def api_set_manual_mode():
    if not _check_token():
        return jsonify({"error": "bad token"}), 401
    body = request.get_json(force=True) or {}
    symbol_key, value = body.get("symbol_key"), bool(body.get("value"))
    if symbol_key not in SYMBOL_MAP:
        return jsonify({"error": f"unknown symbol_key {symbol_key!r}"}), 400
    set_manual_mode(symbol_key, value)
    return jsonify({"ok": True})


def _parse_manual_order_body(body: dict):
    """Shared validation for validate/send below. Returns (params, None)
    on success or (None, (json_response, status_code)) on failure — every
    field is checked before either endpoint touches place_trade(), so a
    typo'd field never silently falls back to some default on a real
    order."""
    symbol_key = body.get("symbol_key")
    direction = body.get("direction")
    sl, tp = body.get("sl"), body.get("tp")
    risk_pct = body.get("risk_pct", 0.01)

    if symbol_key not in SYMBOL_MAP:
        return None, (jsonify({"error": f"unknown symbol_key {symbol_key!r}"}), 400)
    if direction not in ("long", "short"):
        return None, (jsonify({"error": f"direction must be long/short, got {direction!r}"}), 400)
    try:
        sl, tp, risk_pct = float(sl), float(tp), float(risk_pct)
    except (TypeError, ValueError):
        return None, (jsonify({"error": "sl/tp/risk_pct must be numbers"}), 400)
    if sl <= 0 or tp <= 0:
        return None, (jsonify({"error": "sl/tp must be positive prices"}), 400)
    if not (0 < risk_pct <= 0.05):
        return None, (jsonify({"error": f"risk_pct must be in (0, 0.05], got {risk_pct}"}), 400)

    return {"symbol_key": symbol_key, "direction": direction, "sl": sl, "tp": tp, "risk_pct": risk_pct}, None


def _preview_manual_order(params: dict):
    """Runs place_trade(dry_run=True) (which fetches the live tick itself
    — this file doesn't import MetaTrader5 directly, l7_execution/
    __init__.py already owns that try/import), then sanity-checks sl/tp
    landed on the correct side of the actual live price for the given
    direction. Returns (preview_dict, None) or (None, (json, status))."""
    symbol = SYMBOL_MAP[params["symbol_key"]]
    preview = place_trade(
        symbol, params["direction"], params["sl"], params["tp"],
        risk_pct=params["risk_pct"], leverage=30, magic=MANUAL_MAGIC, dry_run=True,
    )
    price = preview["would_send"]["price"]
    direction, sl, tp = params["direction"], params["sl"], params["tp"]
    if direction == "long" and not (sl < price < tp):
        return None, (jsonify({"error": f"long needs sl < price < tp — got sl={sl}, price={price}, tp={tp}"}), 400)
    if direction == "short" and not (tp < price < sl):
        return None, (jsonify({"error": f"short needs tp < price < sl — got tp={tp}, price={price}, sl={sl}"}), 400)
    return preview, None


@app.route("/api/manual_order/validate", methods=["POST"])
def api_manual_order_validate():
    """Always a dry run — pure preview, never touches manual_mode or the
    token gate, so the app can show a live preview as the user types
    without needing the token in the loop yet."""
    body = request.get_json(force=True) or {}
    params, err = _parse_manual_order_body(body)
    if err:
        return err
    preview, err = _preview_manual_order(params)
    if err:
        return err
    return jsonify({"ok": True, "preview": preview})


@app.route("/api/manual_order/send", methods=["POST"])
def api_manual_order_send():
    """The only path in this whole app that can place a real order — see
    module docstring for the full list of gates. Every one of them is
    checked here, in order, before place_trade(dry_run=False) is called;
    any failure returns before that point, order never sent."""
    if not _check_token():
        return jsonify({"error": "bad token"}), 401

    body = request.get_json(force=True) or {}
    if not body.get("confirm"):
        return jsonify({"error": "confirm must be true to send a real order"}), 400

    params, err = _parse_manual_order_body(body)
    if err:
        return err
    symbol_key, symbol = params["symbol_key"], SYMBOL_MAP[params["symbol_key"]]

    if not load_overrides().get("manual_mode", {}).get(symbol_key, False):
        return jsonify({"error": f"{symbol_key} is not in Manual mode — flip its Auto/Manual "
                                  f"switch first"}), 403

    acct = account_summary()
    if acct["login"] != DEMO_LOGIN:
        # Hard stop — see module docstring's "Manual orders" section and
        # DEMO_LOGIN's comment above. This is not meant to ever trigger
        # given how this app is set up to only ever run against the demo
        # terminal, but it stays a real check, not just a comment.
        return jsonify({"error": f"refusing to send: connected account {acct['login']} is not "
                                  f"the demo account ({DEMO_LOGIN}) this feature is scoped to"}), 403

    if has_open_position(symbol, MANUAL_MAGIC):
        return jsonify({"error": f"{symbol_key} already has an open manual position — close it "
                                  f"first"}), 409

    # Re-run the same sl/tp-vs-live-price sanity check /validate does —
    # the price may have moved since the user last hit Validate, and this
    # is the one call in the app allowed to actually place an order, so
    # it re-checks rather than trusting a preview from a moment ago.
    _, err = _preview_manual_order(params)
    if err:
        return err

    trade = place_trade(
        symbol, params["direction"], params["sl"], params["tp"],
        risk_pct=params["risk_pct"], leverage=30, magic=MANUAL_MAGIC,
        comment="l7_manual", dry_run=False,
    )
    result = {"action": "manual_trade", "direction": params["direction"], "trade": trade}
    append_entry(symbol_key, TIMEFRAME, MANUAL_MAGIC, result)
    return jsonify({"ok": True, "trade": trade})


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
