"""
manual_overrides.py — shared, live-editable store for the discretionary
BIAS / KEY_LEVELS / "paused right now" flags, so a click in
live_monitor.py's dashboard actually reaches run_scheduled.py's next
cycle instead of requiring a source-code edit every time (which is how
this worked before 2026-07-25 — BIAS/KEY_LEVELS were hardcoded dicts
directly in run_scheduled.py).

data/manual_overrides.json is the live state both scripts read/write.
The DEFAULT_* dicts below only seed that file the first time it doesn't
exist yet (fresh checkout, or after deleting the file to reset) — once
it exists, it's the only thing either script actually reads; editing
DEFAULT_BIAS etc. here after that point does nothing until the file is
deleted. This intentionally does NOT cover PAUSE_WINDOWS (the
pre-scheduled calendar date ranges in run_scheduled.py, e.g. GOLD's
Aug 7-10 whipsaw window) — those stay hardcoded there; `paused_now`
below is the separate, immediate, click-to-toggle pause layered on top
of them.

Every change goes through set_bias()/set_key_level()/set_paused_now(),
which write the new state AND append one line to
data/manual_overrides_log.jsonl (timestamp, field, symbol, old, new) —
same idea as journal.py, so there's a record of exactly when and why a
bias/pause/level changed, not just its current value.

Not thread-safe against two writers at the exact same instant — fine
here since only one human is expected to be clicking one dashboard at
a time, and the write itself is a single atomic os.replace(), so the
worst case is two near-simultaneous clicks landing in the opposite
order, never a half-written/corrupted file.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OVERRIDES_PATH = DATA_DIR / "manual_overrides.json"
OVERRIDES_LOG_PATH = DATA_DIR / "manual_overrides_log.jsonl"

# Seed values only — see module docstring. Match what BIAS/KEY_LEVELS
# used to hardcode directly in run_scheduled.py before this file existed.
DEFAULT_BIAS = {"US30": None, "GOLD": None}
DEFAULT_KEY_LEVELS = {
    "US30": {"invalidation_up": None, "invalidation_down": None},
    "GOLD": {"invalidation_up": 4180.0, "invalidation_down": 3958.0},
}
DEFAULT_PAUSED_NOW = {"US30": False, "GOLD": False}


def _defaults() -> dict:
    return {
        "bias": dict(DEFAULT_BIAS),
        "key_levels": {k: dict(v) for k, v in DEFAULT_KEY_LEVELS.items()},
        "paused_now": dict(DEFAULT_PAUSED_NOW),
    }


def load_overrides() -> dict:
    """Read the live override state, seeding the file with defaults the first time it's missing."""
    if not OVERRIDES_PATH.exists():
        state = _defaults()
        save_overrides(state)
        return state
    with open(OVERRIDES_PATH) as f:
        state = json.load(f)
    # Backfill any keys added since a file was first created (e.g. a new symbol added later).
    defaults = _defaults()
    for top_key, default_val in defaults.items():
        state.setdefault(top_key, default_val)
        if isinstance(default_val, dict):
            for symbol_key, sub_default in default_val.items():
                state[top_key].setdefault(symbol_key, sub_default)
    return state


def save_overrides(state: dict) -> None:
    """Atomic write — write to a temp file then os.replace(), so a reader never sees a half-written file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = OVERRIDES_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, OVERRIDES_PATH)


def _log_change(field: str, symbol_key: str, old, new) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "field": field,
        "symbol_key": symbol_key,
        "old": old,
        "new": new,
    }
    with open(OVERRIDES_LOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def set_bias(symbol_key: str, value: str | None) -> None:
    """value: "long", "short", or None (neutral)."""
    state = load_overrides()
    old = state["bias"].get(symbol_key)
    if old == value:
        return
    state["bias"][symbol_key] = value
    save_overrides(state)
    _log_change("bias", symbol_key, old, value)


def set_key_level(symbol_key: str, which: str, value: float | None) -> None:
    """which: "invalidation_up" or "invalidation_down"."""
    state = load_overrides()
    levels = state["key_levels"].setdefault(symbol_key, {"invalidation_up": None, "invalidation_down": None})
    old = levels.get(which)
    if old == value:
        return
    levels[which] = value
    save_overrides(state)
    _log_change(f"key_level.{which}", symbol_key, old, value)


def set_paused_now(symbol_key: str, value: bool) -> None:
    state = load_overrides()
    old = state["paused_now"].get(symbol_key, False)
    if old == value:
        return
    state["paused_now"][symbol_key] = value
    save_overrides(state)
    _log_change("paused_now", symbol_key, old, value)


def read_change_log() -> list[dict]:
    """All manual-override changes ever made, oldest first. Empty list if none yet."""
    if not OVERRIDES_LOG_PATH.exists():
        return []
    entries = []
    with open(OVERRIDES_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
