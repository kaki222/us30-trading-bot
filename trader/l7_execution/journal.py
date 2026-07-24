"""
journal.py — Layer 7 support: append-only trade/decision journal.

Every run_once() result - "skip" (with a reason) or "trade" (dry-run or
real) - gets one JSON line appended here. This is what makes a weekly
review possible: without it, every decision run_scheduled.py makes just
returns a dict and vanishes, which is exactly the gap flagged when this
was first planned out.

JSON Lines (one JSON object per line), not CSV: run_once()'s result
dicts are nested (a "trade" result carries a whole signal dict and a
whole would_send/request dict inside it) and their shape differs
between "skip" and "trade" entries, which doesn't fit a flat CSV schema
cleanly. JSONL handles that with no schema wrangling, and appending a
line is a single atomic write - no read-modify-write of an existing
file, so two runs can never corrupt each other's entries.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

JOURNAL_PATH = Path(__file__).resolve().parents[2] / "data" / "journal.jsonl"


def append_entry(symbol_key: str, timeframe: str, magic: int, result: dict,
                  path: Path = JOURNAL_PATH) -> None:
    """Append one run_once() result to the journal, tagged with when/what/which-account-role it was."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol_key": symbol_key,
        "timeframe": timeframe,
        "magic": magic,
        "result": result,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def read_entries(path: Path = JOURNAL_PATH) -> list[dict]:
    """Read every journal entry, oldest first. Empty list if the journal doesn't exist yet."""
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
