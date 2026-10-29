"""
mpmf/ingestion/decay.py — Temporal decay & effective-weight (spec §"Temporal Decay").

Lazy decay: effective weight is computed at read time from base_weight and
last_reinforced_at, never swept eagerly. Metadata is stored in node wiki under
reserved keys so it round-trips through the existing storage layer:

    _qid           originating question_id (retrieval attribution)
    _base_weight   original confidence/importance (0..1)
    _reinforced    last reinforced timestamp (unix seconds)
"""

import math
import time

DECAY_FACTOR = 0.95          # per decay_period
DECAY_PERIOD = 7 * 86400     # 7 days in seconds
REINFORCE_BUMP = 0.05

KEY_QID = "_qid"
KEY_BASE = "_base_weight"
KEY_REINFORCED = "_reinforced"


def effective_weight(wiki: dict, now: float = None,
                     decay_factor: float = DECAY_FACTOR,
                     decay_period: float = DECAY_PERIOD) -> float:
    """Compute a node's current effective weight from its stored metadata."""
    now = now if now is not None else time.time()
    try:
        base = float(wiki.get(KEY_BASE, 0.5))
    except (TypeError, ValueError):
        base = 0.5
    try:
        reinforced = float(wiki.get(KEY_REINFORCED, now))
    except (TypeError, ValueError):
        reinforced = now
    age = max(0.0, now - reinforced)
    return base * (decay_factor ** (age / decay_period))


def reinforce(wiki: dict, now: float = None, bump: float = REINFORCE_BUMP) -> dict:
    """Return wiki updates that reinforce a node (retrieved-and-used)."""
    now = now if now is not None else time.time()
    try:
        base = float(wiki.get(KEY_BASE, 0.5))
    except (TypeError, ValueError):
        base = 0.5
    return {KEY_BASE: f"{min(1.0, base + bump):.3f}", KEY_REINFORCED: str(int(now))}
