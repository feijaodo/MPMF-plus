"""Three dates per memory: when it was said, when we ingested it, when it happened.

Every memory node carries:

  _said_at         when the user actually said it (ISO-8601 UTC, "Z")
  _said_at_source  "explicit" when the caller supplied the real time,
                   "ingest"   when we only know when we wrote it
  _ingested_at     when this store first wrote the node (always present)
  _fact_date       a date stated *inside* the fact text, else "" (unknown)

Why three and not one: before 0.3 the only time a memory had was its ingest
time, so a 2024 conversation imported in 2026 looked newer than a 2026 fact
imported in 2026. That is exactly how the wrong car wins a recency question.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Optional

__all__ = ["normalise_ts", "ingest_now", "extract_fact_date", "date_fields"]

# Dates written in prose. Ordered most-specific first; the first hit wins.
_PATTERNS = (
    re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b"),
    re.compile(r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+20\d{2})\b"),
    re.compile(r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2})\b"),
    re.compile(r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2})\b"),
)

_FORMATS = ("%Y-%m-%d", "%B %d %Y", "%b %d %Y", "%d %B %Y", "%B %Y", "%b %Y")


def normalise_ts(ts) -> Optional[str]:
    """Normalise a timestamp to ISO-8601 UTC with a trailing Z.

    Accepts epoch seconds, epoch milliseconds, ISO strings (with or without
    offset) and the Grok export's ``{"$date": {"$numberLong": "1775164844179"}}``.
    Returns None when the value cannot be interpreted — never a guess.
    """
    if ts is None or ts == "":
        return None
    if isinstance(ts, dict):
        inner = ts.get("$date", ts)
        ts = inner.get("$numberLong", inner) if isinstance(inner, dict) else inner
    if isinstance(ts, datetime):
        return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    s = str(ts).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}T", s):
        return s.replace("+00:00", "Z")
    try:
        f = float(s)
    except (TypeError, ValueError):
        return None
    if f > 1e11:          # milliseconds
        f /= 1000.0
    if f <= 0:
        return None
    try:
        return datetime.fromtimestamp(f, timezone.utc).isoformat().replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return None


def ingest_now() -> str:
    """Now, in the same format as everything else."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def extract_fact_date(text: str) -> Optional[str]:
    """The first date stated in ``text`` as YYYY-MM-DD, else None.

    Only dates the text actually states. A bare year is not a date, and we do
    not resolve "last week" — that needs the saying time, which is separately
    recorded as _said_at.
    """
    if not text:
        return None
    for rx in _PATTERNS:
        m = rx.search(text)
        if not m:
            continue
        raw = m.group(1).replace(",", "").replace(".", "").strip()
        for fmt in _FORMATS:
            try:
                return datetime.strptime(raw, fmt).date().isoformat()
            except ValueError:
                continue
    return None


def date_fields(text: str, said_at=None, conv_id: str = "") -> dict:
    """The wiki fragment every write carries. Single source of truth."""
    ing = ingest_now()
    said = normalise_ts(said_at)
    out = {
        "_ingested_at": ing,
        "_said_at": said or ing,
        "_said_at_source": "explicit" if said else "ingest",
        "_fact_date": extract_fact_date(text) or "",
    }
    if conv_id:
        out["_conv_id"] = str(conv_id)
    return out
