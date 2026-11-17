"""Ingest has the right of way over background embedding jobs (r105).

Ollama serialises requests: one 64-text batch occupies the model for 7-40 s, and a
turn's own embeddings queue behind it. On 2026-11-17 a turn's ingest took 224 s
because a background audit was pushing 64-text batches at the same time.

The protocol is a file marker, so it works across processes:

    with ingest_pending():        # the gateway's ingest path
        ...embed the turn...

    wait_for_ingest()             # batch jobs, before each request: sleeps while a
                                  # fresh marker exists, then gets on with it

A marker older than STALE_S is treated as a crash and ignored (and removed), so a
killed ingest cannot park every batch job forever.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

STALE_S = 300.0          # a marker this old is a crashed ingest, not a waiting one
POLL_S = 1.0


def marker_path() -> Path:
    from . import paths      # resolves MPMF_HOME at call time (no import-time path)

    return Path(paths.home()) / "health" / "ingest_pending.json"


def _write(path: Path, pid: int) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"pid": pid, "ts": time.time()}))
    os.replace(tmp, path)


def ingest_pending():
    """Mark the ingest as in flight for the lifetime of the with-block."""
    path = marker_path()

    @contextmanager
    def _cm():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _write(path, os.getpid())
        except Exception:
            pass                                   # never fail a turn over a marker
        try:
            yield
        finally:
            try:
                path.unlink()
            except Exception:
                pass

    return _cm()


def ingest_waiting() -> bool:
    """True if a live ingest is waiting for the embedder right now."""
    path = marker_path()
    try:
        d = json.loads(path.read_text())
        age = time.time() - float(d.get("ts") or 0)
    except Exception:
        return False
    if age > STALE_S:
        try:
            path.unlink()
        except Exception:
            pass
        return False
    return True


def wait_for_ingest(max_wait: float = 180.0) -> float:
    """Block while an ingest is in flight. Returns the seconds actually waited."""
    t0 = time.time()
    while ingest_waiting() and (time.time() - t0) < max_wait:
        time.sleep(POLL_S)
    return round(time.time() - t0, 2)
