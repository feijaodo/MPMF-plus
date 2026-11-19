"""Priority embedding: foreground ingest (and rumination) beat batch jobs.

Ollama serialises requests: one 64-text batch occupies the model for 7-40 s, and a
turn's own embeddings queue behind it. On 2026-11-17 a turn's ingest took 224 s
because a background audit was pushing 64-text batches at the same time — and the
same night the 22:30Z rumination run spent its whole 234 s ceiling in the intake
gate's embedding loop without ever printing its counts.

The protocol is a file marker per priority holder, so it works across processes:

    with ingest_pending():        # the gateway's ingest path
        ...embed the turn...

    with rumination_pending():    # historical_rumination's run (r107)
        ...re-analyse, then run the intake gate...

    wait_for_ingest()             # batch jobs, before each request: sleeps while
                                  # ANY priority holder is live, then goes on

Two holders, one rule: a batch job yields to both. Ingest is the shortest and most
user-visible, so it keeps its own marker file (``ingest_pending.json``) and the
existing name; rumination gets ``rumination_pending.json`` with a longer staleness
window because a run may legitimately last several minutes.

A marker older than its kind's STALE_S is treated as a crash and ignored (and
removed), so a killed ingest or rumination cannot park every batch job forever.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

STALE_S = 300.0          # ingest: a marker this old is a crashed ingest
RUMINATION_STALE_S = 900.0   # rumination: a run may legitimately last ~7 min
POLL_S = 1.0

#: kind -> (filename, staleness window). Order = priority order (for reporting).
MARKERS = {
    "ingest": ("ingest_pending.json", STALE_S),
    "rumination": ("rumination_pending.json", RUMINATION_STALE_S),
}


def marker_path(kind: str = "ingest") -> Path:
    from . import paths      # resolves MPMF_HOME at call time (no import-time path)

    name = MARKERS.get(kind, (kind + "_pending.json", STALE_S))[0]
    return Path(paths.home()) / "health" / name


def _write(path: Path, pid: int, token: str = "") -> None:
    # r112: a private temp file, not a shared "<marker>.tmp". Two holders raced on
    # the same temp path, so one could rename the other's half-written bytes into
    # place. The token identifies WHICH hold wrote the marker.
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump({"pid": pid, "ts": time.time(), "token": token}, fh)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _pid_alive(pid: int) -> bool:
    """Is `pid` a live process? A marker from a SIGKILLed holder is not a hold."""
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # exists, owned by someone else
    except Exception:
        return False
    return True


#: r109: when the CURRENT hold of each kind started, so the holder can report
#: how long the marker has been held (ingest_ms alone cannot prove the marker;
#: store.log now prints marker_ms from here).
_HOLD_START: dict = {}


def held_ms(kind: str = "ingest") -> float:
    """Milliseconds the CURRENT hold of `kind` has been held (0.0 if none)."""
    t0 = _HOLD_START.get(kind)
    return (time.perf_counter() - t0) * 1000.0 if t0 else 0.0


def _holder(kind: str):
    """Mark `kind` as in flight for the lifetime of the with-block."""
    path = marker_path(kind)

    @contextmanager
    def _cm():
        import uuid

        token = uuid.uuid4().hex
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _write(path, os.getpid(), token)
            _HOLD_START[kind] = time.perf_counter()
        except Exception:
            pass                                   # never fail a turn over a marker
        try:
            yield
        finally:
            _HOLD_START.pop(kind, None)
            # r112: delete only OUR marker. Another holder may have taken the file
            # over while we held it; an unconditional unlink let A's exit cancel
            # B's live hold, so B's batch started while A was still embedding.
            try:
                if json.loads(path.read_text()).get("token") == token:
                    path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

    return _cm()


def ingest_pending():
    """Mark the ingest as in flight for the lifetime of the with-block."""
    return _holder("ingest")


def rumination_pending():
    """Mark a rumination run as in flight (r107): batch jobs yield to it too."""
    return _holder("rumination")


def _waiting(kind: str) -> bool:
    path = marker_path(kind)
    try:
        d = json.loads(path.read_text())
        age = time.time() - float(d.get("ts") or 0)
        pid = int(d.get("pid") or 0)
    except Exception:
        return False
    stale = MARKERS.get(kind, ("", STALE_S))[1]
    # r112: a marker whose holder is DEAD is not a hold, whatever its age. Before
    # this, a SIGKILLed holder blocked every batch job for the full window.
    if age > stale or not _pid_alive(pid):
        try:
            if int(json.loads(path.read_text()).get("pid") or 0) == pid:
                path.unlink()
        except Exception:
            pass
        return False
    return True


def ingest_waiting() -> bool:
    """True if a live ingest is waiting for the embedder right now."""
    return _waiting("ingest")


def rumination_waiting() -> bool:
    """True if a live rumination run is using the embedder right now."""
    return _waiting("rumination")


def embed_busy(ignore: Optional[str] = None) -> str:
    """The name of a live priority holder, or "" when a batch job may go.

    `ignore` lets the holder itself poll without waiting on its own marker.
    """
    for kind in MARKERS:
        if kind == ignore:
            continue
        if _waiting(kind):
            return kind
    return ""


def wait_for_ingest(max_wait: float = 180.0) -> float:
    """Block while an ingest OR a rumination run is in flight.

    Returns the seconds actually waited. One entry point on purpose: a batch job
    that asked to yield to the turn must yield to the gate that holds the store
    lock too, or it just adds contention to the run it is blocking.
    """
    t0 = time.time()
    while embed_busy() and (time.time() - t0) < max_wait:
        time.sleep(POLL_S)
    return round(time.time() - t0, 2)
