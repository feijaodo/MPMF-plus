"""a1: "which code does THIS process actually run?" stamp.

When ``mpmf_plus.provider`` is imported / the adapter initialises, the process writes
``$MPMF_HOME/health/loaded_rev/<pid>.json``::

    {"pid": 123, "started": "<process start, UTC>", "code_rev": "<CODE_REV>",
     "per_store_scope": true, "ts": "<UTC>"}

``per_store_scope: true`` says this code reads ``append_segments_stores`` (per-store segments) and
ignores the old global ``append_segments`` key. Code from before a1 never writes a stamp, so a
gateway without a stamp is, by definition, old code. The segments guard reads the stamp of the
gateway pid before it lets any store use segments.

Failure-proof: everything is inside try/except, the write is atomic (temp file + rename), and a
failure (read-only dir, full disk, ...) is swallowed. Stamps whose pid is dead (or whose pid was
reused by a different process: start time differs) are removed on every write.
Env: ``MPMF_LOADED_REV_DIR`` overrides the directory (tests, guard proofs); the stamp is skipped
under pytest unless that variable is set, so a test run never litters the real health dir.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Optional

PER_STORE_SCOPE = True


def stamp_dir() -> str:
    d = os.environ.get("MPMF_LOADED_REV_DIR")
    if d:
        return d
    from . import paths as _paths
    return str(_paths.health_dir() / "loaded_rev")


def proc_start(pid: int) -> Optional[float]:
    """Process start as epoch seconds (from /proc), or None when the pid does not exist."""
    try:
        with open("/proc/%d/stat" % pid) as fh:
            raw = fh.read()
        fields = raw[raw.rindex(")") + 2:].split()      # after "pid (comm) "
        ticks = int(fields[19])                          # starttime, field 22 overall
        with open("/proc/stat") as fh:
            btime = next(int(l.split()[1]) for l in fh if l.startswith("btime"))
        return btime + ticks / os.sysconf("SC_CLK_TCK")
    except Exception:
        return None


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def _prune(d: str, keep_pid: int) -> None:
    for name in os.listdir(d):
        if not name.endswith(".json"):
            continue
        try:
            pid = int(name[:-5])
        except ValueError:
            continue
        if pid == keep_pid:
            continue
        p = os.path.join(d, name)
        try:
            started = proc_start(pid)
            if started is None:
                os.unlink(p)                             # dead pid
                continue
            with open(p) as fh:
                rec = json.load(fh)
            if abs(parse_iso(rec["started"]) - started) > 2.5:
                os.unlink(p)                             # pid reused by another process
        except Exception:
            try:
                if proc_start(pid) is None:
                    os.unlink(p)
            except Exception:
                pass


def write_stamp(code_rev: Optional[str] = None) -> Optional[str]:
    """Write this process's stamp. Returns its path, or None when skipped / failed. Never raises."""
    try:
        if "pytest" in sys.modules and not os.environ.get("MPMF_LOADED_REV_DIR"):
            return None
        if code_rev is None:
            from ._rev import CODE_REV as code_rev
        d = stamp_dir()
        os.makedirs(d, exist_ok=True)
        pid = os.getpid()
        st = proc_start(pid)
        rec = {"pid": pid, "started": iso(st) if st is not None else None, "code_rev": code_rev,
               "per_store_scope": PER_STORE_SCOPE, "ts": iso(time.time())}
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".stamp_")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(rec, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, os.path.join(d, "%d.json" % pid))
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            return None
        try:
            _prune(d, pid)
        except Exception:
            pass
        return os.path.join(d, "%d.json" % pid)
    except Exception:
        return None
