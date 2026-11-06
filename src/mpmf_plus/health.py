"""Health checks for MPMF+ stores, logs and schedules.

Moved out of the host daily-check scripts.
Every path comes from .paths (MPMF_HOME / HOME); no host literal.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Optional

from . import paths


def _age_h(path: Path) -> Optional[float]:
    try:
        return (time.time() - path.stat().st_mtime) / 3600.0
    except Exception:
        return None


def check_thinking(store: Optional[Path] = None, max_h: float = 24.0) -> List[str]:
    p = Path(store) if store else paths.thinking_store()
    age = _age_h(p)
    if age is None:
        return [f"thinking store unreadable: {p}"]
    if age > max_h:
        return [f"thinking store last write {age:.1f} h ago (>{max_h:.0f} h): {p}"]
    return []


def check_scorecard(history: Optional[Path] = None, max_h: float = 26.0) -> List[str]:
    _ev = os.environ.get("MPMF_EVAL_DIR")
    p = Path(history) if history else Path(
        os.environ.get("MPMF_SCORECARD_HISTORY")
        or (Path(_ev) / "scorecard_history.jsonl" if _ev
            else paths.home() / "eval" / "scorecard_history.jsonl"))
    try:
        last = None
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line).get("ts") or last
            except Exception:
                continue
        if not last:
            return [f"nightly scorecard has never written a line: {p}"]
        ts = time.mktime(time.strptime(last, "%Y-%m-%dT%H:%M:%SZ"))
        age = (time.time() - ts) / 3600.0
        if age > max_h:
            return [f"nightly scorecard last line {age:.1f} h ago (>{max_h:.0f} h): {last}"]
        return []
    except FileNotFoundError:
        return [f"nightly scorecard history missing: {p}"]


def check_ingest_silence(max_h: float = 12.0, log: Optional[Path] = None) -> List[str]:
    """Alert when no real turn has reached store.log for max_h hours."""
    p = Path(log) if log else Path(os.environ.get("MPMF_STORE_LOG")
                                   or (paths.home() / "health" / "store.log"))
    try:
        last = None
        for line in open(p):
            line = line.strip()
            if line:
                last = line.split(" ", 1)[0]
        if not last:
            return [f"store.log has no ingest lines: {p}"]
        ts = time.mktime(time.strptime(last[:19], "%Y-%m-%dT%H:%M:%S"))
        age = (time.time() - ts) / 3600.0
        if age > max_h:
            return [f"no ingest for {age:.1f} h (>{max_h:.0f} h): {last} {p}"]
        return []
    except FileNotFoundError:
        return [f"store.log missing: {p}"]


def check_store(store: Optional[Path] = None) -> List[str]:
    p = Path(store) if store else paths.store_path()
    if not p.exists():
        return [f"store missing: {p}"]
    if p.stat().st_size < 1024:
        return [f"store too small ({p.stat().st_size} B): {p}"]
    return []


def check_failures(max_h: float = 24.0, log: Optional[Path] = None) -> List[str]:
    p = Path(log) if log else paths.log_path("failures.log")
    try:
        cut = time.time() - max_h * 3600
        n = 0
        for line in open(p):
            if "TRACEBACK" in line or "ImportError" in line:
                try:
                    ts = time.mktime(time.strptime(line.split(" ", 1)[0][:19], "%Y-%m-%dT%H:%M:%S"))
                except Exception:
                    continue
                if ts >= cut:
                    n += 1
        return [f"{n} import/traceback failures in {max_h:.0f} h: {p}"] if n else []
    except FileNotFoundError:
        return []


def run_daily() -> List[str]:
    problems: List[str] = []
    for fn in (check_store, check_thinking, check_scorecard, check_failures):
        try:
            problems.extend(fn())
        except Exception as e:
            problems.append(f"{fn.__name__} crashed: {e!r}")
    return problems


def check_disk(threshold_gb: float = 3.0, path: Optional[str] = None) -> Optional[str]:
    """Return an alert string when free space is under threshold_gb."""
    import shutil
    from pathlib import Path as _P
    p = path or str(paths.home())
    try:
        free = shutil.disk_usage(p).free / 1e9
    except OSError as e:
        return "disk check failed for %s: %s" % (p, e)
    if free < threshold_gb:
        return ("disk low: %.1f GB free (threshold %.1f GB) — store saves need "
                "2x store size; prune /root/backups and round scratch" % (free, threshold_gb))
    return None
