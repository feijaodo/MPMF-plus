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


# ── r91b: known stores ────────────────────────────────────────────────────
# Every .mpmf file in $MPMF_HOME belongs to a named agent. A file that is NOT
# on this list is either a new agent (add it here on purpose) or a stray created
# by a process that guessed a path — and guessing is what r91a's store guard now
# forbids. Audits flag only the ones off the list; the listed files are never
# reported as suspicious.
KNOWN_STORES = {
    "athena.mpmf": "athena (facts / recall)",
    "athena-thinking.mpmf": "athena (thinking traces)",
    "espy.mpmf": "espy",
    "shared.mpmf": "shared — Espy Property Scout bank (written via MPMF_STORE)",
    "ayu.mpmf": "ayu",
    "pre.mpmf": "pre",
    "id.mpmf": "id",
}


def check_known_stores(home: Optional[Path] = None) -> List[str]:
    """Alert on .mpmf files that are not on KNOWN_STORES (strays)."""
    base = Path(home) if home else paths.home()
    try:
        files = sorted(p.name for p in base.glob("*.mpmf") if p.is_file())
    except OSError as e:
        return [f"known-stores scan failed for {base}: {e}"]
    return [f"unknown store: {base / f} ({base.joinpath(f).stat().st_size} B) "
            f"— not in KNOWN_STORES; add it or archive it"
            for f in files if f not in KNOWN_STORES]


def check_ingest_silence_turns(db: Optional[Path] = None, log: Optional[Path] = None,
                               min_age_min: float = 10.0) -> List[str]:
    """Alert when a user turn is NEWER than the last store.log line.

    Age-based silence lied: a long quiet night with no turns looked identical to
    a broken ingest. This compares the two clocks instead — the newest user
    message in Hermes' state.db vs the newest ingest line. If the user said something
    after the last ingest, ingestion dropped that turn: that is the bug. No new
    turn since the last ingest = healthy, however old the line is.
    """
    import sqlite3
    dbp = Path(db) if db else Path(os.environ.get("HERMES_STATE_DB")
                                   or (Path.home() / ".hermes" / "state.db"))
    lp = Path(log) if log else Path(os.environ.get("MPMF_STORE_LOG")
                                    or (paths.home() / "health" / "store.log"))
    try:
        last_ing = None
        for line in open(lp):
            line = line.strip()
            if line:
                last_ing = line.split(" ", 1)[0]
    except FileNotFoundError:
        return [f"store.log missing: {lp}"]
    if not last_ing:
        return [f"store.log has no ingest lines: {lp}"]
    try:
        ing_ts = time.mktime(time.strptime(last_ing[:19], "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return [f"store.log last line unparseable: {last_ing!r}"]
    try:
        con = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True)
        try:
            row = con.execute(
                "select max(timestamp) from messages where role='user'").fetchone()
        finally:
            con.close()
    except Exception as e:
        return [f"ingest-silence: cannot read {dbp}: {e!r}"]
    newest = row[0] if row else None
    if not newest:
        return []
    if isinstance(newest, (int, float)):
        turn_ts = float(newest) / (1000.0 if newest > 1e11 else 1.0)
    else:
        s = str(newest).replace("Z", "").replace("T", " ")[:19]
        try:
            turn_ts = time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            return [f"ingest-silence: unparseable message timestamp {newest!r}"]
    if turn_ts > ing_ts + 1:
        age_min = (time.time() - turn_ts) / 60.0
        if age_min < min_age_min:
            return []           # turn is still being ingested right now
        return [f"ingest SILENT: user turn at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(turn_ts))} "
                f"is newer than last store.log line {last_ing} "
                f"({(turn_ts - ing_ts) / 60.0:.1f} min behind) — {lp}"]
    return []


def check_autorevert_fresh(max_h: float = 26.0) -> List[str]:
    """Alert when the nightly's auto-revert verdicts are stale (r91c).

    autorevert_last.txt is written by every real nightly run, fired or not. If it
    is older than 26 h, last night's scorecard or its two revert checks did not
    run — which is exactly how the rc=2 argparse bug hid for a week.
    """
    p = paths.home() / "eval" / "autorevert_last.txt"
    try:
        age = (time.time() - p.stat().st_mtime) / 3600.0
    except FileNotFoundError:
        return [f"autorevert_last.txt missing: {p} (nightly has never logged a "
                f"reranker/history verdict)"]
    except OSError as e:
        return [f"autorevert_last.txt unreadable: {p}: {e}"]
    if age > max_h:
        return [f"autorevert_last.txt {age:.1f} h old (>{max_h:.0f} h): {p} — "
                f"last night's scorecard / auto-revert checks did not run"]
    return []


def run_daily() -> List[str]:
    problems: List[str] = []
    for fn in (check_store, check_thinking, check_scorecard, check_failures,
               check_known_stores, check_ingest_silence_turns,
               check_autorevert_fresh):
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
