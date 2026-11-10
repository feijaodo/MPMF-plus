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


# r92 step 3: counted turns are REAL user turns only. state.db also holds cron prompts
# (source='cron', id 'cron_*'), subagent sessions and harness runs; those are not the user.
_NOT_REAL_SOURCES = ("cron", "subagent", "harness", "eval", "test")
_CRON_PROMPT = "[IMPORTANT: You are running as a scheduled cron job"


def _state_db(db: Optional[Path] = None) -> Path:
    return Path(db) if db else Path(os.environ.get("HERMES_STATE_DB")
                                    or (Path.home() / ".hermes" / "state.db"))


def _ts(v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) / (1000.0 if v > 1e11 else 1.0)
    import calendar
    t = str(v).replace("Z", "").replace("T", " ")[:19]
    return float(calendar.timegm(time.strptime(t, "%Y-%m-%d %H:%M:%S")))


def newest_real_turn_span(db: Optional[Path] = None) -> Optional[tuple]:
    """(start, end) epoch seconds of the newest REAL user turn, else None.

    start = the user message; end = the last message of that same session at or after
    it (the assistant's final reply / last tool result), i.e. when the turn ENDED. A turn
    still running has an end that keeps moving, so it never looks old. Real = the
    filters of newest_real_turn. Raises on an unreadable db.
    """
    import sqlite3
    con = sqlite3.connect(f"file:{_state_db(db)}?mode=ro", uri=True)
    try:
        marks = ",".join("?" * len(_NOT_REAL_SOURCES))
        row = con.execute(
            "select m.session_id, m.timestamp from messages m join sessions s on s.id = m.session_id "
            f"where m.role = 'user' and coalesce(s.source, '') not in ({marks}) "
            "and s.id not like 'cron\\_%' escape '\\' "
            "and substr(coalesce(m.content, ''), 1, ?) != ? "
            "order by m.timestamp desc limit 1",
            (*_NOT_REAL_SOURCES, len(_CRON_PROMPT), _CRON_PROMPT)).fetchone()
        if not row:
            return None
        end_row = con.execute("select max(timestamp) from messages where session_id = ? and timestamp >= ?",
                              (row[0], row[1])).fetchone()
    finally:
        con.close()
    start = _ts(row[1])
    if start is None:
        return None
    end = _ts(end_row[0]) if end_row else None
    return start, max(start, end if end is not None else start)


def newest_real_turn(db: Optional[Path] = None) -> Optional[float]:
    """Epoch seconds of the START of the newest user message in a REAL user session, else None.

    Excludes sessions whose source is cron/subagent/harness/eval/test, ids starting
    ``cron_``, and any message that is a cron-job prompt. Raises on an unreadable db.
    """
    span = newest_real_turn_span(db)
    return span[0] if span else None


def check_thinking(store: Optional[Path] = None, max_h: float = 24.0,
                   db: Optional[Path] = None, grace_h: float = 2.0) -> List[str]:
    """Alert when the thinking store has not caught up with the user's last REAL turn.

    r92 step 3: the age is measured against the user's newest real turn, not the wall
    clock. If the user has been away (no real turn newer than the last write) the store
    is simply up to date -> silent. If a real turn is newer than the last write by
    more than ``max_h``, and that turn ENDED more than ``grace_h`` ago (capture runs
    every 30 min; e8: the grace is counted from the end of the turn, not its start),
    the capture is broken -> alert.
    """
    p = Path(store) if store else paths.thinking_store()
    try:
        mtime = p.stat().st_mtime
    except Exception:
        return [f"thinking store unreadable: {p}"]
    try:
        span = newest_real_turn_span(db)
    except Exception as e:
        return [f"thinking check: cannot read real turns from {_state_db(db)}: {e!r}"]
    turn, turn_end = span if span else (None, None)
    if turn is None or turn <= mtime:
        return []                          # the user away, or store newer than their last turn
    behind_h = (turn - mtime) / 3600.0
    if behind_h > max_h and (time.time() - turn_end) / 3600.0 > grace_h:
        return [f"thinking store last write {behind_h:.1f} h BEFORE the user's last real turn "
                f"(>{max_h:.0f} h): {p}"]
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
            if "TRACEBACK" in line or "ImportError" in line or "INGEST DEAD-LETTER" in line:
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


def deploy_ts(path: Optional[Path] = None) -> Optional[float]:
    """f1: epoch seconds of the last successful ./deploy.sh (first token of <health>/deploy_at).

    The watchdog judges only turns that STARTED after this instant, so a failure from
    before the deploy cannot fail a post-deploy check. None = no file / unreadable.
    """
    p = Path(path) if path else paths.log_path("deploy_at")
    try:
        tok = p.read_text().split()[0]
        return _ts(tok)
    except Exception:
        return None


def check_ingest_silence_turns(db: Optional[Path] = None, log: Optional[Path] = None,
                               min_age_min: float = 10.0,
                               since: Optional[float] = None) -> List[str]:
    """Alert when a user turn is NEWER than the last store.log line.

    Age-based silence lied: a long quiet night with no turns looked identical to
    a broken ingest. This compares the two clocks instead — the newest user
    message in Hermes' state.db vs the newest ingest line. If the user said something
    after the last ingest, ingestion dropped that turn: that is the bug. No new
    turn since the last ingest = healthy, however old the line is.

    ``min_age_min`` is the grace window, counted from the END of the last real turn
    (its final message in state.db), not from the user message that started it.

    f1: when ``since`` is given (the post-restart watchdog), it is raised to the deploy
    timestamp (``deploy_ts()``) if that is later: a turn that STARTED before the deploy is
    never judged by a post-deploy check, whatever its ingest outcome was.
    """
    if since is not None:
        d = deploy_ts()
        if d is not None and d > since:
            since = d
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
    import calendar
    try:
        ing_ts = float(calendar.timegm(time.strptime(last_ing[:19], "%Y-%m-%dT%H:%M:%S")))
    except ValueError:
        return [f"store.log last line unparseable: {last_ing!r}"]
    try:
        span = newest_real_turn_span(db)    # real user sessions only (r92 step 3)
    except Exception as e:
        return [f"ingest-silence: cannot read {_state_db(db)}: {e!r}"]
    if span is None:
        return []
    turn_ts, turn_end = span
    if turn_ts > ing_ts + 1:
        if since is not None and turn_ts < since:
            return []           # the newest turn predates the window (e.g. a restart)
        age_min = (time.time() - turn_end) / 60.0   # e8: grace runs from the END of the turn
        if age_min < min_age_min:
            return []           # turn just ended / still running: ingest is not due yet
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


def check_dead_letter(path: Optional[Path] = None) -> List[str]:
    """f1: alert while any ingest dead-letter row is unresolved (no re-ingest recorded)."""
    p = Path(path) if path else Path(os.environ.get("MPMF_DEAD_LETTER")
                                     or paths.log_path("ingest_dead_letter.jsonl"))
    def rows(q):
        try:
            return [json.loads(l) for l in open(q, encoding="utf-8") if l.strip()]
        except FileNotFoundError:
            return []
    done = {(r.get("conv"), r.get("msg")) for r in rows(p.with_name(p.stem + ".resolved.jsonl"))}
    open_ = {(r.get("conv"), r.get("msg")) for r in rows(p)} - done
    return [f"{len(open_)} ingest turn(s) dead-lettered and not re-ingested: {p} "
            f"(python -m mpmf_plus.reingest --run)"] if open_ else []


def run_daily() -> List[str]:
    problems: List[str] = []
    for fn in (check_store, check_thinking, check_scorecard, check_failures,
               check_known_stores, check_ingest_silence_turns,
               check_autorevert_fresh, check_dead_letter):
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
                "2x store size; prune old backups and scratch" % (free, threshold_gb))
    return None
