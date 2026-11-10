#!/usr/bin/env python3
"""e8: the watchdog grace window is counted from the END of the last real user turn,
and the alert says "verify before rolling back".

Throwaway state.db / store.log in a temp dir. The host watchdog script is loaded from
$HOME/.hermes/scripts (skipped when absent) only for its alert wording.
"""
import importlib.util, os, sqlite3, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mpmf_plus import health

NOW = time.time()
MIN = 60.0
GRACE_MIN = 3.0          # watchdog GRACE_S default 180 s


def mkdb(path, msgs):
    c = sqlite3.connect(path)
    c.execute("create table sessions (id text primary key, source text)")
    c.execute("create table messages (id integer primary key, session_id text, role text, content text, timestamp real)")
    c.execute("insert into sessions values ('s1','telegram')")
    for role, ts in msgs:
        c.execute("insert into messages (session_id, role, content, timestamp) values ('s1',?,?,?)", (role, "x", ts))
    c.commit(); c.close()


def mklog(path, ts):
    open(path, "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)) + " ingest_ms=1 facts=1 skipped=0 status=ok conv=x msg=0\n")


def test_grace_counts_from_end_of_turn():
    d = Path(tempfile.mkdtemp(prefix="mpmf-e8-"))
    log = d / "store.log"
    mklog(log, NOW - 60 * MIN)
    chk = lambda db: health.check_ingest_silence_turns(db=db, log=log, min_age_min=GRACE_MIN)
    # user message 30 min ago (start is OLD), reply finished 1 min ago (end is INSIDE grace) -> silent
    mkdb(d / "just_ended.db", [("user", NOW - 30 * MIN), ("assistant", NOW - 1 * MIN)])
    assert chk(d / "just_ended.db") == []
    # same start, turn ended 10 min ago (outside grace) -> fires
    mkdb(d / "old.db", [("user", NOW - 30 * MIN), ("assistant", NOW - 10 * MIN)])
    got = chk(d / "old.db")
    assert got and "ingest SILENT" in got[0], got
    # thinking check: grace_h counted from the end too
    store = d / "t.mpmf"; store.touch(); os.utime(store, (NOW - 40 * 3600,) * 2)
    mkdb(d / "t_just.db", [("user", NOW - 5 * 3600), ("assistant", NOW - 0.5 * 3600)])
    assert health.check_thinking(store=store, db=d / "t_just.db") == []
    mkdb(d / "t_old.db", [("user", NOW - 5 * 3600), ("assistant", NOW - 3 * 3600)])
    assert health.check_thinking(store=store, db=d / "t_old.db")


def test_alert_wording():
    p = Path(os.environ.get("MPMF_WATCHDOG_SCRIPT")
             or Path.home() / ".hermes/scripts/mpmf_post_restart_watchdog.py")
    if not p.is_file():
        import pytest; pytest.skip("host watchdog script not present")
    spec = importlib.util.spec_from_file_location("wd", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    text = m.format_alert(123, "Mon Nov  9 10:00:00 2026", ["ingest: x"])
    assert "verify before rolling back" in text
    assert text.index("verify before rolling back") < text.index("git -C")


if __name__ == "__main__":
    test_grace_counts_from_end_of_turn(); test_alert_wording(); print("OK test_watchdog_grace")
