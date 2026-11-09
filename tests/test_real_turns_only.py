#!/usr/bin/env python3
"""r92 step 3: silence checks count only REAL user turns.

  * silent when only cron/subagent sessions are newer than the last ingest;
  * fires on a simulated real missed turn;
  * thinking-store age is relative to the user's last real turn (silent when they are away,
    fires when a real turn is far newer than the last capture).
Throwaway state.db, store.log and store file; nothing outside the temp dir is read.
"""
import os, sqlite3, sys, tempfile, time, calendar
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mpmf_plus import health

CRON_PROMPT = "[IMPORTANT: You are running as a scheduled cron job. DELIVERY: ..."
NOW = time.time()
H = 3600.0

def mkdb(path, msgs):
    c = sqlite3.connect(path)
    c.execute("create table sessions (id text primary key, source text)")
    c.execute("create table messages (id integer primary key, session_id text, role text, content text, timestamp real)")
    seen = set()
    for sid, src, role, content, ts in msgs:
        if sid not in seen:
            c.execute("insert into sessions values (?,?)", (sid, src)); seen.add(sid)
        c.execute("insert into messages (session_id, role, content, timestamp) values (?,?,?,?)", (sid, role, content, ts))
    c.commit(); c.close()

def mklog(path, ts):
    open(path, "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)) + " ingest_ms=1 facts=1 skipped=0 status=ok conv=x msg=0\n")

def setmtime(path, ts):
    Path(path).touch(); os.utime(path, (ts, ts))

def main():
    d = Path(tempfile.mkdtemp(prefix="mpmf-rt-"))
    db, log, store = d / "state.db", d / "store.log", d / "thinking.mpmf"
    fails = []
    def expect(name, got, want_alert):
        ok = bool(got) == want_alert
        print("[%s] %s -> %s" % ("ok" if ok else "FAIL", name, got[0][:110] if got else "silent"))
        if not ok: fails.append(name)

    # A. last real turn 30 h ago and ingested; cron/subagent/harness prompts since -> silent
    real_ts = NOW - 30 * H
    mklog(log, real_ts + 5)
    mkdb(db, [("20261108_x", "telegram", "user", "hello", real_ts),
              ("cron_6490_1", "cron", "user", CRON_PROMPT, NOW - 2 * H),
              ("cron_d897_1", "cron", "user", CRON_PROMPT, NOW - 1 * H),
              ("sub_1", "subagent", "user", "do a thing", NOW - 3 * H),
              ("odd_1", "telegram", "user", CRON_PROMPT + " (leaked)", NOW - 4 * H)])
    expect("A ingest-silence, only cron/subagent newer", health.check_ingest_silence_turns(db=db, log=log, min_age_min=0), False)
    # control: the OLD query (all user messages) would have fired
    con = sqlite3.connect(db); old_max = con.execute("select max(timestamp) from messages where role='user'").fetchone()[0]; con.close()
    if not old_max > real_ts + 1: fails.append("control: old rule should see a newer turn")
    print("[ok] control: old rule max(all user ts) is %.1f h newer than last ingest" % ((old_max - real_ts) / H))

    # B. a real missed turn 3 h ago, log last written before it -> fires
    mkdb(d / "b.db", [("20261109_y", "telegram", "user", "remember this", NOW - 3 * H),
                      ("cron_1", "cron", "user", CRON_PROMPT, NOW - 1 * H)])
    mklog(log, NOW - 5 * H)
    expect("B ingest-silence, real missed turn", health.check_ingest_silence_turns(db=d / "b.db", log=log, min_age_min=0), True)
    # B2. the same real turn but ingested -> silent
    mklog(log, NOW - 3 * H + 10)
    expect("B2 ingest-silence, real turn ingested", health.check_ingest_silence_turns(db=d / "b.db", log=log, min_age_min=0), False)

    # C. thinking store 36 h old but the user away (last real turn older than the store) -> silent
    setmtime(store, NOW - 36 * H)
    mkdb(d / "c.db", [("20261107_z", "telegram", "user", "bye", NOW - 40 * H),
                      ("cron_1", "cron", "user", CRON_PROMPT, NOW - 1 * H)])
    expect("C thinking, the user away 40 h, store 36 h old", health.check_thinking(store=store, db=d / "c.db"), False)
    # D. real turn 1 h ago, store last written 36 h ago (35 h behind) -> grace: turn too fresh -> silent
    mkdb(d / "d.db", [("20261109_q", "telegram", "user", "now", NOW - 1 * H)])
    expect("D thinking, real turn only 1 h ago (grace)", health.check_thinking(store=store, db=d / "d.db"), False)
    # E. real turn 5 h ago, store last written 36 h ago -> capture broken -> fires
    mkdb(d / "e.db", [("20261109_w", "telegram", "user", "now", NOW - 5 * H)])
    expect("E thinking, real turn 5 h ago, store 31 h behind it", health.check_thinking(store=store, db=d / "e.db"), True)
    # F. missing store still reported
    expect("F thinking, store missing", health.check_thinking(store=d / "nope.mpmf", db=d / "e.db"), True)
    # G. unreadable db is an alert, not a silent pass
    expect("G ingest-silence, unreadable db", health.check_ingest_silence_turns(db=d / "missing.db", log=log, min_age_min=0), True)
    print("FAIL: %s" % fails if fails else "OK test_real_turns_only")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
