"""f1 — (1) the watchdog judges only real turns that STARTED after the deploy timestamp;
(2) JSONDecodeError in extraction: one retry, then dead-letter + alert, never silent;
(3) re-ingest of a dead-lettered turn writes facts."""
import json
import os
import sqlite3
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mpmf_plus import health                     # noqa: E402
from mpmf_plus import ingest as ing              # noqa: E402
from mpmf_plus import reingest as rei            # noqa: E402
from test_r83_ingest import FakeMem, fake_vec, reply, DIM   # noqa: E402

NOW = time.time()
MIN = 60.0
Z = lambda t: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))      # noqa: E731


# ---------- 1) watchdog scope -------------------------------------------------------------

def _db(path, msgs):
    c = sqlite3.connect(path)
    c.execute("create table sessions (id text primary key, source text)")
    c.execute("create table messages (id integer primary key, session_id text, role text, "
              "content text, timestamp real)")
    c.execute("insert into sessions values ('s1','telegram')")
    for role, ts in msgs:
        c.execute("insert into messages (session_id, role, content, timestamp) values ('s1',?,?,?)",
                  (role, "x", ts))
    c.commit(); c.close()


def _log(path, ts):
    open(path, "w").write(Z(ts) + " ingest_ms=1 facts=1 skipped=0 status=ok conv=x msg=0\n")


def test_pre_deploy_turn_is_ignored_post_deploy_turn_is_judged(tmp_path):
    deploy = NOW - 60 * MIN
    dfile = tmp_path / "deploy_at"
    dfile.write_text(Z(deploy) + " deploy-test\n")
    assert abs(health.deploy_ts(dfile) - deploy) < 1.5
    log = tmp_path / "store.log"
    _log(log, NOW - 200 * MIN)                      # last ingest long before everything
    os.environ["MPMF_HEALTH_DIR"] = str(tmp_path)
    try:
        # watchdog passes since=<gateway start>, which here is OLDER than the deploy
        since = NOW - 120 * MIN
        # a turn that started BEFORE the deploy and was never ingested: not judged
        _db(tmp_path / "pre.db", [("user", NOW - 90 * MIN), ("assistant", NOW - 89 * MIN)])
        assert health.check_ingest_silence_turns(db=tmp_path / "pre.db", log=log,
                                                 min_age_min=3, since=since) == []
        # control: same turn WITHOUT a deploy file -> it is judged and fails
        os.environ["MPMF_HEALTH_DIR"] = str(tmp_path / "nodeploy")
        got = health.check_ingest_silence_turns(db=tmp_path / "pre.db", log=log,
                                                min_age_min=3, since=since)
        assert got and "ingest SILENT" in got[0]
        os.environ["MPMF_HEALTH_DIR"] = str(tmp_path)
        # a turn that started AFTER the deploy and was not ingested: judged, fails
        _db(tmp_path / "post.db", [("user", NOW - 30 * MIN), ("assistant", NOW - 29 * MIN)])
        got = health.check_ingest_silence_turns(db=tmp_path / "post.db", log=log,
                                                min_age_min=3, since=since)
        assert got and "ingest SILENT" in got[0]
        # a post-deploy turn that WAS ingested: clean
        _log(log, NOW - 28 * MIN)
        assert health.check_ingest_silence_turns(db=tmp_path / "post.db", log=log,
                                                 min_age_min=3, since=since) == []
    finally:
        os.environ.pop("MPMF_HEALTH_DIR", None)


def test_deploy_ts_missing_or_garbled_is_none(tmp_path):
    assert health.deploy_ts(tmp_path / "nope") is None
    (tmp_path / "bad").write_text("garbage\n")
    assert health.deploy_ts(tmp_path / "bad") is None


def test_deploy_sh_writes_the_timestamp_the_watchdog_reads():
    sh = open(os.path.join(os.path.dirname(__file__), "..", "deploy.sh")).read()
    i = sh.index('deploy_at')
    assert sh.index('checkout -q --detach "$TAG"') < i < sh.index('echo "DEPLOYED')
    assert health.paths.log_path("deploy_at").name == "deploy_at"


# ---------- 2) extraction: retry once, then dead-letter -----------------------------------

GOOD = lambda: reply({"s": "The user drives a 2024 Volvo V60.", "entity": "car",         # noqa: E731
                      "ev": "I drive a 2024 Volvo V60."})
BAD = '{"facts": [{"fact_sentence": "x", ]}'                                        # JSONDecodeError
TEXT = "I drive a 2024 Volvo V60."


def _ingest(store, call, tmp_path, **kw):
    return ing.ingest(TEXT, conv_id="convF1", msg_index=3, store=store, call_model=call,
                      embed_fn=fake_vec, embed_dim=DIM, log_path=str(tmp_path / "store.log"),
                      dead_letter_path=str(tmp_path / "dl.jsonl"), **kw)


@pytest.fixture()
def health_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MPMF_HEALTH_DIR", str(tmp_path))
    return tmp_path


def test_json_error_is_retried_once_and_succeeds(health_dir):
    calls = []

    def call(body):
        calls.append(1)
        return BAD if len(calls) == 1 else GOOD()
    res = _ingest(FakeMem(str(health_dir / "a.mpmf")), call, health_dir)
    assert len(calls) == 2 and res["status"] == "ok" and len(res["written"]) == 1
    assert not (health_dir / "dl.jsonl").exists()


def test_second_json_error_dead_letters_alerts_and_raises(health_dir, capsys):
    calls = []

    def call(body):
        calls.append(1)
        return BAD
    with pytest.raises(Exception, match="unparseable"):
        _ingest(FakeMem(str(health_dir / "b.mpmf")), call, health_dir)
    assert len(calls) == 2                                   # exactly one retry
    rows = [json.loads(l) for l in open(health_dir / "dl.jsonl")]
    assert len(rows) == 1
    r = rows[0]
    assert set(r) == {"conv", "msg", "text", "error", "ts"}
    assert r["conv"] == "convF1" and r["msg"] == 3 and r["text"] == TEXT
    assert "JSONDecodeError" in r["error"] or "unparseable" in r["error"]
    assert r["ts"].endswith("Z")
    # the alert: failures.log line + stderr + a daily-check finding
    assert "INGEST DEAD-LETTER" in (health_dir / "failures.log").read_text()
    assert "INGEST DEAD-LETTER" in capsys.readouterr().err
    assert health.check_dead_letter(health_dir / "dl.jsonl")
    assert health.check_failures(log=health_dir / "failures.log")
    assert not (health_dir / "store.log").exists()           # nothing claimed as ingested


def test_non_json_failure_is_not_retried_or_dead_lettered(health_dir):
    calls = []

    def call(body):
        calls.append(1)
        return json.dumps({"nofacts": 1})                    # valid JSON, wrong shape
    with pytest.raises(Exception):
        _ingest(FakeMem(str(health_dir / "c.mpmf")), call, health_dir)
    assert len(calls) == 1 and not (health_dir / "dl.jsonl").exists()


# ---------- 3) re-ingest ------------------------------------------------------------------

def test_reingest_writes_facts_and_clears_the_alert(health_dir):
    store = FakeMem(str(health_dir / "d.mpmf"))
    dl = str(health_dir / "dl.jsonl")
    with pytest.raises(Exception):
        _ingest(store, lambda b: BAD, health_dir)
    assert len(rei.unresolved(dl)) == 1 and health.check_dead_letter(health_dir / "dl.jsonl")
    log = str(health_dir / "store.log")
    out = rei.reingest(store, dl, log_path=log, call_model=lambda b: GOOD(),
                       embed_fn=fake_vec, embed_dim=DIM)
    assert out == [{"conv": "convF1", "msg": 3, "status": "ok", "facts": 1, "skipped": 0}]
    line = open(log).read().strip().splitlines()[-1]
    assert "facts=1" in line and "conv=convF1" in line and "msg=3" in line
    assert rei.unresolved(dl) == [] and health.check_dead_letter(health_dir / "dl.jsonl") == []
    assert os.path.getsize(dl) > 0                           # the dead-letter row is never deleted
    assert rei.reingest(store, dl, log_path=log, call_model=lambda b: GOOD(),
                        embed_fn=fake_vec, embed_dim=DIM) == []      # idempotent


def test_reingest_failure_stays_unresolved(health_dir):
    store = FakeMem(str(health_dir / "e.mpmf"))
    dl = str(health_dir / "dl.jsonl")
    with pytest.raises(Exception):
        _ingest(store, lambda b: BAD, health_dir)
    out = rei.reingest(store, dl, log_path="", call_model=lambda b: BAD, embed_fn=fake_vec,
                       embed_dim=DIM)
    assert out[0]["status"] == "failed" and len(rei.unresolved(dl)) == 1
