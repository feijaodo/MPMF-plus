"""r105: the embed gate (ingest has right of way) and the still-open-turn fix.

Two false/stalled things happened on 2026-11-17: a 224 s ingest because a background
audit held Ollama in 64-text batches, and a "ingest SILENT" page for a turn that was
still running (27 minutes of tool calls, reply persisted only at the end).
"""
import json
import os
import time

import pytest

from mpmf_plus import embed_gate, health


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MPMF_HOME", str(tmp_path))
    monkeypatch.setenv("MPMF_STORE_LOG", str(tmp_path / "store.log"))
    return tmp_path


def test_marker_exists_only_inside_the_block(home):
    path = embed_gate.marker_path()
    assert not path.exists()
    with embed_gate.ingest_pending():
        assert path.exists(), "the marker must exist while the ingest runs"
        assert embed_gate.ingest_waiting() is True
    assert not path.exists(), "the marker must be gone when the ingest ends"
    assert embed_gate.ingest_waiting() is False


def test_marker_cleared_when_the_ingest_raises(home):
    path = embed_gate.marker_path()
    with pytest.raises(ValueError):
        with embed_gate.ingest_pending():
            raise ValueError("extraction failed")
    assert not path.exists(), "a failed ingest must not park every batch job"


def test_wait_returns_immediately_with_no_marker(home):
    assert embed_gate.wait_for_ingest(max_wait=5) < 0.5


def test_wait_blocks_while_an_ingest_is_waiting(home):
    path = embed_gate.marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": os.getpid(), "ts": time.time()}))
    waited = embed_gate.wait_for_ingest(max_wait=0.4)
    assert waited >= 0.3, "a fresh marker must hold the batch job"


def test_a_stale_marker_is_ignored_and_removed(home):
    path = embed_gate.marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": 999999, "ts": time.time() - embed_gate.STALE_S - 60}))
    assert embed_gate.ingest_waiting() is False
    assert not path.exists(), "a crashed ingest's marker is cleaned up, not honoured"


def test_ingest_sets_the_marker_while_it_runs(home, monkeypatch):
    """The shipped ingest path is what background jobs key on."""
    from mpmf_plus import ingest as ingest_mod

    seen = {}

    def fake_impl(*a, **k):
        seen["waiting"] = embed_gate.ingest_waiting()
        return {"status": "ok"}

    monkeypatch.setattr(ingest_mod, "_ingest_impl", fake_impl)
    out = ingest_mod.ingest("a turn", store=object())
    assert out == {"status": "ok"}
    assert seen["waiting"] is True, "ingest() must hold the gate while it runs"
    assert not embed_gate.marker_path().exists()


# ------------------------------------------------ a turn that is still running

def _silence_case(monkeypatch, turn_ts, turn_end, min_age_min=5.0):
    log = os.environ["MPMF_STORE_LOG"]
    with open(log, "w") as fh:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(turn_ts - 600))
        fh.write("%s ingest_ms=1 facts=1 status=ok\n" % ts)
    monkeypatch.setattr(health, "newest_real_turn_span", lambda db=None: (turn_ts, turn_end))
    monkeypatch.setattr(health, "deploy_ts", lambda: None)
    return health.check_ingest_silence_turns(log=log, min_age_min=min_age_min)


def test_an_open_turn_is_not_silence(home, monkeypatch):
    """Only a user message so far = the turn is still running, however long it takes."""
    now = time.time()
    alerts = _silence_case(monkeypatch, turn_ts=now - 360, turn_end=now - 360)
    assert alerts == [], "a 6-minute open turn must not page anyone"


def test_a_turn_open_for_hours_is_reported(home, monkeypatch):
    now = time.time()
    alerts = _silence_case(monkeypatch, turn_ts=now - 7200, turn_end=now - 7200)
    assert alerts and "still open" in alerts[0]


def test_a_completed_turn_past_the_grace_window_still_alerts(home, monkeypatch):
    """The real bug must still page: a finished turn with no ingest line."""
    now = time.time()
    alerts = _silence_case(monkeypatch, turn_ts=now - 600, turn_end=now - 300, min_age_min=5.0)
    assert alerts and "ingest SILENT" in alerts[0]


def test_a_completed_turn_inside_the_grace_window_is_quiet(home, monkeypatch):
    now = time.time()
    alerts = _silence_case(monkeypatch, turn_ts=now - 300, turn_end=now - 30, min_age_min=5.0)
    assert alerts == []
