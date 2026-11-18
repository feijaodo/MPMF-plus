"""r107: TWO priority holders, one rule — batch jobs yield to both.

The embed gate used to know only about ingest. Rumination (historical_rumination)
spends minutes in the intake gate's embedding loop and wrote no marker at all, so
an audit or re-embed sweep could sit on the embedder while the run burned its cron
ceiling — the 22:30Z run on 2026-11-17 is exactly that failure.

rumination_pending() is the second holder. wait_for_ingest() (the ONE entry point
every batch job uses) now returns only when no priority holder is live, so the
holders added here protect every existing caller without touching them.
"""
from __future__ import annotations

import json
import os
import time

import pytest

from mpmf_plus import embed_gate


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MPMF_HOME", str(tmp_path))
    return tmp_path


def test_rumination_has_its_own_marker(home):
    ipath = embed_gate.marker_path("ingest")
    rpath = embed_gate.marker_path("rumination")
    assert ipath.name == "ingest_pending.json"
    assert rpath.name == "rumination_pending.json"
    assert ipath != rpath


def test_rumination_marker_exists_only_inside_the_block(home):
    path = embed_gate.marker_path("rumination")
    assert not path.exists()
    with embed_gate.rumination_pending():
        assert path.exists()
        assert embed_gate.rumination_waiting() is True
        assert embed_gate.ingest_waiting() is False, "rumination must not look like an ingest"
    assert not path.exists()
    assert embed_gate.rumination_waiting() is False


def test_batch_jobs_yield_to_rumination(home):
    with embed_gate.rumination_pending():
        waited = embed_gate.wait_for_ingest(max_wait=0.4)
        assert waited >= 0.3, "a live rumination run must hold a batch job"


def test_batch_jobs_yield_to_ingest_still(home):
    with embed_gate.ingest_pending():
        waited = embed_gate.wait_for_ingest(max_wait=0.4)
        assert waited >= 0.3


def test_no_marker_means_go(home):
    assert embed_gate.wait_for_ingest(max_wait=5) < 0.5
    assert embed_gate.embed_busy() == ""


def test_embed_busy_names_the_holder(home):
    with embed_gate.ingest_pending():
        assert embed_gate.embed_busy() == "ingest"
        assert embed_gate.embed_busy(ignore="ingest") == ""
    with embed_gate.rumination_pending():
        assert embed_gate.embed_busy() == "rumination"
        assert embed_gate.embed_busy(ignore="rumination") == ""


def test_a_crashed_rumination_marker_is_ignored_and_removed(home):
    path = embed_gate.marker_path("rumination")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": 999999,
                                "ts": time.time() - embed_gate.RUMINATION_STALE_S - 60}))
    assert embed_gate.rumination_waiting() is False
    assert not path.exists(), "a crashed run's marker is cleaned up, not honoured"


def test_a_live_rumination_marker_outlives_the_ingest_window(home):
    """A rumination run may legitimately last longer than an ingest marker may."""
    path = embed_gate.marker_path("rumination")
    path.parent.mkdir(parents=True, exist_ok=True)
    age = embed_gate.STALE_S + 60          # 6 min: stale for ingest, fine for rumination
    path.write_text(json.dumps({"pid": os.getpid(), "ts": time.time() - age}))
    assert embed_gate.RUMINATION_STALE_S > embed_gate.STALE_S
    assert embed_gate.rumination_waiting() is True


def test_markers_clear_when_the_holder_raises(home):
    with pytest.raises(ValueError):
        with embed_gate.rumination_pending():
            raise ValueError("run failed")
    assert not embed_gate.marker_path("rumination").exists()


def test_rumination_run_holds_the_marker(monkeypatch, home):
    """historical_rumination.main() must hold it for the whole run (r107)."""
    import importlib.util
    path = os.path.expanduser("~/.hermes/scripts/historical_rumination.py")
    if not os.path.exists(path):
        pytest.skip("host rumination script not present")
    spec = importlib.util.spec_from_file_location("hist_rum_r107", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    seen = {}

    def fake_locked():
        seen["waiting"] = embed_gate.rumination_waiting()
        seen["ingest_waiting"] = embed_gate.ingest_waiting()
        return 0

    monkeypatch.setattr(mod, "_main_locked", fake_locked)
    monkeypatch.setattr(mod, "_is_dry", lambda: False)
    rc = mod.main()
    assert rc == 0
    assert seen == {"waiting": True, "ingest_waiting": False}
    assert not embed_gate.marker_path("rumination").exists()
