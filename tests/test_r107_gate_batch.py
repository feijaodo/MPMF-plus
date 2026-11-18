"""r107: the intake gate embeds in BATCHES, and never re-embeds an unchanged lesson.

Before this, `_run_gate` called `_vec(text)` once per staged lesson: ~343
sequential single-text Ollama round-trips per run. At 4-6 s per request under
load that is 234 s — the whole cron ceiling — which is exactly what the 22:30Z
historical-rumination run hit before it could print its counts.

The fix has two halves and both are asserted here:
  * misses go out in chunks of GATE_EMBED_BATCH (32), not one request each
  * a lesson whose embedded string did not change is served from the persistent
    cache, so held lessons cost nothing on the next run

Everything is patched on ``cg.A`` (the module the gate actually calls — it is
``mpmf_plus.provider``), not on ``mpmf_plus.memory``: a test that patches the
twin proves nothing about the code that runs.
"""
from __future__ import annotations

import json
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mpmf_plus import cognition as cg  # noqa: E402
from mpmf_plus import memory as M  # noqa: E402

A = cg.A                      # mpmf_plus.provider — the gate's embedder
DIM = 1024


@pytest.fixture()
def batch(monkeypatch, tmp_path):
    """Patch the batch embedder; record every chunk size; give a private cache."""
    calls = {"chunks": [], "single": 0}

    def fake_batch(texts, dim=DIM, batch=32):
        calls["chunks"].append(len(texts))
        return [[float((len(t) % 7) + 1)] * dim for t in texts]

    def fake_single(text, dim=DIM, cap=None):
        calls["single"] += 1
        return [float((len(text) % 7) + 1)] * dim

    monkeypatch.setattr(A, "_embed_floats_batch", fake_batch)
    monkeypatch.setattr(A, "_embed_floats_local", fake_single)
    monkeypatch.setattr(A, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(A, "EMBED_MODEL", "fake-embed")
    return calls


def _text(i):
    return "lesson %d | trigger %d | question %d" % (i, i, i)


def test_the_real_batch_embedder_splits_into_chunks_of_32(monkeypatch):
    """The chunking lives in A._embed_floats_batch — assert it on the real code."""
    sizes = []

    class _Resp:
        status_code = 200

        def __init__(self, n):
            self._n = n

        def json(self):
            return {"embeddings": [[1.0] * DIM for _ in range(self._n)]}

    def fake_post(url, headers=None, json=None, timeout=None):
        sizes.append(len(json["input"]))
        return _Resp(len(json["input"]))

    monkeypatch.setattr(A, "requests", type("R", (), {"post": staticmethod(fake_post)}))
    out = A._embed_floats_batch([_text(i) for i in range(70)], batch=32)
    assert sizes == [32, 32, 6]
    assert len(out) == 70 and all(len(v) == DIM for v in out)


def test_vecs_sends_all_misses_in_one_call(batch):
    cg._vecs([_text(i) for i in range(70)])
    assert batch["chunks"] == [70], "chunking is the batch embedder's job, not the gate's"
    assert batch["single"] == 0


def test_a_second_run_reuses_every_vector(batch):
    cg._vecs([_text(i) for i in range(10)])
    assert batch["chunks"] == [10]
    assert cg.LAST_EMBED_STATS == {"new": 10, "reused": 0, "batches": 1}

    cg._vecs([_text(i) for i in range(10)])
    assert batch["chunks"] == [10], "nothing changed, so nothing may be embedded"
    assert cg.LAST_EMBED_STATS == {"new": 0, "reused": 10, "batches": 0}


def test_only_the_changed_lesson_is_re_embedded(batch):
    cg._vecs([_text(i) for i in range(5)])
    batch["chunks"].clear()
    cg._vecs([_text(i) for i in range(4)] + ["a brand new lesson | t | q"])
    assert batch["chunks"] == [1]
    assert cg.LAST_EMBED_STATS == {"new": 1, "reused": 4, "batches": 1}


def test_cache_prunes_lessons_that_are_no_longer_staged(batch, tmp_path):
    cg._vecs([_text(i) for i in range(5)])
    path = tmp_path / "cache" / cg.GATE_VECS_NAME
    assert len(json.loads(path.read_text())["vecs"]) == 5
    cg._vecs([_text(0)])
    assert len(json.loads(path.read_text())["vecs"]) == 1


def test_dry_run_writes_nothing(batch, tmp_path):
    path = tmp_path / "cache" / cg.GATE_VECS_NAME
    assert not path.exists()
    cg._vecs([_text(i) for i in range(3)], dry_run=True)
    assert not path.exists(), "a dry run must not create the cache"


def test_a_failed_batch_falls_back_per_text(monkeypatch, tmp_path):
    """One unembeddable lesson is held; the rest must still get vectors."""
    seen = []

    def bad_batch(texts, dim=DIM, batch=32):
        raise RuntimeError("/api/embed 400")

    def one(text, dim=DIM, cap=None):
        seen.append(text)
        if "poison" in text:
            raise RuntimeError("cannot embed this lesson")
        return [1.0] * dim

    monkeypatch.setattr(A, "_embed_floats_batch", bad_batch)
    monkeypatch.setattr(A, "_embed_floats_local", one)
    monkeypatch.setattr(A, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(A, "EMBED_MODEL", "fake-embed")

    out = cg._vecs([_text(1), "poison | t | q", _text(2)])
    assert [v is None for v in out] == [False, True, False]
    assert len(seen) == 3


def test_a_model_change_invalidates_the_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "CACHE_DIR", tmp_path / "cache")
    calls = []

    def fake(texts, dim=DIM, batch=32):
        calls.append(len(texts))
        return [[1.0] * dim for _ in texts]

    monkeypatch.setattr(A, "_embed_floats_batch", fake)
    monkeypatch.setattr(A, "EMBED_MODEL", "model-a")
    cg._vecs([_text(0)])
    assert calls == [1]
    monkeypatch.setattr(A, "EMBED_MODEL", "model-b")
    cg._vecs([_text(0)])
    assert calls == [1, 1], "a different embedder must not reuse model-a's vectors"


def test_the_gate_runs_the_batched_path():
    """_run_gate must call _vecs (batched + cached), never _vec in a loop."""
    import inspect
    src = inspect.getsource(cg._run_gate)
    assert "_vecs(" in src, "the gate must embed through the batched/cached helper"
    assert "v = _vec(text)" not in src, "the per-lesson single round-trip must be gone"
    assert '"embedded"' in src and '"vectors_reused"' in src
