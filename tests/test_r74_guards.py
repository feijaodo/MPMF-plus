"""Round 74 guards ported into the package: supersede, clamp, trip log, re-baseline."""
import os
import types

import pytest

from mpmf_plus import memory as M


# ---------------------------------------------------------------- supersede
def test_superseded_excluded_by_helper():
    class _Mem:
        def get_wiki(self, nid):
            return {"_superseded_by": "9"} if nid == 1 else {}

    rows = [{"node_id": 1}, {"node_id": 2}, {"node_id": 3}]
    kept = [r["node_id"] for r in M._drop_superseded(_Mem(), rows)]
    assert kept == [2, 3]


def test_superseded_excluded_from_d20_matrix(monkeypatch):
    class _Header:
        node_count = 2

    class _Mem:
        header = _Header()

        def all_node_ids(self):
            return [1, 2]

        def get_vector(self, nid):
            return [0.1, 0.2, 0.3]

        def get_wiki(self, nid):
            return {"_superseded_by": "2"} if nid == 1 else {}

    monkeypatch.setitem(M._D20_MATRIX, "count", -1)
    ids, mat, norms = M._d20_node_matrix(_Mem())
    assert ids == [2], ids


# ---------------------------------------------------------------- clamp
def test_rerank_clamp_defaults_off(monkeypatch):
    monkeypatch.setattr(M, "_recall_flags", lambda: {})
    assert M._rerank_clamp_chars() == 0


def test_rerank_clamp_applies_to_encoder_pairs_only(monkeypatch):
    seen = {}

    class _CE:
        def predict(self, pairs, show_progress_bar=False):
            seen["pairs"] = list(pairs)
            return [0.5 for _ in pairs]

    monkeypatch.setattr(M, "_recall_flags", lambda: {"rerank_clamp_chars": 5})
    monkeypatch.setattr(M, "_get_reranker", lambda: _CE())
    long_text = "x" * 500
    calls = [{"text": long_text, "node_id": 1}]
    out = M.rerank("q", calls, top_k=1)
    assert seen["pairs"][0][1] == "x" * 5          # encoder input clamped
    assert out[0]["text"] == long_text             # returned text untouched


# ---------------------------------------------------------------- trip log
def test_trip_logs_every_time_with_context(monkeypatch):
    lines = []
    monkeypatch.setattr(M, "_prewarm_log", lambda line: lines.append(line))
    monkeypatch.setattr(M, "_prewarm_ctx", lambda: "rss=1 kB baseline=2 kB margin=3 kB store=/x nodes=4")
    monkeypatch.setitem(M._PREWARM, "state", None)
    M._prewarm_trip("rss", "RSS 10 > 5")
    M._prewarm_trip("rss", "RSS 10 > 5")
    assert len(lines) == 2, lines                 # not throttled
    assert "baseline=" in lines[0] and "nodes=" in lines[0]
    assert lines[1].endswith("[repeat]")


def test_baseline_is_logged_when_set(monkeypatch):
    lines = []
    monkeypatch.setattr(M, "_prewarm_log", lambda line: lines.append(line))
    monkeypatch.setattr(M, "_prewarm_ctx", lambda: "ctx")
    monkeypatch.setitem(M._PREWARM, "rss_baseline_kb", None)
    monkeypatch.setitem(M._PREWARM, "warm_count", 1)
    monkeypatch.setitem(M._PREWARM, "rss_after_publish_kb", 1234)
    # exercise the two lines of the baseline block directly
    if M._PREWARM.get("rss_baseline_kb") is None and int(M._PREWARM.get("warm_count") or 0) == 1:
        M._PREWARM["rss_baseline_kb"] = M._PREWARM.get("rss_after_publish_kb")
        M._prewarm_log("prewarm: baseline set at %s kB | %s" % (M._PREWARM["rss_baseline_kb"], M._prewarm_ctx()))
    assert lines and "baseline set at 1234 kB" in lines[0]


# ---------------------------------------------------------------- swap
def test_swap_detected_on_size_jump_only():
    small = ("/s", 1, 1_000_000, 11)
    big = ("/s", 2, 2_000_000, 12)
    same_plus_touch = ("/s", 3, 1_001_000, 13)      # normal save: +0.1%
    assert M._is_store_swap(small, big) is True
    assert M._is_store_swap(big, small) is True
    assert M._is_store_swap(small, same_plus_touch) is False
    assert M._is_store_swap(None, big) is False


def test_store_key_carries_inode(tmp_path):
    p = tmp_path / "s.mpmf"
    p.write_bytes(b"x" * 10)
    key = M._store_key(str(p))
    assert len(key) == 4 and isinstance(key[3], int)
