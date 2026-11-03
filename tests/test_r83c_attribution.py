"""r83c: facts attributed to the CONTEXT messages must not be written.

The 67e window is [prev-2, prev-1, current]. A fact whose msg_index points at
prev-1 belongs to that earlier turn's ingest; writing it here gives it this
turn's _said_at/_conv_id (the r83b judge found 3 such facts in 20).
"""
import json

import pytest

from mpmf_plus import ingest as ing


class FakeStore:
    """Just enough MPMFMemory for ingest: a txn, a create_node and no dedup hits."""

    def __init__(self):
        self.created = []

    def write_txn(self, *a, **k):
        class T:
            def __enter__(s):
                return None

            def __exit__(s, *exc):
                return False
        return T()

    def create_node(self, **kw):
        self.created.append(kw)
        return len(self.created)


class FakeCore:
    def __init__(self):
        self.nodes = []
        self._in_txn = True   # stamp_embedder checks this

    def write_txn(self, *a, **k):
        class T:
            def __enter__(s):
                return None

            def __exit__(s, *exc):
                return False
        return T()

    def create_node(self, **kw):
        self.nodes.append(kw)
        return len(self.nodes)


def _reply(*idxs):
    return json.dumps({"facts": [
        {"msg_index": i, "fact_sentence": "Fact number %d about the user." % n,
         "entity": "other", "evidence_sentence": "ev %d" % n,
         "fact_date": "2026-11-03"} for n, i in enumerate(idxs)]})


def test_only_current_message_facts_are_written(monkeypatch):
    monkeypatch.setattr(ing._dedup, "check_dedup", lambda *a, **k: ("new", None))
    store = FakeStore()
    core = FakeCore()
    # ingest() looks for `store.mem` or `store` and a write_txn on either
    store.mem = core
    res = ing.ingest("the current turn", said_at="2026-11-03T10:00:00Z",
                     conv_id="c1", msg_index=7,
                     context=[{"role": "user", "text": "two back"},
                              {"role": "user", "text": "one back"}],
                     store=store, call_model=lambda body: _reply(0, 1, 2, 2),
                     embed_fn=lambda t: [0.5] * 1024, log_path="")
    assert res["dropped_ctx"] == 2, res
    assert len(res["written"]) == 2, res
    facts_written = [n for n in core.nodes if n["name"].startswith("f_")]
    assert len(facts_written) == 2, core.nodes      # (+1 for the embedder stamp)
    # every written node carries THIS turn's index, never the context's
    assert all(n["wiki"]["_msg_index"] == "7" for n in facts_written)
    assert all(n["wiki"]["_said_at"] == "2026-11-03T10:00:00Z" for n in facts_written)


def test_missing_msg_index_is_kept_as_current(monkeypatch):
    monkeypatch.setattr(ing._dedup, "check_dedup", lambda *a, **k: ("new", None))
    store = FakeStore()
    store.mem = FakeCore()
    res = ing.ingest("turn", store=store, call_model=lambda body: _reply(None),
                     embed_fn=lambda t: [0.5] * 1024, log_path="")
    assert res["dropped_ctx"] == 0
    assert len(res["written"]) == 1
