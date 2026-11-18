"""r108: the enrichment writer must address the node by ID, and its vector must
be the vector of the text that node's row actually holds.

Live evidence (2026-11-18 04:00-04:04Z sweep, 25 pending nodes): 117 names in
the store cover 295 nodes and `S_openclaw` existed twice, both pending. The
sweep extracted node A's text, then re-resolved the write target by NAME
(`get_node_id_by_name`), so one extraction's vector landed on the other node.
Result: 9 of 24 nodes ended with cos(vector, embed(embed_source)) between 0.59
and 0.86 while the writer reported success, and only 24 of 25 reported writes
were visible (one name collapsed).

Two invariants are pinned here:
  1. re_enrich_pending addresses each pending node by its own id, even when two
     pending nodes share a name;
  2. _apply_extraction_to_node embeds the text the addressed node now holds, not
     the (possibly stale, possibly sibling) `text` argument.
"""
import contextlib
import sys
import types

import pytest

import os

LIVE = os.environ.get("MPMF_LIB_PATH") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, LIVE)


class _Snap:
    """Two nodes, same name, different text — the live shape that broke."""

    def __init__(self):
        self.texts = {11: "alpha fact about the gateway on port 18789",
                      12: "beta fact about the module root /usr/lib/node_modules"}
        self.status = {11: "pending", 12: "pending"}

    def all_node_ids(self):
        return list(self.texts)

    def get_wiki(self, nid):
        return {"_enrichment_status": self.status.get(nid, "pending")}

    def get_name(self, nid):
        return "S_dup"          # identical name for both

    def get_text(self, nid):
        return self.texts[nid]


def test_re_enrich_pending_addresses_each_duplicate_by_id(monkeypatch):
    import mpmf_plus.provider as P

    snap = _Snap()
    monkeypatch.setattr(P, "_open", lambda: types.SimpleNamespace(mem=snap))
    monkeypatch.setattr(P, "_query_similar_nodes", lambda t, top_k=5: [])
    monkeypatch.setattr(P, "_invalidate_open_cache", lambda: None)
    monkeypatch.setattr(P, "call_extractor",
                        lambda text, **kw: {"summary": text[:80], "wiki": {}, "planes": {}})

    class Txn:
        mem = snap

        def save(self):
            pass

    @contextlib.contextmanager
    def fake_txn():
        yield Txn()

    monkeypatch.setattr(P, "_write_txn_open", fake_txn)
    monkeypatch.setattr(P, "_STORE_LOCK", contextlib.nullcontext())

    addressed = []

    def fake_apply(m, name, text, extraction, importance, auto_link,
                   incremental_link=True, node_id=None):
        addressed.append(node_id)
        snap.status[node_id] = "enriched"      # a real write flips the addressed node
        return True

    monkeypatch.setattr(P, "_apply_extraction_to_node", fake_apply)

    r = P.re_enrich_pending(limit=10, auto_link=False)
    assert r["pending"] == 2 and r["enriched"] == 2, r

    # r109: a write that does NOT flip the addressed node is a FAILURE, even when
    # the writer returned True — this is the "25 enriched, 0 failed" lie.
    monkeypatch.setattr(P, "_apply_extraction_to_node",
                        lambda *a, **kw: True)
    snap.status = {11: "pending", 12: "pending"}
    r2 = P.re_enrich_pending(limit=10, auto_link=False)
    assert r2["enriched"] == 0 and r2["still_pending"] == 2, r2
    assert addressed == [11, 12], "each pending node must be addressed by its own id, got %r" % (addressed,)


class _FakeMem:
    def __init__(self):
        self.texts = {1: "the stored text of node one", 2: "the sibling's text"}
        self.vec = {}

    def get_node_id_by_name(self, name):
        return 2                     # a name lookup would hit the wrong node

    def get_text(self, nid):
        return self.texts[nid]

    def get_wiki(self, nid):
        return {"_ns": "memory"}

    def get_node(self, nid):
        return types.SimpleNamespace(plane_flags=0)

    def set_node_summary(self, *a):
        pass

    def update_node_wiki(self, *a):
        pass

    def set_node_importance(self, *a):
        pass

    def set_node_vector(self, nid, packed):
        self.vec[nid] = packed


def test_vector_follows_the_addressed_node_text(monkeypatch):
    import mpmf_plus.provider as P

    monkeypatch.setattr(P, "_pack", lambda t: ("packed", t))
    mem = _FakeMem()

    class Txn:
        def save(self):
            pass

    txn = Txn()
    txn.mem = mem
    ok = P._apply_extraction_to_node(txn, "S_dup", "STALE SNAPSHOT TEXT",
                                     {"summary": "s"}, 0.6, False,
                                     incremental_link=False, node_id=1)
    assert ok is True
    assert mem.vec[1] == ("packed", "the stored text of node one"), mem.vec
    assert 2 not in mem.vec, "must not write the same-named sibling"

    # legacy call without node_id still resolves by name (2), but embeds ITS text
    ok = P._apply_extraction_to_node(txn, "S_dup", "STALE SNAPSHOT TEXT",
                                     {"summary": "s"}, 0.6, False, incremental_link=False)
    assert ok is True
    assert mem.vec[2] == ("packed", "the sibling's text"), mem.vec
