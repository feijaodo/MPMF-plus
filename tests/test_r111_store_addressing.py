"""r111: store() must address the node IT created, on every path.

Live shape this pins (2026-11-18): 117 names cover 295 nodes, and `S_openclaw`
existed twice. `store()` persists a raw node, then calls the extraction LLM
(outside the lock), then upgrades "the node" in place. Until r111 both copies
omitted the created id on the SUCCESS path and re-resolved the fact by NAME —
so an extraction that returned while another writer had created a same-named
sibling wrote its summary/wiki/vector/status onto that sibling. r109 fixed the
sweep (which had an id in hand) and r110 fixed only the FAILURE flag; the
success path is what these tests cover.

Five invariants, each asserted through the real public `store()` /
`re_enrich_pending()` of BOTH copies (they are still two files until r115):

  1. interleaved same-name writers each keep their own summary/wiki/vector/status;
  2. auto_extract=False + auto_link does not touch a same-name sibling;
  3. a setter that raises must not be reported as success, and must not leave
     the node marked `_enrichment_status=enriched`;
  4. a commit that fails must not be reported as success or leave it enriched;
  5. a node that already carries `embed_from` keeps a vector built from
     `embed_from` after enrichment (the merged-wiki rule, Q11).

Embedding is an external service in production, so it is replaced with a
deterministic function here — that is the seam, not the logic under test.
Everything else (dedup, txn, save, reload) is the real code path.
"""

import json
import math
import zlib
from pathlib import Path

import pytest

from mpmf.memory import MemoryManager

DIM = 1024

# Distinct topics on purpose: the dedup check must NOT collapse them, or the
# test would silently exercise the dedup early-return instead of the writer.
TEXT_A = "the gateway listens on port 18789 and restarts with slash restart"
TEXT_B = "the weather station logged frost on the north field at dawn"
TEXT_C = "a spare bicycle chain hangs in the workshop behind the blue door"
OVERRIDE = "override sentence that is not the node text at all"
SAME = "S_r111_dup"


def _vec(text):
    """Deterministic bag-of-words embedder, L2-normalised like the real one.

    Same string -> same vector (so `cos == 1.0` is meaningful); different
    strings -> different vectors (so a misdirected write is visible). Uses a
    crc32 bucket per token, which is stable across processes.
    """
    v = [0.0] * DIM
    for tok in str(text or "").lower().replace("/", " ").replace(".", " ").split():
        h = zlib.crc32(tok.encode())
        v[h % DIM] += 1.0 + (h % 7) / 10.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _cos(a, b):
    if a is None or b is None:
        return 0.0
    num = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) ** 2 for x in a))
    nb = math.sqrt(sum(float(y) ** 2 for y in b))
    return num / (na * nb) if na and nb else 0.0


def _reload(path):
    """A fresh reader over the file on disk — the only honest way to ask
    'what did we actually keep?', because the writer's in-memory image can be
    richer than what got saved."""
    m = MemoryManager(str(path), vector_dimensions=DIM)
    m.load()
    return m


def _ids_with_text(m, text):
    return [nid for nid in m.all_node_ids() if (m.get_text(nid) or "") == text]


def _one_with_text(m, text):
    ids = _ids_with_text(m, text)
    assert len(ids) == 1, "expected exactly one node with this text, got %r" % (ids,)
    return ids[0]


@pytest.fixture(params=["provider", "memory"])
def env(request, tmp_path, monkeypatch):
    """Point ONE copy at a throwaway store and stub only the external seams."""
    import importlib

    name = request.param
    mod = importlib.import_module(
        "mpmf_plus.provider" if name == "provider" else "mpmf_plus.memory")

    store = tmp_path / ("r111_%s.mpmf" % name)
    # Both copies keep their own PATH global (r111 is explicitly not the merge
    # round), so the path is set on the module under test and nowhere else.
    monkeypatch.setattr(mod, "PATH", str(store), raising=False)
    if hasattr(mod, "PATH_LEGACY"):
        monkeypatch.setattr(mod, "PATH_LEGACY", str(store) + ".v1-legacy", raising=False)
    if hasattr(mod, "EXTRACTIONS_CACHE"):
        monkeypatch.setattr(mod, "EXTRACTIONS_CACHE",
                            tmp_path / ("extractions_%s.json" % name), raising=False)
    # provider.store() delegates to the turn-ingest path unless the ingest
    # config says "legacy"; that file is the real switch, so use it rather than
    # patching _ingest_mode().
    if hasattr(mod, "_INGEST_CFG"):
        cfg = tmp_path / "ingest.config.json"
        cfg.write_text(json.dumps({"mode": "legacy"}))
        monkeypatch.setattr(mod, "_INGEST_CFG", cfg, raising=False)

    # The embedder is an external service: stub it so the expected vector is
    # computable. `_pack` and every writer resolve this global at call time.
    monkeypatch.setattr(mod, "_embed_floats_local",
                        lambda t=None, dim=None, cap=None, **_k: _vec(t), raising=False)
    # No similarity context: keep the test off any store-wide scan.
    if hasattr(mod, "_query_similar_nodes"):
        monkeypatch.setattr(mod, "_query_similar_nodes",
                            lambda t, top_k=5: [], raising=False)

    if hasattr(mod, "configure_store"):
        mod.configure_store(str(store), mode="create")
    for fn in ("_invalidate_open_cache",):
        if hasattr(mod, fn):
            getattr(mod, fn)()

    return {"mod": mod, "name": name, "path": store}


def _extraction(summary, **wiki):
    """The minimal shape _resolve_payload() accepts."""
    return {"summary": summary,
            "wiki": dict(wiki),
            "planes": {"associative": [], "priority": 3}}


def test_interleaved_same_name_writers_keep_their_own_payload(env, monkeypatch):
    """A's extraction returns AFTER B has created a same-named node.

    A must still land on A. On b934998 A resolves the name again, finds B (the
    newest owner of the slot) and writes A's summary/wiki/vector/status there,
    leaving A itself flagged `pending` forever.
    """
    mod, path = env["mod"], env["path"]
    nested = {"done": False}

    def extractor(text, **kw):
        if text == TEXT_A and not nested["done"]:
            # The other writer creates a same-named sibling while A is waiting
            # on the LLM. store() calls this outside the write lock, exactly
            # like production, so this is not a re-entrant lock.
            nested["done"] = True
            mod.store(text=TEXT_B, name=SAME, summary="raw-b", auto_link=False)
        return _extraction("enriched-a" if text == TEXT_A else "enriched-b",
                           _life_kind="test")

    monkeypatch.setattr(mod, "call_extractor", extractor, raising=False)

    mod.store(text=TEXT_A, name=SAME, summary="raw-a", auto_link=False)

    m = _reload(path)
    a, b = _one_with_text(m, TEXT_A), _one_with_text(m, TEXT_B)

    assert (m.get_name(a), m.get_name(b)) == (SAME, SAME)
    assert m.get_summary(a) == "enriched-a", "A got %r" % (m.get_summary(a),)
    assert m.get_summary(b) == "enriched-b", "B was overwritten: %r" % (m.get_summary(b),)
    for nid, src, label in ((a, TEXT_A, "A"), (b, TEXT_B, "B")):
        assert (m.get_wiki(nid) or {}).get("_enrichment_status") == "enriched", label
        got = _cos(m.get_vector_list(nid), _vec(src))
        assert got > 0.9999, "%s vector does not follow its own text (cos=%r)" % (label, got)


def test_auto_extract_false_does_not_touch_a_same_name_sibling(env):
    """auto_extract=False + auto_link writes the new node only.

    The edge rebuild is a WRITE to the addressed node; with a duplicate name it
    must be the node this call created, not whichever sibling owns the slot.
    """
    mod, path = env["mod"], env["path"]

    mod.store(text=TEXT_B, name=SAME, summary="raw-b",
              auto_extract=False, auto_link=True)
    before = _reload(path)
    sib = _one_with_text(before, TEXT_B)
    sib_edges = sorted((e.edge_id, e.source_node_id, e.target_node_id)
                       for e in before._edges)

    mod.store(text=TEXT_C, name=SAME, summary="raw-c",
              auto_extract=False, auto_link=True)

    m = _reload(path)
    sib2, new = _one_with_text(m, TEXT_B), _one_with_text(m, TEXT_C)

    assert m.get_summary(sib2) == "raw-b"
    assert m.get_summary(new) == "raw-c"
    assert _cos(m.get_vector_list(sib2), _vec(TEXT_B)) > 0.9999
    assert _cos(m.get_vector_list(new), _vec(TEXT_C)) > 0.9999
    assert sorted((e.edge_id, e.source_node_id, e.target_node_id)
                  for e in m._edges) == sib_edges, (
        "the sibling's edge set changed when a same-named node was added")


def test_raising_setter_must_not_report_success_or_enrich(env, monkeypatch):
    """A swallowed setter failure used to return success and mark `enriched`.

    That is the combination that hides corruption: the store claims the node is
    enriched while its summary or vector is still the phase-1 raw one.
    """
    mod, path = env["mod"], env["path"]
    monkeypatch.setattr(mod, "call_extractor",
                        lambda text, **kw: _extraction("enriched", _life_kind="test"),
                        raising=False)

    def boom(self, node_id, summary):
        raise RuntimeError("set_node_summary refused (r111 test)")

    monkeypatch.setattr(MemoryManager, "set_node_summary", boom)

    with pytest.raises(RuntimeError):
        mod.store(text=TEXT_A, name="S_r111_sum", summary="raw-a", auto_link=False)

    m = _reload(path)
    nid = _one_with_text(m, TEXT_A)
    assert (m.get_wiki(nid) or {}).get("_enrichment_status") != "enriched", (
        "node marked enriched although its summary write raised")


def test_raising_commit_must_not_report_success_or_enrich(env, monkeypatch):
    """A commit that fails must surface, and must not leave the node enriched."""
    from mpmf.query import MPMFMemory

    mod, path = env["mod"], env["path"]
    armed = {"on": False}

    def extractor(text, **kw):
        armed["on"] = True          # phase 1 has already committed by now
        return _extraction("enriched", _life_kind="test")

    monkeypatch.setattr(mod, "call_extractor", extractor, raising=False)

    real_save = MPMFMemory.save

    def flaky_save(self):
        if armed["on"]:
            raise OSError("no space left on device (r111 test)")
        return real_save(self)

    monkeypatch.setattr(MPMFMemory, "save", flaky_save)

    with pytest.raises(OSError):
        mod.store(text=TEXT_A, name="S_r111_commit", summary="raw-a", auto_link=False)

    m = _reload(path)
    nid = _one_with_text(m, TEXT_A)
    assert (m.get_wiki(nid) or {}).get("_enrichment_status") == "pending", (
        "a failed commit left the node claiming to be enriched")


def test_existing_embed_from_survives_enrichment(env, monkeypatch):
    """Enrichment must keep the vector on `embed_from`, not on the node text.

    `update_node_wiki` MERGES, so the old `embed_from` stays in the row while
    the vector — if computed from the freshly built payload instead of the
    merged stored wiki — starts describing the text. Both are "valid" vectors;
    only one matches the row's own stated source (SPEC I1).
    """
    mod, path = env["mod"], env["path"]

    mod.store(text=TEXT_A, name="S_r111_emb", summary="raw-a",
              wiki={"embed_from": OVERRIDE}, auto_extract=False, auto_link=False)

    seeded = _reload(path)
    nid0 = _one_with_text(seeded, TEXT_A)
    assert (seeded.get_wiki(nid0) or {}).get("embed_from") == OVERRIDE

    monkeypatch.setattr(mod, "call_extractor",
                        lambda text, **kw: _extraction("enriched-emb", _life_kind="test"),
                        raising=False)

    mod.re_enrich_pending(limit=10, auto_link=False)

    m = _reload(path)
    nid = _one_with_text(m, TEXT_A)
    wiki = m.get_wiki(nid) or {}
    assert wiki.get("_enrichment_status") == "enriched", wiki.get("_enrichment_status")
    assert m.get_summary(nid) == "enriched-emb"
    assert wiki.get("embed_from") == OVERRIDE, "enrichment dropped the override"
    got = _cos(m.get_vector_list(nid), _vec(OVERRIDE))
    assert got > 0.9999, (
        "vector follows the node text, not the surviving embed_from (cos=%r)" % (got,))
