"""r107: EVERY writer that stores a vector must store the vector of embed_source.

The rule (r106, unchanged):  ``embed_source(text, embed_from) = embed_text(embed_from or text)``
and ``embed_text(t) = t[:512]``. This file proves each writer OBEYS it for a node
whose text is longer than 512 chars — the case where the two old truncation rules
disagree (head-only 512 vs head+tail 1200) and where embedding the SUMMARY instead
of the text silently passes every other test.

Why the fake lives at ``_embed_http_once`` and not at ``_embed_floats_local``:
patching the outer function would bypass ``_truncate_for_embed`` and the test
would pass even while the writer sent the wrong string. Patching the single
Ollama round-trip means the fake sees EXACTLY what would have been embedded, so
"cos == 1.0 against embed_source" is a real assertion about the pipeline.

Writers covered (the complete inventory from the r107 grep, see the module list
in the r107 report):
  ingest      mpmf_plus.ingest.ingest()            -> node vector
  remember    mpmf.query.MPMFMemory.remember()     -> node vector
  cog stage   cognition.stage_lesson()             -> store() -> node vector
  enrichment  memory.store(auto_extract=True)      -> _apply_extraction_to_node
  rumination  historical_rumination -> stage_lesson (same store() writer as cog)
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mpmf.embed_text import embed_source, embed_text  # noqa: E402
from mpmf.memory import MemoryManager  # noqa: E402
from mpmf_plus import memory as M  # noqa: E402

DIM = 1024
LONG = (
    "Round 107 vector-source proof. " * 40
)  # ~1240 chars: over 512 and over the old 1200 head+tail threshold
assert len(LONG) > 1200


def fake_vec(text: str):
    """Deterministic vector of the STRING THE EMBEDDER RECEIVED."""
    h = hashlib.sha256(text.encode("utf-8", "ignore")).digest()
    v = [((h[i % 32] / 255.0) - 0.5) for i in range(DIM)]
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def _packv(v):
    """float list -> packed float32 bytes (the store's vector field type)."""
    return struct.pack("<%df" % DIM, *v)


def cos(a, b) -> float:
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(x * x for x in b) ** 0.5 or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A throwaway store + a fake Ollama round-trip in BOTH library modules."""
    path = str(tmp_path / "r107.mpmf")
    mm = MemoryManager(path, vector_dimensions=DIM)
    with mm.write_txn():                     # materialise the file
        mm.create_node(name="seed", node_type=1, plane_flags=1, text="seed",
                       summary="seed", wiki={"_ns": "memory"},
                       vector=_packv(fake_vec("seed")), importance=0.5)

    sent = []

    def fake_once(text):
        sent.append(text)
        return fake_vec(text)

    monkeypatch.setattr(M, "_embed_http_once", fake_once, raising=True)
    monkeypatch.setattr(M, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(M, "EXTRACTIONS_CACHE", tmp_path / "cache" / "ex.json")

    monkeypatch.setenv("MPMF_STORE_PATH", path)   # any later settings resolve
    prev = object.__getattribute__(M._VIEW, "_settings")
    M.configure_store(path=path, mode="attach")
    try:
        yield {"path": path, "sent": sent}
    finally:
        object.__setattr__(M._VIEW, "_settings", prev)
        M._invalidate_open_cache()


def _vector_of(path, name):
    mm = MemoryManager(path, vector_dimensions=DIM)
    mm.load()
    nid = mm.get_node_id_by_name(name)
    assert nid is not None, f"{name} was not written"
    return mm.get_vector(nid)


def _assert_source_vector(path, name, text, embed_from=""):
    want = embed_source(text, embed_from)
    got = _vector_of(path, name)
    assert cos(got, fake_vec(want)) > 0.999999, (
        f"{name}: stored vector is not embed_source()\n"
        f"  source len {len(want)}  text len {len(text)}\n"
        f"  cos vs embed_source = {cos(got, fake_vec(want)):.6f}\n"
        f"  cos vs summary     = "
        f"{cos(got, fake_vec(embed_text(text[:300]))):.6f}")


# ---------------------------------------------------------------- the writers

class _FakeEmbedder:
    """Stands in for EmbeddingProvider: same rule, deterministic output."""

    def embed(self, text, kind="document"):
        return fake_vec(text)

    def pack(self, text):
        return struct.pack("<%df" % DIM, *fake_vec(text))


def test_remember_embeds_embed_source(store):
    from mpmf.query import MPMFMemory
    m = MPMFMemory(store["path"], vector_dimensions=DIM, use_ollama=False)
    m.embedder = _FakeEmbedder()
    with m.mem.write_txn():
        m.remember(LONG, name="r107_remember")
    _assert_source_vector(store["path"], "r107_remember", LONG)


def test_remember_honours_embed_from(store):
    from mpmf.query import MPMFMemory
    from mpmf.embed_text import EMBED_FROM_KEY
    m = MPMFMemory(store["path"], vector_dimensions=DIM, use_ollama=False)
    m.embedder = _FakeEmbedder()
    with m.mem.write_txn():
        m.remember(LONG, name="r107_embed_from", embed_from="the short version")
    _assert_source_vector(store["path"], "r107_embed_from", LONG, "the short version")


def test_store_does_not_embed_the_summary(store):
    """B5: store() used to embed `summary or text[:120]`. It must embed the text."""
    M.store(LONG, name="r107_store", summary=LONG[:300],
            auto_extract=False, auto_link=False)
    _assert_source_vector(store["path"], "r107_store", LONG)


def test_enrichment_does_not_embed_the_summary(store, monkeypatch):
    """The enrichment writer (_apply_extraction_to_node) must keep the text vector."""
    def fake_extract(text, **kw):
        return {"summary": "an enriched summary that is NOT the source",
                "namespace": "memory", "wiki": {}, "planes": {}, "_model": "fake"}

    monkeypatch.setattr(M, "call_extractor", fake_extract)
    monkeypatch.setattr(M, "_query_similar_nodes", lambda *a, **k: [])
    M.store(LONG, name="r107_enriched", summary="", auto_extract=True,
            auto_link=False)
    mm = MemoryManager(store["path"], vector_dimensions=DIM)
    mm.load()
    nid = mm.get_node_id_by_name("r107_enriched")
    assert nid is not None
    assert (mm.get_wiki(nid) or {}).get("_enrichment_status") == "enriched"
    _assert_source_vector(store["path"], "r107_enriched", LONG)


def test_cog_stage_lesson_stores_the_text_vector(store):
    """The cog/rumination writer: stage_lesson -> store(). Same rule."""
    from mpmf_plus import cognition as cg
    r = cg.stage_lesson(LONG + " | trigger clause | question clause",
                        session="r107-proof", rule_id="r107_source_rule",
                        origin="rumination", auto_extract=False)
    assert r["action"] == "staged", r
    _assert_source_vector(store["path"], r["name"], LONG + " | trigger clause | question clause")


def test_ingest_sentence_vector_matches_its_source(store):
    from mpmf_plus import ingest as I

    sentence = LONG[:1000]
    reply = json.dumps({"facts": [{
        "fact_sentence": sentence, "entity": "user", "msg_index": 0,
        "evidence_sentence": "the r107 proof", "kind": "have",
        "mode": "actual", "fact_date": "2026-11-18"}]})

    mm = MemoryManager(store["path"], vector_dimensions=DIM)
    mm.load()
    out = I._ingest_impl("a turn long enough to count", said_at="2026-11-18T00:00:00Z",
                         conv_id="r107", store=mm, call_model=lambda *a, **k: reply,
                         log_path="", namespace="memory")
    assert out["status"] == "ok", out
    w = out["written"][0]
    nid = int(w.get("node_id") or w.get("nid") or w.get("id"))
    mm2 = MemoryManager(store["path"], vector_dimensions=DIM)
    mm2.load()
    assert cos(mm2.get_vector(nid), fake_vec(embed_source(sentence))) > 0.999999


# ------------------------------------------------------- the rule itself

def test_truncation_has_one_rule_in_both_modules():
    """No cap, any cap: the string handed to the embedder is embed_text(text)."""
    from mpmf_plus import provider as P
    for cap in (None, 512, 1200, 4096):
        assert M._truncate_for_embed(LONG, cap) == embed_text(LONG), (M.__name__, cap)
        assert P._truncate_for_embed(LONG, cap) == embed_text(LONG), (P.__name__, cap)


def test_embed_floats_sends_the_truncated_string(store):
    store["sent"].clear()
    M._embed_floats_local(LONG)
    assert store["sent"] == [embed_text(LONG)]


def test_doc_source_is_embed_from_or_text():
    from mpmf.embed_text import EMBED_FROM_KEY
    assert M._doc_source("the text", {}) == embed_text("the text")
    assert M._doc_source("the text", {EMBED_FROM_KEY: "src"}) == embed_text("src")
    assert M._doc_source("the text", {EMBED_FROM_KEY: ""}) == embed_text("the text")
