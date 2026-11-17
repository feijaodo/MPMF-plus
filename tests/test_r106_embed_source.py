"""r106: one rule owns "what string does a node's vector represent".

``embed_source(text, embed_from) = embed_text(embed_from or text)`` -- the summary is
metadata, never the source. Measured on the live store (random 200 of 25,120,
2026-11-17): 94.0% of vectors match the text, 62.5% match the summary, and the summary
only matches when summary == text[:120]. The r105 summary-first rule would have
re-pointed ~23,000 vectors to satisfy the 6% whose summary differs.

set_node_text re-embeds; set_node_summary never does.
"""
import hashlib

import numpy as np
import pytest

from mpmf.embed_text import EMBED_CHARS, EMBED_FROM_KEY, embed_source, embed_text
from mpmf.format import NodeType
from mpmf.query import MPMFMemory


class _FakeEmbedder:
    """Deterministic 8-d embedder: same string in, same vector out, no Ollama."""

    def __init__(self, dim=8):
        self.dim = dim

    def _vec(self, s):
        h = hashlib.sha256(str(s).encode()).digest()
        return np.frombuffer((h * 4)[: self.dim * 4], dtype=np.uint32).astype(np.float32) / 1e9

    def embed(self, s, kind=None):
        return self._vec(s)

    def pack(self, s):                      # create_node() wants packed float32 bytes
        return self._vec(s).astype(np.float32).tobytes()


def _store(tmp_path, name="s.mpmf"):
    m = MPMFMemory(str(tmp_path / name), vector_dimensions=8)
    fake = _FakeEmbedder()
    m.embedder = fake
    m.mem.embedder = lambda t: fake.embed(t, kind="document")
    return m, fake


def _node(m, text="original text", summary="", wiki=None):
    with m.mem.write_txn():
        rec = m.mem.create_node("n1", NodeType.FACT, 1, wiki=wiki)
        nid = getattr(rec, "node_id", getattr(rec, "id", rec))
        m.mem.set_node_text(nid, text)
        if summary:
            m.mem.set_node_summary(nid, summary)
        m.mem.set_node_vector(nid, m.mem.embedder(m.mem.embed_source_of(nid)))
    return nid


# ---------------------------------------------------------------- the rule itself

def test_embed_from_wins_then_text():
    assert embed_source("the text", "the override") == "the override"
    assert embed_source("the text", "") == "the text"
    assert embed_source("", "") == ""


def test_the_summary_is_not_the_source():
    """The only signature is (text, embed_from): a summary cannot enter the rule."""
    import inspect
    assert list(inspect.signature(embed_source).parameters) == ["text", "embed_from"]


def test_the_rule_truncates_through_embed_text():
    assert embed_source("x" * 900, "") == ("x" * 900)[:EMBED_CHARS]
    assert len(embed_source("x" * 900)) == EMBED_CHARS


def test_remember_and_the_store_agree(tmp_path):
    m, fake = _store(tmp_path)
    with m.mem.write_txn():
        rec = m.remember("a fact worth keeping", "fact-1", summary="a curated summary")
    nid = getattr(rec, "node_id", getattr(rec, "id", rec))
    assert list(m.mem.get_vector_list(nid)) == list(fake.embed("a fact worth keeping"))
    assert m.mem.embed_source_of(nid) == "a fact worth keeping"


def test_remember_keeps_embed_from_with_the_node(tmp_path):
    m, fake = _store(tmp_path)
    with m.mem.write_txn():
        rec = m.remember("a fact", "fact-2", summary="sum", embed_from="override")
    nid = getattr(rec, "node_id", getattr(rec, "id", rec))
    assert m.mem.embed_source_of(nid) == "override"
    assert (m.mem.get_wiki(nid) or {}).get(EMBED_FROM_KEY) == "override"


# ------------------------------------------------------------ text edits re-embed

def test_text_edit_reembeds(tmp_path):
    m, _ = _store(tmp_path)
    nid = _node(m)
    before = list(m.mem.get_vector_list(nid))
    with m.mem.write_txn():
        m.mem.set_node_text(nid, "completely different text")
    assert list(m.mem.get_vector_list(nid)) != before


def test_text_edit_reembeds_even_when_the_node_has_a_summary(tmp_path):
    """THE case: a summary must not shadow the text, or the vector goes stale."""
    m, fake = _store(tmp_path)
    nid = _node(m, text="the original text", summary="the curated summary")
    before = list(m.mem.get_vector_list(nid))
    with m.mem.write_txn():
        m.mem.set_node_text(nid, "a wholly rewritten text")
    assert list(m.mem.get_vector_list(nid)) != before
    assert list(m.mem.get_vector_list(nid)) == list(fake.embed("a wholly rewritten text"))


def test_unchanged_text_needs_no_vector(tmp_path):
    m, _ = _store(tmp_path)
    nid = _node(m, text="text", summary="summary")
    with m.mem.write_txn():
        m.mem.set_node_text(nid, "text")
        m.mem.set_node_importance(nid, 0.9)


# --------------------------------------------------------- summary edits are inert

def test_summary_edit_does_not_touch_the_vector(tmp_path):
    m, _ = _store(tmp_path)
    nid = _node(m, text="the text", summary="the summary")
    before = list(m.mem.get_vector_list(nid))
    with m.mem.write_txn():
        m.mem.set_node_summary(nid, "a better summary")
    assert list(m.mem.get_vector_list(nid)) == before
    assert m.mem.get_summary(nid) == "a better summary"
    assert m.mem.embed_source_of(nid) == "the text"


def test_summary_edit_is_allowed_with_no_embedder(tmp_path):
    m, _ = _store(tmp_path)
    nid = _node(m, text="text", summary="summary")
    m.mem.embedder = None
    with m.mem.write_txn():
        m.mem.set_node_summary(nid, "changed summary, same source")
    assert m.mem.get_summary(nid) == "changed summary, same source"


# ------------------------------------------------------ embed_from: it owns the rule

def test_embed_from_makes_text_and_summary_edits_inert(tmp_path):
    m, fake = _store(tmp_path)
    nid = _node(m, text="text", summary="summary", wiki={EMBED_FROM_KEY: "the override"})
    before = list(m.mem.get_vector_list(nid))
    with m.mem.write_txn():
        m.mem.set_node_text(nid, "new text")
        m.mem.set_node_summary(nid, "new summary")
    assert list(m.mem.get_vector_list(nid)) == before
    with m.mem.write_txn():
        m.mem.update_node_wiki(nid, {EMBED_FROM_KEY: "a different override"})
    assert list(m.mem.get_vector_list(nid)) == list(fake.embed("a different override"))


# ------------------------------------------------------------------- the guard

def test_no_embedder_refuses_a_text_change(tmp_path):
    m, _ = _store(tmp_path)
    nid = _node(m, text="text", summary="summary")
    m.mem.embedder = None
    with pytest.raises(RuntimeError, match="embed source .* without set_node_vector"):
        with m.mem.write_txn():
            m.mem.set_node_text(nid, "changed with no embedder")


def test_no_embedder_allows_text_change_with_explicit_vector(tmp_path):
    m, fake = _store(tmp_path)
    nid = _node(m, text="text", summary="summary")
    m.mem.embedder = None
    with m.mem.write_txn():
        m.mem.set_node_text(nid, "handled by the caller")
        m.mem.set_node_vector(nid, fake.embed("handled by the caller"))


def test_raise_leaves_the_store_unsaved(tmp_path):
    m, _ = _store(tmp_path)
    nid = _node(m, text="text", summary="summary")
    m.mem.embedder = None
    with pytest.raises(RuntimeError):
        with m.mem.write_txn():
            m.mem.set_node_text(nid, "lost")
    fresh = MPMFMemory(str(tmp_path / "s.mpmf"), vector_dimensions=8)
    assert fresh.mem.get_text(nid) == "text"
