"""r105: one rule owns "what string does a node's vector represent".

``embed_source(text, summary, embed_from) = embed_text(embed_from or summary or text)``
and the store re-embeds only when the WINNING string changes. The r103 rule keyed on
text, which re-pointed 1,783 summary-embedded vectors at their text and made the daily
audit indict them.
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

def test_embed_from_wins_then_summary_then_text():
    assert embed_source("the text", "the summary", "the override") == "the override"
    assert embed_source("the text", "the summary", "") == "the summary"
    assert embed_source("the text", "", "") == "the text"
    assert embed_source("", "", "") == ""


def test_the_rule_truncates_through_embed_text():
    long_summary = "s" * (EMBED_CHARS + 100)
    assert embed_source("text", long_summary, "") == long_summary[:EMBED_CHARS]
    assert len(embed_source("x" * 900, "", "")) == EMBED_CHARS


def test_remember_and_the_store_agree(tmp_path):
    """remember() must not have a private copy of the precedence."""
    m, fake = _store(tmp_path)
    with m.mem.write_txn():
        rec = m.remember("a fact worth keeping", "fact-1", summary="a curated summary")
    nid = getattr(rec, "node_id", getattr(rec, "id", rec))
    assert list(m.mem.get_vector_list(nid)) == list(
        fake.embed(embed_source("a fact worth keeping", "a curated summary", "")))
    assert m.mem.embed_source_of(nid) == "a curated summary"


def test_remember_keeps_embed_from_with_the_node(tmp_path):
    m, fake = _store(tmp_path)
    with m.mem.write_txn():
        rec = m.remember("a fact", "fact-2", summary="sum", embed_from="override")
    nid = getattr(rec, "node_id", getattr(rec, "id", rec))
    assert m.mem.embed_source_of(nid) == "override"
    assert (m.mem.get_wiki(nid) or {}).get(EMBED_FROM_KEY) == "override"


# ------------------------------------------------- text is the source: text edits

def test_text_source_node_reembeds_when_text_changes(tmp_path):
    m, _ = _store(tmp_path)
    nid = _node(m)                       # no summary -> text is the source
    before = list(m.mem.get_vector_list(nid))
    with m.mem.write_txn():
        m.mem.set_node_text(nid, "completely different text")
    assert m.mem.get_text(nid) == "completely different text"
    assert list(m.mem.get_vector_list(nid)) != before


def test_text_source_node_gaining_a_summary_moves_the_source(tmp_path):
    """The new summary becomes the source, so the vector must follow it."""
    m, fake = _store(tmp_path)
    nid = _node(m)
    with m.mem.write_txn():
        m.mem.set_node_summary(nid, "now curated")
    assert m.mem.embed_source_of(nid) == "now curated"
    assert list(m.mem.get_vector_list(nid)) == list(fake.embed("now curated"))


# --------------------------------------- summary is the source: text edits are inert

def test_summary_source_node_ignores_text_edits(tmp_path):
    """THE case: a node that has a summary and gets its text edited."""
    m, _ = _store(tmp_path)
    nid = _node(m, text="the original text", summary="the curated summary")
    before = list(m.mem.get_vector_list(nid))
    with m.mem.write_txn():
        m.mem.set_node_text(nid, "a wholly rewritten text")
    assert m.mem.get_text(nid) == "a wholly rewritten text"
    assert list(m.mem.get_vector_list(nid)) == before      # vector must NOT change
    assert m.mem.embed_source_of(nid) == "the curated summary"


def test_summary_source_node_reembeds_when_summary_changes(tmp_path):
    m, fake = _store(tmp_path)
    nid = _node(m, text="text", summary="the curated summary")
    with m.mem.write_txn():
        m.mem.set_node_summary(nid, "a better summary")
    assert list(m.mem.get_vector_list(nid)) == list(fake.embed("a better summary"))


def test_clearing_the_summary_falls_back_to_text(tmp_path):
    m, fake = _store(tmp_path)
    nid = _node(m, text="the text", summary="the summary")
    with m.mem.write_txn():
        m.mem.set_node_summary(nid, "")
    assert m.mem.embed_source_of(nid) == "the text"
    assert list(m.mem.get_vector_list(nid)) == list(fake.embed("the text"))


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

def test_no_embedder_refuses_a_source_change(tmp_path):
    m, _ = _store(tmp_path)
    nid = _node(m, text="text", summary="summary")
    m.mem.embedder = None
    with pytest.raises(RuntimeError, match="embed source .* without set_node_vector"):
        with m.mem.write_txn():
            m.mem.set_node_summary(nid, "changed with no embedder")


def test_no_embedder_allows_an_inert_text_edit(tmp_path):
    """Editing text that is not the source changes nothing, so nothing is refused."""
    m, _ = _store(tmp_path)
    nid = _node(m, text="text", summary="summary")
    m.mem.embedder = None
    with m.mem.write_txn():
        m.mem.set_node_text(nid, "rewritten text")
    assert m.mem.get_text(nid) == "rewritten text"


def test_no_embedder_allows_source_change_with_explicit_vector(tmp_path):
    m, fake = _store(tmp_path)
    nid = _node(m, text="text", summary="summary")
    m.mem.embedder = None
    with m.mem.write_txn():
        m.mem.set_node_summary(nid, "handled by the caller")
        m.mem.set_node_vector(nid, fake.embed("handled by the caller"))


def test_raise_leaves_the_store_unsaved(tmp_path):
    m, _ = _store(tmp_path)
    nid = _node(m, text="text", summary="summary")
    m.mem.embedder = None
    with pytest.raises(RuntimeError):
        with m.mem.write_txn():
            m.mem.set_node_summary(nid, "lost")
    fresh = MPMFMemory(str(tmp_path / "s.mpmf"), vector_dimensions=8)
    assert fresh.mem.get_summary(nid) == "summary"
