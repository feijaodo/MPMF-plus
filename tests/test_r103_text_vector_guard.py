"""r103: text and vector move together, enforced by the store layer."""
import numpy as np
import pytest

from mpmf.format import NodeType
from mpmf.query import MPMFMemory


def _store(tmp_path):
    """A store with NO embedder — the shape a cron has when it forgets to wire one."""
    m = MPMFMemory(str(tmp_path / "s.mpmf"), vector_dimensions=8)
    m.mem.embedder = None                  # r103: opt out of the store-level auto-embed
    with m.mem.write_txn():
        rec = m.mem.create_node("n1", NodeType.FACT, 1)
        nid = getattr(rec, "node_id", getattr(rec, "id", rec))
        m.mem.set_node_text(nid, "original text")
        m.mem.set_node_vector(nid, np.ones(8, dtype=np.float32))
    return m, nid


def test_text_change_without_vector_raises(tmp_path):
    m, nid = _store(tmp_path)
    with pytest.raises(RuntimeError, match="without set_node_vector"):
        with m.mem.write_txn():
            m.mem.set_node_text(nid, "different text")


def test_default_store_auto_embeds_so_the_guard_never_fires(tmp_path):
    """The shipped shape: MPMFMemory installs the canonical text -> vector hook, so a
    text change re-embeds in the same txn and no drift is possible."""
    m = MPMFMemory(str(tmp_path / "s.mpmf"), vector_dimensions=8)
    assert m.mem.embedder is not None
    with m.mem.write_txn():
        rec = m.mem.create_node("n1", NodeType.FACT, 1)
        nid = getattr(rec, "node_id", getattr(rec, "id", rec))
        m.mem.set_node_text(nid, "original text")
        m.mem.set_node_vector(nid, np.ones(8, dtype=np.float32))
    before = list(m.mem.get_vector_list(nid))
    with m.mem.write_txn():
        m.mem.set_node_text(nid, "completely different text")
    assert m.mem.get_text(nid) == "completely different text"
    assert list(m.mem.get_vector_list(nid)) != before


def test_text_change_with_vector_is_allowed(tmp_path):
    m, nid = _store(tmp_path)
    with m.mem.write_txn():
        m.mem.set_node_text(nid, "different text")
        m.mem.set_node_vector(nid, np.ones(8, dtype=np.float32) * 0.5)
    assert m.mem.get_text(nid) == "different text"


def test_unchanged_text_needs_no_vector(tmp_path):
    m, nid = _store(tmp_path)
    with m.mem.write_txn():
        m.mem.set_node_text(nid, "original text")
        m.mem.set_node_importance(nid, 0.9)


def test_raise_leaves_store_unsaved(tmp_path):
    m, nid = _store(tmp_path)
    with pytest.raises(RuntimeError):
        with m.mem.write_txn():
            m.mem.set_node_text(nid, "mutated")
    fresh = MPMFMemory(str(tmp_path / "s.mpmf"), vector_dimensions=8)
    assert fresh.mem.get_text(nid) == "original text"
