"""r102: every node write method must raise outside write_txn() instead of silently no-oping.

Covered methods (all public node mutators on the core MemoryManager):
  create_node, create_edge, update_node_wiki, set_node_text, set_node_summary,
  set_node_vector, set_node_importance
"""
import numpy as np
import pytest

from mpmf.format import NodeType, Plane
from mpmf.query import MPMFMemory

DIM = 8

GUARDED = [
    ("create_node", lambda m, nid: m.create_node("x_%d" % nid, NodeType.FACT, 1)),
    ("create_edge", lambda m, nid: m.create_edge(nid, nid, Plane.CAUSAL, "rel")),
    ("update_node_wiki", lambda m, nid: m.update_node_wiki(nid, {"k": "v"})),
    ("set_node_text", lambda m, nid: m.set_node_text(nid, "new text")),
    ("set_node_summary", lambda m, nid: m.set_node_summary(nid, "new summary")),
    ("set_node_vector", lambda m, nid: m.set_node_vector(nid, np.ones(DIM, dtype=np.float32))),
    ("set_node_importance", lambda m, nid: m.set_node_importance(nid, 0.9)),
]


def _nid(rec):
    return getattr(rec, "node_id", getattr(rec, "id", rec))


@pytest.fixture()
def mem(tmp_path):
    m = MPMFMemory(str(tmp_path / "g.mpmf"), vector_dimensions=DIM).mem
    with m.write_txn():
        nid = _nid(m.create_node("seed", NodeType.FACT, 1))
        m.set_node_text(nid, "original")
    return m, nid


@pytest.mark.parametrize("name,call", GUARDED, ids=[g[0] for g in GUARDED])
def test_write_method_raises_outside_txn(mem, name, call):
    m, nid = mem
    with pytest.raises(RuntimeError) as e:
        call(m, nid)
    assert name + "() called outside write_txn()" in str(e.value)


@pytest.mark.parametrize("name,call", GUARDED, ids=[g[0] for g in GUARDED])
def test_write_method_works_inside_txn(mem, name, call):
    m, nid = mem
    with m.write_txn():
        call(m, nid)


def test_text_write_persists_when_wrapped(tmp_path):
    p = str(tmp_path / "p.mpmf")
    m = MPMFMemory(p, vector_dimensions=DIM).mem
    with m.write_txn():
        nid = _nid(m.create_node("seed2", NodeType.FACT, 1))
        m.set_node_text(nid, "wrapped text")
    again = MPMFMemory(p, vector_dimensions=DIM).mem
    again.load()
    assert again.get_text(nid) == "wrapped text"
