"""Round 79: the dedup candidate masks must survive a version bump.

Regression pin: masks used to be keyed on the store version, so every write
rebuilt both arrays. Now only appended rows are classified (plus a rotating
1/64 re-verification slice), and the verdict must not change.
"""
import numpy as np

from mpmf.memory import MemoryManager
from mpmf_plus import dedup_helpers as D


def _store(tmp_path, n=6):
    p = str(tmp_path / "m.mpmf")
    m = MemoryManager(p, vector_dimensions=8)
    import random
    rnd = random.Random(79)
    with m.write_txn():
        for i in range(n):
            m.create_node(name="n%d" % i, node_type=0, plane_flags=1,
                          text="text %d" % i, summary="",
                          wiki={"_ns": "memory" if i % 3 else "other"})
            nid = m.get_node_id_by_name("n%d" % i)
            assert nid is not None
            m.set_node_vector(nid, np.array([rnd.random() for _ in range(8)], dtype=np.float32))
    return p


def _full_rebuild(m):
    ids, M, norms, ver = m.vector_rows()
    ns = np.zeros(len(ids), bool)
    keep = np.ones(len(ids), bool)
    for i, nid in enumerate(ids):
        w = m.get_wiki(nid) or {}
        ns[i] = w.get("_ns", "memory") == "memory"
        keep[i] = not w.get("_superseded_by")
    return ns, keep


def test_masks_incremental_equals_full_rebuild(tmp_path):
    p = _store(tmp_path)
    m = MemoryManager(p, vector_dimensions=8)
    m.load()
    D.invalidate_masks()
    ids, M, norms, ver = m.vector_rows()
    ns1, keep1 = D._candidate_masks(m, ids, ver, "memory")
    assert (ns1 == _full_rebuild(m)[0]).all()

    # append rows, reopen (new instance, same path, new version)
    with m.write_txn():
        for i in range(2):
            m.create_node(name="x%d" % i, node_type=0, plane_flags=1,
                          text="extra %d" % i, summary="", wiki={"_ns": "memory"})
            _n = m.get_node_id_by_name("x%d" % i)
            assert _n is not None
            # vector_rows() only covers rows that HAVE a vector: a row-less
            # node would not grow the mask, which is the correct behaviour.
            m.set_node_vector(_n, np.array([0.1] * 8, dtype=np.float32))
    m2 = MemoryManager(p, vector_dimensions=8)
    m2.load()
    ids2, M2, norms2, ver2 = m2.vector_rows()
    assert len(ids2) == len(ids) + 2 and ver2 != ver
    ns2, keep2 = D._candidate_masks(m2, ids2, ver2, "memory")
    ref_ns, ref_keep = _full_rebuild(m2)
    assert (ns2 == ref_ns).all(), "incremental namespace mask drifted"
    assert (keep2 == ref_keep).all(), "incremental keep mask drifted"
    assert len(D._NEAR_CACHE["ids"]) == len(ids2)


def test_supersession_seen_after_refresh(tmp_path):
    """A superseded node must stop being a merge candidate."""
    p = _store(tmp_path)
    m = MemoryManager(p, vector_dimensions=8)
    m.load()
    D.invalidate_masks()
    ids, M, norms, ver = m.vector_rows()
    D._candidate_masks(m, ids, ver, "memory")
    _nid = m.get_node_id_by_name("n1")
    _sup = m.get_node_id_by_name("n2")
    assert _nid is not None and _sup is not None
    nid = int(_nid)
    with m.write_txn():
        m.update_node_wiki(nid, {"_superseded_by": str(int(_sup))})
    m2 = MemoryManager(p, vector_dimensions=8)
    m2.load()
    # force the rotating scan to cover every row: one bump per row
    ids2, M2, norms2, ver2 = m2.vector_rows()
    for k in range(len(ids2)):
        ns, keep = D._candidate_masks(m2, ids2, ver2 + k, "memory")
    row = list(ids2).index(nid)
    assert keep[row] is np.False_ or keep[row] == False  # noqa: E712
