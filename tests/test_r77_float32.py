"""Round 77 — vectors as float32 in memory, on-disk format unchanged.

What is pinned here:
  * load -> save -> reload is byte-identical (the vector block is untouched);
  * reads are zero-copy float32 row views, not per-node Python float lists;
  * write_txn can add and update rows (append + in-place upsert);
  * StaleImageError still guards a non-transactional save;
  * the embedder provenance node survives a round-trip with no vector;
  * empty / zero / wrong-length vectors fail loudly instead of making NaNs.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mpmf.format import NodeType, Plane  # noqa: E402
from mpmf.memory import MemoryManager, StaleImageError  # noqa: E402
from mpmf.vectors import VectorStore  # noqa: E402

DIM = 8


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return (v / np.float32(np.linalg.norm(v))).astype(np.float32)


def _open(path, dim: int = DIM) -> MemoryManager:
    m = MemoryManager(str(path), vector_dimensions=dim)
    m.load()
    return m


def _add(m: MemoryManager, seed: int, name: str = ""):
    with m.write_txn():
        return m.create_node(
            name or f"n{seed}",
            int(NodeType.FACT),
            int(Plane.ASSOCIATIVE.value),
            text=f"text {seed}",
            vector=_vec(seed).tobytes(),
        )


# --------------------------------------------------------------------- format
def test_round_trip_is_byte_identical(tmp_path):
    p = tmp_path / "s.mpmf"
    m = _open(p)
    for i in range(5):
        _add(m, i)

    before = p.read_bytes()
    m2 = _open(p)
    with m2.write_txn():
        pass  # no-op transaction: serialize + atomic write of a loaded image
    after = p.read_bytes()

    assert after == before, "save(load(f)) must not change one byte of the file"

    m3 = _open(p)
    for i in range(5):
        v = m3.get_vector(i + 1)
        assert v.dtype == np.float32
        np.testing.assert_allclose(v, _vec(i), rtol=0, atol=1e-6)


def test_vector_block_is_the_same_bytes_as_float32_rows(tmp_path):
    p = tmp_path / "blk.mpmf"
    m = _open(p)
    _add(m, 7)
    raw = p.read_bytes()
    hdr = m.header
    # The one stored row, read straight out of the file as float32.
    on_disk = np.frombuffer(raw, dtype="<f4", count=DIM, offset=hdr.vector_block_offset)
    np.testing.assert_array_equal(on_disk, _vec(7))


# ---------------------------------------------------------------------- reads
def test_reads_are_float32_row_views(tmp_path):
    p = tmp_path / "v.mpmf"
    m = _open(p)
    node = _add(m, 42)

    v = m.get_vector(node.node_id)
    assert isinstance(v, np.ndarray), type(v)
    assert v.dtype == np.float32
    assert v.flags.owndata is False, "a read must be a view into the matrix"
    assert not v.flags.writeable

    # The node dict holds a boolean, never a list of floats.
    content = m._content[node.node_id]
    assert not hasattr(content, "vector")
    assert "vector" not in content.__slots__
    assert content.has_vector is True

    ids, M, norms, version = m.vector_rows()
    assert M.dtype == np.float32 and M.shape == (1, DIM)
    assert M.flags.owndata is False, "snapshot must not rebuild a matrix copy"
    assert ids == (node.node_id,)
    assert version == m._vectors.version


def test_get_vector_list_is_the_opt_in_copy(tmp_path):
    p = tmp_path / "l.mpmf"
    m = _open(p)
    node = _add(m, 3)
    lst = m.get_vector_list(node.node_id)
    assert isinstance(lst, list) and len(lst) == DIM
    np.testing.assert_allclose(lst, _vec(3), rtol=0, atol=1e-6)


# --------------------------------------------------------------------- writes
def test_write_txn_add_and_update(tmp_path):
    p = tmp_path / "w.mpmf"
    m = _open(p)
    a = _add(m, 1)
    v1 = m.vector_rows()[3]

    b = _add(m, 2)  # append a second row
    ids, M, _n, v2 = m.vector_rows()
    assert M.shape == (2, DIM)
    assert v2 > v1, "appending must bump the version"
    assert set(ids) == {a.node_id, b.node_id}

    with m.write_txn():
        m.set_node_vector(a.node_id, _vec(9))  # in-place upsert of an existing row
    assert m.vector_rows()[3] > v2, "an update must bump the version too"
    assert len(m.vector_rows()[1]) == 2, "update must not append a duplicate row"
    np.testing.assert_allclose(m.get_vector(a.node_id), _vec(9), rtol=0, atol=1e-6)

    re = _open(p)
    np.testing.assert_allclose(re.get_vector(a.node_id), _vec(9), rtol=0, atol=1e-6)
    np.testing.assert_allclose(re.get_vector(b.node_id), _vec(2), rtol=0, atol=1e-6)


def test_drop_swaps_row_and_keeps_the_index_consistent():
    vs = VectorStore(4)
    vs.append(1, [1.0, 0.0, 0.0, 0.0])
    vs.append(2, [0.0, 1.0, 0.0, 0.0])
    vs.append(3, [0.0, 0.0, 1.0, 0.0])

    assert vs.drop(1) is True
    assert vs.get(1) is None
    assert vs.drop(1) is False
    assert len(vs) == 2 and set(vs.ids()) == {2, 3}
    np.testing.assert_allclose(vs.get(3), [0.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(vs.get(2), [0.0, 1.0, 0.0, 0.0])


def test_stale_image_error_still_fires(tmp_path):
    p = tmp_path / "st.mpmf"
    a = _open(p)
    _add(a, 1)
    b = _open(p)          # same image as a right now
    _add(a, 2)            # a saves; b's image is now stale
    with pytest.raises(StaleImageError):
        b.periodic_save()


def test_embedder_provenance_node(tmp_path):
    from mpmf_plus.store import read_embedder, stamp_embedder

    p = tmp_path / "pr.mpmf"
    m = _open(p)
    stamp_embedder(m, "test-embed", DIM)  # takes its own write_txn
    rec = read_embedder(m)
    assert rec["embed_model"] == "test-embed"
    assert str(rec["embed_dim"]) == str(DIM)

    re = _open(p)
    rec2 = read_embedder(re)
    assert rec2["embed_model"] == "test-embed"
    assert re.get_vector(1) is None, "the provenance node carries no vector"
    assert re._content[1].has_vector is False


# ----------------------------------------------------------------- validation
@pytest.mark.parametrize("bad", [
    [],                      # zero-length
    np.zeros(DIM, np.float32),   # all-zero (no direction)
    [0.0] * DIM,
    np.zeros((2, 4), np.float32),  # wrong shape
    [1.0, 2.0],              # wrong length
])
def test_bad_vectors_are_rejected_loudly(tmp_path, bad):
    p = tmp_path / "z.mpmf"
    m = _open(p)
    node = _add(m, 1)
    with pytest.raises(ValueError):
        with m.write_txn():
            m.set_node_vector(node.node_id, bad)
    # the store is untouched and still readable
    np.testing.assert_allclose(m.get_vector(node.node_id), _vec(1), rtol=0, atol=1e-6)


def test_zero_dimension_store_cannot_store_a_vector():
    with pytest.raises(ValueError):
        VectorStore(0).append(1, [])


def test_nan_vector_is_rejected():
    vs = VectorStore(3)
    with pytest.raises(ValueError):
        vs.append(1, [1.0, float("nan"), 0.0])


def test_read_only_view_cannot_corrupt_the_matrix():
    vs = VectorStore(3)
    vs.append(1, [1.0, 0.0, 0.0])
    v = vs.get(1)
    with pytest.raises(ValueError):
        v[0] = 5.0
