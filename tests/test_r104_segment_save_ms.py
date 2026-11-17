"""r104: every save reports save_ms — including the append-only segment path.

The 72 h append-only soak's write-time leg read `save_ms` from the store log and found
0 samples since the store started appending segments, because (a) the r9x ingest log
line dropped the field and (b) nothing timed a segment save. Both are fixed: the store
records `last_save_ms`/`last_save_kind` on whichever path it takes, and ingest logs it.
"""
import numpy as np

from mpmf.memory import MemoryManager

DIM = 16


def _vec(seed):
    r = np.random.RandomState(seed)
    return r.rand(DIM).astype("<f4").tobytes()


def _mk(path, segments):
    m = MemoryManager(str(path), vector_dimensions=DIM, append_segments=segments)
    m.load()
    return m


def test_full_image_save_reports_ms(tmp_path):
    m = _mk(tmp_path / "full.mpmf", False)
    with m.write_txn():
        m.create_node("n1", 1, 1, text="t", summary="s", wiki={}, vector=_vec(1))
    assert m.last_save_kind == "full"
    assert m.last_save_ms > 0.0


def test_segment_save_reports_ms(tmp_path):
    m = _mk(tmp_path / "seg.mpmf", True)
    with m.write_txn():                      # first save on a new file is the base image
        n = m.create_node("n1", 1, 1, text="t", summary="s", wiki={}, vector=_vec(1))
    assert m.last_save_kind == "full"

    with m.write_txn():                      # second save appends
        m.create_node("n2", 1, 1, text="t2", summary="s2", wiki={}, vector=_vec(2))
    assert m.last_save_kind == "segment"
    assert m.last_save_ms > 0.0

    # a vector-only update also lands as a segment, and is still timed
    with m.write_txn():
        m.set_node_vector(n.node_id, _vec(9))
    assert m.last_save_kind == "segment"
    assert m.last_save_ms > 0.0


def test_noop_save_keeps_kind_and_reports_ms(tmp_path):
    m = _mk(tmp_path / "noop.mpmf", True)
    with m.write_txn():
        m.create_node("n1", 1, 1, text="t", summary="s", wiki={}, vector=_vec(1))
    with m.write_txn():
        pass
    assert m.last_save_kind == "noop"
    assert m.last_save_ms >= 0.0
