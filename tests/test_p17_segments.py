"""P17 — append-only segments: round trip, parity with the full image, torn tail,
backward compatibility, compaction, two-process concurrency."""
import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

from mpmf.memory import MemoryManager
from mpmf.format import Plane
from mpmf import compact as compact_mod

DIM = 16


def _vec(seed):
    r = np.random.RandomState(seed)
    return r.rand(DIM).astype("<f4").tobytes()


def _mk(path, on, **kw):
    m = MemoryManager(str(path), vector_dimensions=DIM, append_segments=on, **kw)
    m.load()
    return m


def _add(m, i):
    with m.write_txn():
        n = m.create_node("n%d" % i, 1, 1, text="text %d é" % i, summary="s%d" % i,
                          wiki={"k": "v%d" % i}, vector=_vec(i))
        if i > 0:
            m.create_edge(n.node_id, max(1, n.node_id - 1), Plane.ASSOCIATIVE, "rel%d" % i)


def _state(m):
    return (sorted(m._nodes),
            {i: (m.get_name(i), m.get_text(i), m.get_summary(i), m.get_wiki(i)) for i in m._nodes},
            sorted((e.edge_id, e.source_node_id, e.target_node_id, m.get_edge_label(e.edge_id))
                   for e in m._edges),
            {i: bytes(m.get_vector(i)) for i in m._nodes if m.has_vector(i)})


def test_flag_default_off(tmp_path, monkeypatch):
    monkeypatch.delenv("MPMF_APPEND_SEGMENTS", raising=False)
    monkeypatch.setenv("MPMF_HOME", str(tmp_path))
    m = MemoryManager(str(tmp_path / "s.mpmf"), vector_dimensions=DIM)
    assert m._segments_on is False
    m.load()
    _add(m, 1)
    assert m.last_save_kind == "full"


def test_off_is_legacy_bytes(tmp_path):
    """Flag OFF writes the pre-P17 layout (flags=0, reserved zeros, checksum to EOF)."""
    m = _mk(tmp_path / "a.mpmf", False)
    for i in range(3):
        _add(m, i)
    raw = (tmp_path / "a.mpmf").read_bytes()
    assert m.header.flags == 0 and bytes(m.header._reserved) == bytes(80)
    assert m.header.checksum == hashlib.sha256(raw[256:]).digest()


def test_segment_roundtrip_matches_full(tmp_path):
    a = _mk(tmp_path / "seg.mpmf", True)
    b = _mk(tmp_path / "full.mpmf", False)
    for i in range(12):
        _add(a, i)
        _add(b, i)
    assert a.last_save_kind == "segment"
    for m in (a, b):
        with m.write_txn():
            m.update_node_wiki(3, {"extra": "1"})
            m.set_node_text(4, "changed")
            m.set_node_vector(5, _vec(99))
            m.set_node_importance(6, 0.9)
    ra, rb = _mk(tmp_path / "seg.mpmf", True), _mk(tmp_path / "full.mpmf", False)
    assert _state(ra) == _state(rb) == _state(a)
    assert ra.get_node(6).importance == pytest.approx(0.9)
    assert ra._seg_count > 0


def test_segment_write_is_small(tmp_path):
    m = _mk(tmp_path / "s.mpmf", True)
    for i in range(50):
        _add(m, i)
    size = os.path.getsize(tmp_path / "s.mpmf")
    _add(m, 100)
    assert m.last_save_kind == "segment"
    assert m.last_save_bytes < 600 < size


def test_torn_tail_ignored_then_truncated(tmp_path):
    p = tmp_path / "s.mpmf"
    m = _mk(p, True)
    for i in range(5):
        _add(m, i)
    good = _state(m)
    with open(p, "ab") as f:
        f.write(b"MSG1\x40\x00\x00\x00\x00\x00\x00\x00partial")
    r = _mk(p, True)
    assert _state(r) == good and r._torn
    _add(r, 7)                      # next append truncates the torn bytes
    assert len(_mk(p, True)._nodes) == 6


def test_corrupt_base_still_refused(tmp_path):
    p = tmp_path / "s.mpmf"
    m = _mk(p, True)
    _add(m, 1)
    raw = bytearray(p.read_bytes())
    raw[300] ^= 0xFF
    p.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="Checksum"):
        _mk(p, True)


def test_legacy_file_read_and_upgraded_by_flag(tmp_path):
    p = tmp_path / "s.mpmf"
    m = _mk(p, False)
    for i in range(4):
        _add(m, i)
    on = _mk(p, True)                       # flag on over a legacy image
    assert _state(on) == _state(m)
    _add(on, 9)                             # first save writes a segment-capable base
    _add(on, 10)
    assert on.last_save_kind == "segment"
    assert len(_mk(p, False)._nodes) == 6   # flag OFF still reads segments


def test_flag_off_over_segment_file_rewrites_plain(tmp_path):
    p = tmp_path / "s.mpmf"
    m = _mk(p, True)
    for i in range(4):
        _add(m, i)
    off = _mk(p, False)
    _add(off, 8)
    assert off.last_save_kind == "full"
    assert off.header.flags == 0
    assert len(_mk(p, False)._nodes) == 5


def test_compaction(tmp_path):
    p = tmp_path / "s.mpmf"
    m = _mk(p, True)
    for i in range(20):
        _add(m, i)
    before = _state(_mk(p, True))
    rep = compact_mod.compact(str(p), dim=DIM)
    assert rep["segments"] == 19 and rep["parity"] == "ok"
    r = _mk(p, True)
    assert r._seg_count == 0 and _state(r) == before
    compact_mod.compact(str(p), dim=DIM, legacy=True)
    r = _mk(p, False)
    assert r.header.flags == 0 and _state(r) == before


def test_auto_compact_threshold(tmp_path):
    m = _mk(tmp_path / "s.mpmf", True, compact_after_segments=3)
    kinds = []
    for i in range(8):
        _add(m, i)
        kinds.append(m.last_save_kind)
    assert "full" in kinds[3:]              # rolled over into a fresh base
    assert len(_mk(tmp_path / "s.mpmf", True)._nodes) == 8


_WRITER = """
import sys, zlib
sys.path.insert(0, %r)
import numpy as np
from mpmf.memory import MemoryManager
p, tag, n, on = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1"
m = MemoryManager(p, vector_dimensions=16, append_segments=on); m.load()
for i in range(n):
    v = np.random.RandomState(zlib.crc32(('%%s%%d' %% (tag, i)).encode())).rand(16).astype('<f4').tobytes()
    with m.write_txn(timeout=60):
        m.create_node('c_%%s_%%d' %% (tag, i), 1, 1, text='x %%s %%d' %% (tag, i), vector=v)
"""


@pytest.mark.parametrize("on", ["1", "0"])
def test_two_process_writers_lose_nothing(tmp_path, on):
    p = tmp_path / "s.mpmf"
    m = _mk(p, on == "1")
    _add(m, 0)
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    code = _WRITER % src
    procs = [subprocess.Popen([sys.executable, "-c", code, str(p), t, "25", on]) for t in "AB"]
    assert [q.wait(120) for q in procs] == [0, 0]
    r = _mk(p, on == "1")
    names = {r.get_name(i) for i in r._nodes}
    assert len(r._nodes) == 51
    assert all(("c_%s_%d" % (t, i)) in names for t in "AB" for i in range(25))


def test_mixed_flag_writers_do_not_lose_nodes(tmp_path):
    p = tmp_path / "s.mpmf"
    _add(_mk(p, True), 0)          # the file must exist before two writers open it
    a, b = _mk(p, True), _mk(p, False)
    _add(a, 1)
    _add(b, 2)          # flag-off writer rewrites a plain image, folding A's segment
    _add(a, 3)          # A sees a rewritten base and reloads
    _add(b, 4)
    _add(a, 5)
    assert len(_mk(p, True)._nodes) == 6
