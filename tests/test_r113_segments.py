"""A damaged frame is evidence, not permission to discard the suffix."""
import logging
import struct

import pytest
from mpmf import compact, rollback, segments
from mpmf.memory import MemoryManager


def opened(p, on=True):
    m = MemoryManager(p, vector_dimensions=16, append_segments=on)
    m.load()
    return m


def append(m, name):
    with m.write_txn():
        m.create_node(name, 1, 1, text=name)


@pytest.fixture
def five(tmp_path):
    # Record boundaries independently via the on-disk sizes between real commits.
    p = tmp_path / 'five.mpmf'
    m = opened(p)
    append(m, 'base')
    offsets = [p.stat().st_size]
    for i in range(1, 6):
        append(m, str(i))
        assert m.last_save_kind == 'segment'
        offsets.append(p.stat().st_size)
    return p, offsets


def damage(p, pos, index=8):
    data = bytearray(p.read_bytes())
    data[pos + index] ^= 1
    p.write_bytes(data)
    return bytes(data)


@pytest.mark.parametrize('on', [True, False])
def test_bad_crc_preserves_all_later_frames(five, caplog, on):
    p, o = five
    original = p.read_bytes()
    damaged = damage(p, o[1])
    m = opened(p, on)
    assert set(m._name_index) == {'base', '1'}  # prefix is readable
    # Attempt the write even on the old implementation, so this proves the actual loss.
    try:
        with pytest.raises(segments.CorruptSegmentsError, match=f'corrupt at offset {o[1]}'):
            append(m, 'new')
    finally:
        assert p.read_bytes()[o[2]:] == original[o[2]:], 'frames 3–5 changed'
        assert p.read_bytes() == damaged
    assert any(r.levelno >= logging.ERROR and f'corrupt at offset {o[1]}' in r.message
               and 'crc' in r.message.lower() for r in caplog.records)
    assert segments.scan(damaged, o[0]).status == 'corrupt'
    assert segments.scan(damaged, o[0]).offset == o[1]


@pytest.mark.parametrize('operation', ['compact', 'rollback', 'save'])
def test_rewrite_refused(five, operation):
    p, o = five
    damaged = damage(p, o[1])
    try:
        with pytest.raises(segments.CorruptSegmentsError, match=f'corrupt at offset {o[1]}'):
            if operation == 'compact':
                compact.compact(str(p), dim=16)
            elif operation == 'rollback':
                rollback.compact_to_legacy(str(p), 16)
            else:
                opened(p, False).periodic_save()
    finally:
        assert p.read_bytes() == damaged


def test_last_partial_payload_is_the_only_thing_removed(five):
    p, o = five
    raw = p.read_bytes()[:-1]
    p.write_bytes(raw)
    m = opened(p)
    assert set(m._name_index) == {'base', '1', '2', '3', '4'}
    assert p.read_bytes() == raw
    append(m, 'replacement')
    assert p.read_bytes()[:o[4]] == raw[:o[4]]
    assert set(opened(p)._name_index) == {'base', '1', '2', '3', '4', 'replacement'}
    torn = segments.scan(raw, o[0])
    assert (torn.status, torn.offset) == ('incomplete_tail', o[4])
    clean = segments.scan(p.read_bytes(), o[0])
    assert (clean.status, clean.offset) == ('clean', p.stat().st_size)


@pytest.mark.parametrize('extra,status', [(b'MS','incomplete_tail'), (b'MSG1\x40','incomplete_tail'),
                                         (b'MX','corrupt'), (b'NOPE\x40','corrupt')])
def test_short_header(five, extra, status):
    p, o = five
    raw = p.read_bytes() + extra
    p.write_bytes(raw)
    result = segments.scan(raw, o[0])
    assert (result.status, result.offset) == (status, o[-1])
    if status == 'corrupt':
        with pytest.raises(segments.CorruptSegmentsError, match=f'corrupt at offset {o[-1]}'):
            append(opened(p), 'new')
        assert p.read_bytes() == raw
    else:
        append(opened(p), 'new')
        assert p.read_bytes()[:o[-1]] == raw[:o[-1]]


@pytest.mark.parametrize('field', ['magic','length'])
def test_bad_middle_header(five, field):
    p, o = five
    raw = bytearray(p.read_bytes())
    if field == 'magic':
        raw[o[1]] ^= 1
    else:
        # A broken length can imitate a short last payload, but verified frames follow.
        struct.pack_into('<I', raw, o[1]+4, len(raw)*2)
    p.write_bytes(raw)
    with pytest.raises(segments.CorruptSegmentsError, match=f'corrupt at offset {o[1]}'):
        append(opened(p), 'new')
    assert p.read_bytes() == raw


@pytest.mark.parametrize('growth', [b'', b'MS'])
def test_cached_writer_notices_in_place_crc_damage(five, growth):
    p, o = five
    cached = opened(p)
    st = p.stat()
    raw = damage(p, o[1]) + growth
    p.write_bytes(raw)
    # Timestamp granularity can hide a same-size edit. Force this branch rather
    # than depending on the filesystem clock ticking between fixture and fault.
    if not growth:
        import os
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    with pytest.raises(segments.CorruptSegmentsError, match=f'corrupt at offset {o[1]}'):
        append(cached, 'new')
    assert p.read_bytes() == raw


def malformed(p, offsets):
    # An encoder can emit a well-framed but undecodable delta. Preserve length,
    # recompute CRC, and leave all later frames byte-identical.
    import zlib
    start, end = offsets[1:3]
    raw = bytearray(p.read_bytes())
    payload = start + segments.FRAME_SIZE
    struct.pack_into('<4I', raw, payload, 0, 0, 0, 0)  # trailing node bytes
    struct.pack_into('<I', raw, start+8, zlib.crc32(raw[payload:end]))
    p.write_bytes(raw)
    return bytes(raw)


@pytest.mark.parametrize('entry', ['load', 'growth', 'rewrite'])
def test_crc_valid_decode_failure(five, entry, caplog):
    import os
    p, o = five
    cached = opened(p)
    st = p.stat()
    raw = malformed(p, o)
    if entry == 'growth':
        raw += b'MS'
        p.write_bytes(raw)
    if entry == 'rewrite':
        # No stat-key change: compact must reach its under-lock disk scan rather
        # than finding damage via reload or a pre-existing _segment_error.
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert cached._current_stat() == cached._stat_key and not cached._segment_error
        with pytest.raises(segments.CorruptSegmentsError, match=f'corrupt at offset {o[1]}: .*decode'):
            cached.compact()
    else:
        m = opened(p) if entry == 'load' else cached
        with pytest.raises(segments.CorruptSegmentsError, match=f'corrupt at offset {o[1]}: .*decode'):
            append(m, 'new')
        assert set(m._name_index) == {'base', '1'} and (m._seg_end, m._seg_count) == (o[1], 1)
    assert any(r.levelno >= logging.ERROR and f'corrupt at offset {o[1]}' in r.message
               and 'trailing bytes' in r.message for r in caplog.records)
    assert p.read_bytes() == raw


@pytest.mark.parametrize('op', ['append', 'compact', 'legacy', 'rollback'])
def test_final_length_damage_is_quarantined(five, op, caplog):
    p, o = five
    raw = bytearray(p.read_bytes())
    struct.pack_into('<I', raw, o[-2]+4, len(raw)*2)
    p.write_bytes(raw)
    if op == 'append':
        append(opened(p), 'new')
    elif op == 'compact':
        opened(p).compact()
    elif op == 'rollback':
        rollback.compact_to_legacy(str(p), 16)
    else:
        opened(p, False).periodic_save()
    q, = p.parent.glob(p.name + '.tail-*.quarantine')
    assert q.read_bytes() == raw[o[-2]:]
    assert any(str(q) in r.message and r.levelno >= logging.WARNING for r in caplog.records)
    assert set(opened(p)._name_index) == {'base', '1', '2', '3', '4'} | ({'new'} if op == 'append' else set())


def test_quarantine_fsync_failure_preserves_source(five, monkeypatch):
    import os
    p, _ = five
    raw = p.read_bytes()[:-1]
    p.write_bytes(raw)
    m = opened(p)
    def failed(_fd):
        raise OSError('simulated fsync failure')
    monkeypatch.setattr(os, 'fsync', failed)
    with pytest.raises(OSError, match='fsync'):
        append(m, 'must-not-land')
    assert p.read_bytes() == raw


@pytest.mark.parametrize('fault', ['shrink', 'stat'])
def test_stale_during_transaction_never_rewrites(five, fault, monkeypatch):
    import os
    from mpmf.memory import StaleImageError
    p, o = five
    m = opened(p)
    raw = p.read_bytes()
    real_stat = os.stat
    def unavailable(path, *args, **kwargs):
        if path == str(p):
            raise OSError('cannot stat test image')
        return real_stat(path, *args, **kwargs)
    with pytest.raises(StaleImageError):
        with m.write_txn():
            m.create_node('stale', 1, 1, text='stale')
            if fault == 'shrink':
                raw = raw[:o[-2]]  # clean boundary; even a full frame scan would pass
                p.write_bytes(raw)
            else:
                monkeypatch.setattr(os, 'stat', unavailable)
    assert p.read_bytes() == raw


def test_recovery_inventory_and_copy(five):
    from mpmf.recover_segments import inspect

    p, offsets = five
    raw = damage(p, offsets[1])
    files = set(p.parent.iterdir())
    report = inspect(p)
    assert report['status'] == 'corrupt' and report['offset'] == offsets[1]
    assert report['frames_before'] == [[offsets[0], offsets[1]]]
    assert report['frames_after'] == [list(pair) for pair in zip(offsets[2:], offsets[3:])]
    assert set(p.parent.iterdir()) == files and p.read_bytes() == raw
    copy = p.parent / 'evidence.bin'
    assert inspect(p, copy)['sha256'] == report['sha256']
    assert p.read_bytes() == copy.read_bytes() == raw
    for existing in (p, copy):
        with pytest.raises(FileExistsError):
            inspect(p, existing)
    assert p.read_bytes() == copy.read_bytes() == raw
