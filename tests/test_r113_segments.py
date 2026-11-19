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
        with pytest.raises(ValueError, match=f'corrupt at offset {o[1]}'):
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
        with pytest.raises(ValueError, match=f'corrupt at offset {o[1]}'):
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
        with pytest.raises(ValueError, match=f'corrupt at offset {o[-1]}'):
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
    with pytest.raises(ValueError, match=f'corrupt at offset {o[1]}'):
        append(opened(p), 'new')
    assert p.read_bytes() == raw


def test_cached_writer_notices_in_place_crc_damage(five):
    p, o = five
    cached = opened(p)
    raw = damage(p, o[1])
    with pytest.raises(ValueError, match=f'corrupt at offset {o[1]}'):
        append(cached, 'new')
    assert p.read_bytes() == raw
