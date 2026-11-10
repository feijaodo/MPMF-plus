"""P17 — append-only write segments for the MPMF image.

Legacy behaviour (flag OFF, the default): every save serialises the WHOLE image
and renames it over the store, so a one-node write moves ~130 MB.

Segment mode (flag ON): the base image is written once; each later write
APPENDS one framed delta segment (the nodes / edges / vectors that changed) to
the end of the same file and fsyncs it. ``mpmf.compact`` folds the segments
back into a fresh base.

File layout of a segment-capable store::

    [256-byte header][base image ... string pool][seg][seg]...[seg]
                      ^ header.checksum covers [256:base_end] only

* header.flags bit 0 (``FLAG_SEGMENTS``) and header.reserved[0:12] =
  ``b"SEG1" + u64 base_end`` mark the file as segment-capable.
* A segment frame is ``b"MSG1" | u32 payload_len | u32 crc32(payload) | payload``.
  A torn tail (crash mid-append) fails the length/CRC check and is ignored by
  readers; the next writer truncates it under the lock.
* Segment records carry NO file offsets (the base image's offset fields are
  recomputed on every full serialise and are meaningless in a delta).

Compatibility: a file written with the flag OFF is byte-for-byte the legacy
layout. New code reads both. Code from before P17 refuses a file that carries
segments with ``Checksum mismatch`` (loud, never a silent stale read); run
``python -m mpmf.compact`` to turn it back into a legacy image.
"""

from __future__ import annotations

import struct
import zlib
from typing import List, Optional, Tuple

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

from .format import EDGE_SIZE, NODE_SIZE, EdgeRecord, NodeRecord

FLAG_SEGMENTS = 0x1
RES_MAGIC = b"SEG1"
SEG_MAGIC = b"MSG1"
_FRAME = struct.Struct("<4sII")
FRAME_SIZE = _FRAME.size            # 12
_COUNTS = struct.Struct("<IIII")
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")

NODE_HAS_VEC_ROW = 0x1      # a float32 row follows in this segment
NODE_HAS_VECTOR = 0x2       # content.has_vector


def pack_reserved(base_end: int) -> bytes:
    """The 80-byte header ``reserved`` field of a segment-capable file."""
    return RES_MAGIC + _U64.pack(int(base_end)) + bytes(80 - 12)


def read_base_end(header) -> Optional[int]:
    """base_end of a segment-capable header, or None for a legacy image."""
    if not (int(header.flags) & FLAG_SEGMENTS):
        return None
    res = bytes(header._reserved)
    if res[:4] != RES_MAGIC:
        return None
    return _U64.unpack_from(res, 4)[0]


def _lp(s: str) -> bytes:
    b = s.encode("utf-8")
    return _U32.pack(len(b)) + b


def _clean_node(rec: NodeRecord) -> bytes:
    n = NodeRecord.unpack(rec.pack())
    n.name_offset = n.vector_offset = n.summary_offset = 0
    n.wiki_offset = n.text_offset = n.metadata_offset = n.edge_list_offset = 0
    return n.pack()


def _clean_edge(rec: EdgeRecord) -> bytes:
    e = EdgeRecord.unpack(rec.pack())
    e.relation_type = 0
    e.metadata_offset = 0
    return e.pack()


def encode(nodes, deleted_nodes, edges, deleted_edges, dim: int) -> bytes:
    """Build one framed segment.

    nodes:  iterable of (NodeRecord, name, text, summary, wiki: dict,
            has_vector: bool, vec_bytes: bytes|None)
    edges:  iterable of (EdgeRecord, label)
    """
    nodes = list(nodes)
    edges = list(edges)
    deleted_nodes = sorted(deleted_nodes)
    deleted_edges = sorted(deleted_edges)
    out = bytearray(_COUNTS.pack(len(nodes), len(deleted_nodes),
                                 len(edges), len(deleted_edges)))
    for rec, name, text, summary, wiki, has_vector, vec in nodes:
        flags = (NODE_HAS_VECTOR if has_vector else 0) | \
                (NODE_HAS_VEC_ROW if vec is not None else 0)
        out += _clean_node(rec)
        out += bytes([flags])
        out += _lp(name) + _lp(text) + _lp(summary)
        out += _U32.pack(len(wiki))
        for k, v in wiki.items():
            out += _lp(str(k)) + _lp(str(v))
        if vec is not None:
            if len(vec) != dim * 4:
                raise ValueError("segment vector must be %d bytes, got %d"
                                 % (dim * 4, len(vec)))
            out += vec
    for nid in deleted_nodes:
        out += _U64.pack(nid)
    for rec, label in edges:
        out += _clean_edge(rec) + _lp(label or "")
    for eid in deleted_edges:
        out += _U64.pack(eid)
    payload = bytes(out)
    return _FRAME.pack(SEG_MAGIC, len(payload), zlib.crc32(payload)) + payload


def scan(data, start: int = 0) -> Tuple[List[memoryview], int, bool]:
    """Walk frames from ``start``.

    Returns (payload views, end offset of the last good frame, torn) where
    ``torn`` is True when bytes remain that do not form a valid frame.
    """
    mv = memoryview(data)
    n = len(mv)
    pos = start
    payloads: List[memoryview] = []
    while pos + FRAME_SIZE <= n:
        magic, length, crc = _FRAME.unpack_from(mv, pos)
        if magic != SEG_MAGIC:
            break
        end = pos + FRAME_SIZE + length
        if end > n:
            break
        payload = mv[pos + FRAME_SIZE:end]
        if zlib.crc32(payload) != crc:
            break
        payloads.append(payload)
        pos = end
    return payloads, pos, pos < n


def decode(payload, dim: int):
    """Inverse of encode(). Returns (nodes, deleted_nodes, edges, deleted_edges)."""
    mv = memoryview(payload)
    n_nodes, n_dn, n_edges, n_de = _COUNTS.unpack_from(mv, 0)
    pos = _COUNTS.size

    def lp():
        nonlocal pos
        ln = _U32.unpack_from(mv, pos)[0]
        pos += 4
        s = bytes(mv[pos:pos + ln]).decode("utf-8")
        pos += ln
        return s

    nodes = []
    for _ in range(n_nodes):
        rec = NodeRecord.unpack(bytes(mv[pos:pos + NODE_SIZE]))
        pos += NODE_SIZE
        flags = mv[pos]
        pos += 1
        name, text, summary = lp(), lp(), lp()
        n_wiki = _U32.unpack_from(mv, pos)[0]
        pos += 4
        wiki = {}
        for _ in range(n_wiki):
            k = lp()
            wiki[k] = lp()
        vec = None
        if flags & NODE_HAS_VEC_ROW:
            nbytes = dim * 4
            raw = bytes(mv[pos:pos + nbytes])
            if len(raw) != nbytes:
                raise ValueError("truncated vector in segment")
            vec = _np.frombuffer(raw, dtype="<f4", count=dim) if _np is not None \
                else list(struct.unpack("<%df" % dim, raw))
            pos += nbytes
        nodes.append((rec, name, text, summary, wiki,
                      bool(flags & NODE_HAS_VECTOR), vec))
    dnodes = []
    for _ in range(n_dn):
        dnodes.append(_U64.unpack_from(mv, pos)[0])
        pos += 8
    edges = []
    for _ in range(n_edges):
        rec = EdgeRecord.unpack(bytes(mv[pos:pos + EDGE_SIZE]))
        pos += EDGE_SIZE
        edges.append((rec, lp()))
    dedges = []
    for _ in range(n_de):
        dedges.append(_U64.unpack_from(mv, pos)[0])
        pos += 8
    if pos != len(mv):
        raise ValueError("segment payload has %d trailing bytes" % (len(mv) - pos))
    return nodes, dnodes, edges, dedges
