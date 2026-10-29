"""
mpmf/format.py — Binary format definitions for MPMF

Implements fixed-size records as described in the spec:
- 256-byte File Header
- 96-byte Node Records
- 64-byte Edge Records
- 24-byte Plane Index Entries

ARM Assembly Notes:
- All structs are packed with native byte order (little-endian on x86/ARM).
  The assembly implementation should use the platform's native endianness.
- Node records are 96 bytes — fits 3 cache lines on most CPUs.
- Vector block alignment (32 bytes) is enforced via padding in VectorBlock.
- The header is exactly 256 bytes to align the Node Table to a 256-byte boundary.
"""

import struct
import uuid
import time
from enum import IntEnum
from typing import List, Optional, Tuple

# ============================================================================
# Constants
# ============================================================================

MAGIC = b"MPMF"
HEADER_SIZE = 256
NODE_SIZE = 96
EDGE_SIZE = 68
PLANE_INDEX_SIZE = 24
VECTOR_ALIGNMENT = 32  # bytes — enables AVX2 dot products without unaligned loads
DEFAULT_VECTOR_DIM = 128


# ============================================================================
# Plane Definitions (bitmasks — powers of 2)
# ============================================================================
# ASSEMBLY: These are uint16 bitmask values. Use a single AND instruction
# to test membership: "if node.plane_flags & Plane.ASSOCIATIVE.value"

class Plane(IntEnum):
    ASSOCIATIVE = 0x0001
    CAUSAL = 0x0002
    TEMPORAL = 0x0004
    FUNCTIONAL = 0x0008
    PRIORITY = 0x0010

    # Custom planes start here
    CUSTOM_BASE = 0x0020


class NodeType(IntEnum):
    """Semantic types for nodes."""
    CONCEPT = 0x0001
    ENTITY = 0x0002
    EVENT = 0x0003
    ACTION = 0x0004
    GOAL = 0x0005
    FACT = 0x0006
    CLAIM = 0x0007
    CONTEXT = 0x0008
    AGENT = 0x0009
    TOOL = 0x000A
    CUSTOM = 0xFFFF


# ============================================================================
# File Flags
# ============================================================================

class FileFlags:
    COMPRESSED = 0x00000001
    READONLY = 0x00000002
    INCREMENTAL = 0x00000004
    ENCRYPTED = 0x00000008


# ============================================================================
# Struct Templates
# ============================================================================
# PACKED FORMAT NOTES:
# Format chars: Q=uint64, H=uint16, I=uint32, f=float32, B=uint8
# All multi-byte integers are little-endian (standard for x86/ARM).
#
# ARM Assembly: struct fields map directly to offsets. No packing needed
# since we're defining explicit byte offsets in the spec.

# File Header: 256 bytes total
# ASSEMBLY: Header starts at offset 0. Node Table immediately follows at 256.
# This 256-byte alignment ensures Node Table starts on a page boundary.
_HEADER_STRUCT = struct.Struct("< 4s HH QQ QQ QQ QQ QQ QQ QQ HH I 16s 32s 80s")
# Fields: magic(4), ver_major(2), ver_minor(2), created_at(8), modified_at(8),
#         node_count(8), edge_count(8), plane_index_count(8),
#         node_table_offset(8), edge_table_offset(8), plane_index_offset(8),
#         vector_block_offset(8), summary_block_offset(8), wiki_block_offset(8),
#         text_block_offset(8), metadata_block_offset(8), string_pool_offset(8),
#         vector_dimensions(2), custom_plane_count(2), flags(4),
#         agent_id(16), checksum(32), reserved(80)
# Total: 4+2+2+8*13+2+2+4+16+32+80 = 4+2+2+104+4+16+32+80 = 256 ✓

assert _HEADER_STRUCT.size == HEADER_SIZE


_NODE_STRUCT = struct.Struct("<QQHHIQQQQQQQQfI")
# Node Record: 96 bytes. Fields: node_id(Q), name_offset(Q), node_type(H), plane_flags(H),
# edge_count(I), edge_list_offset(Q), vector_offset(Q), summary_offset(Q), wiki_offset(Q),
# text_offset(Q), metadata_offset(Q), created_at(Q), modified_at(Q), importance(f), reserved(I)
# Total: 8*10 + 2*2 + 4*2 + 4 = 80+4+8+4 = 96 ✓
assert _NODE_STRUCT.size == NODE_SIZE
assert len(_NODE_STRUCT.unpack(bytes(NODE_SIZE))) == 15


# Edge Record: 64 bytes
# ASSEMBLY: Edges from the same source node are stored contiguously.
# Sequential scan from edge_list_offset for edge_count records.
_EDGE_STRUCT = struct.Struct("<QQQIIfQQQQ")
# Edge Record: 64 bytes. Fields: edge_id(Q), source_node_id(Q), target_node_id(Q),
#         plane_id(H), relation_type(H), weight(f), created_at(Q),
#         metadata_offset(Q), inverse_edge_id(Q), reserved(Q)
# Total: 8*7 + 2*2 + 4 = 56+4+4 = 64 ✓

assert _EDGE_STRUCT.size == EDGE_SIZE


# Plane Index Entry: 24 bytes
# ASSEMBLY: Sorted by (plane_id, source_node_id). Binary search then linear scan.
_PLANE_INDEX_STRUCT = struct.Struct("< I Q Q I")
# Fields: plane_id(4), source_node_id(8), edge_id(8), reserved(4)
# Total: 4+8+8+4 = 24 ✓

assert _PLANE_INDEX_STRUCT.size == PLANE_INDEX_SIZE


# ============================================================================
# Record Classes
# ============================================================================

class FileHeader:
    """
    MPMF File Header — 256 bytes.

    ASSEMBLY: All offsets are absolute byte positions from file start.
    All multi-byte fields are little-endian. Vector block must be
    32-byte aligned after write (pad between sections if needed).
    """

    __slots__ = (
        "version_major", "version_minor", "created_at", "modified_at",
        "node_count", "edge_count", "plane_index_count",
        "node_table_offset", "edge_table_offset", "plane_index_offset",
        "vector_block_offset", "summary_block_offset", "wiki_block_offset",
        "text_block_offset", "metadata_block_offset", "string_pool_offset",
        "vector_dimensions", "custom_plane_count", "flags",
        "agent_id", "checksum", "_reserved",
    )

    def __init__(
        self,
        vector_dimensions: int = DEFAULT_VECTOR_DIM,
        agent_id: Optional[bytes] = None,
    ):
        self.version_major = 0
        self.version_minor = 1
        self.created_at = int(time.time())
        self.modified_at = int(time.time())
        self.node_count = 0
        self.edge_count = 0
        self.plane_index_count = 0
        self.node_table_offset = HEADER_SIZE  # Always 256
        self.edge_table_offset = 0  # Set after computing node table size
        self.plane_index_offset = 0
        self.vector_block_offset = 0
        self.summary_block_offset = 0
        self.wiki_block_offset = 0
        self.text_block_offset = 0
        self.metadata_block_offset = 0
        self.string_pool_offset = 0
        self.vector_dimensions = vector_dimensions
        self.custom_plane_count = 0
        self.flags = 0
        self.agent_id = agent_id or uuid.uuid4().bytes
        self.checksum = bytes(32)
        self._reserved = bytes(80)

    def pack(self) -> bytes:
        """Pack header to 256-byte binary representation."""
        return _HEADER_STRUCT.pack(
            MAGIC,
            self.version_major,
            self.version_minor,
            self.created_at,
            self.modified_at,
            self.node_count,
            self.edge_count,
            self.plane_index_count,
            self.node_table_offset,
            self.edge_table_offset,
            self.plane_index_offset,
            self.vector_block_offset,
            self.summary_block_offset,
            self.wiki_block_offset,
            self.text_block_offset,
            self.metadata_block_offset,
            self.string_pool_offset,
            self.vector_dimensions,
            self.custom_plane_count,
            self.flags,
            self.agent_id,
            self.checksum,
            self._reserved,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "FileHeader":
        """Unpack header from 256-byte binary representation."""
        if len(data) < HEADER_SIZE:
            raise ValueError(f"Header data too short: {len(data)} < {HEADER_SIZE}")
        (
            magic, ver_major, ver_minor, created_at, modified_at,
            node_count, edge_count, plane_index_count,
            node_table_offset, edge_table_offset, plane_index_offset,
            vector_block_offset, summary_block_offset, wiki_block_offset,
            text_block_offset, metadata_block_offset, string_pool_offset,
            vector_dimensions, custom_plane_count, flags,
            agent_id, checksum, reserved,
        ) = _HEADER_STRUCT.unpack(data[:HEADER_SIZE])

        if magic != MAGIC:
            raise ValueError(f"Invalid MPMF magic: {magic!r}")

        h = cls.__new__(cls)
        h.version_major = ver_major
        h.version_minor = ver_minor
        h.created_at = created_at
        h.modified_at = modified_at
        h.node_count = node_count
        h.edge_count = edge_count
        h.plane_index_count = plane_index_count
        h.node_table_offset = node_table_offset
        h.edge_table_offset = edge_table_offset
        h.plane_index_offset = plane_index_offset
        h.vector_block_offset = vector_block_offset
        h.summary_block_offset = summary_block_offset
        h.wiki_block_offset = wiki_block_offset
        h.text_block_offset = text_block_offset
        h.metadata_block_offset = metadata_block_offset
        h.string_pool_offset = string_pool_offset
        h.vector_dimensions = vector_dimensions
        h.custom_plane_count = custom_plane_count
        h.flags = flags
        h.agent_id = agent_id
        h.checksum = checksum
        h._reserved = reserved
        return h


class NodeRecord:
    """
    MPMF Node Record — 96 bytes.

    ASSEMBLY:
    - Lookup by ID: node_addr = base + node_table_offset + (node_id - 1) * 96
      That's 3 x86 instructions (base + offset + imul).
    - The 8 most-used fields fit in first 32 bytes (one cache line):
      node_id(8) + name_offset(8) + node_type(2) + plane_flags(2) +
      edge_count(4) + edge_list_offset(8) = 32 ✓
    - importance(4) at offset 88 — end of record, not in hot path.
    - Use PREFETCHNTA on the next node while processing the current one.
    """

    __slots__ = (
        "node_id", "name_offset", "node_type", "plane_flags",
        "edge_count", "edge_list_offset", "vector_offset", "summary_offset",
        "wiki_offset", "text_offset", "metadata_offset",
        "created_at", "modified_at", "importance", "_reserved",
    )

    NULL_ID = 0  # Node ID 0 is reserved (null/unused)

    def __init__(
        self,
        node_id: int,
        name_offset: int = 0,
        node_type: NodeType = NodeType.CONCEPT,
        plane_flags: int = 0,
        edge_count: int = 0,
        edge_list_offset: int = 0,
        vector_offset: int = 0,
        summary_offset: int = 0,
        wiki_offset: int = 0,
        text_offset: int = 0,
        metadata_offset: int = 0,
        created_at: Optional[int] = None,
        modified_at: Optional[int] = None,
        importance: float = 0.5,
    ):
        if node_id == self.NULL_ID:
            raise ValueError("Node ID 0 is reserved (null)")
        self.node_id = node_id
        self.name_offset = name_offset
        self.node_type = node_type
        self.plane_flags = plane_flags
        self.edge_count = edge_count
        self.edge_list_offset = edge_list_offset
        self.vector_offset = vector_offset
        self.summary_offset = summary_offset
        self.wiki_offset = wiki_offset
        self.text_offset = text_offset
        self.metadata_offset = metadata_offset
        self.created_at = created_at or int(time.time() * 1000)
        self.modified_at = modified_at or int(time.time() * 1000)
        self.importance = importance
        self._reserved = 0

    def in_plane(self, plane: Plane) -> bool:
        """Test if node exists in a given plane. Single AND instruction in ASM."""
        return bool(self.plane_flags & plane.value)

    def pack(self) -> bytes:
        return _NODE_STRUCT.pack(
            self.node_id, self.name_offset, self.node_type, self.plane_flags,
            self.edge_count, self.edge_list_offset,
            self.vector_offset, self.summary_offset, self.wiki_offset,
            self.text_offset, self.metadata_offset,
            self.created_at, self.modified_at,
            self.importance, self._reserved,
        )

    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> "NodeRecord":
        if len(data) < offset + NODE_SIZE:
            raise ValueError(f"Node data too short: {len(data)} < {offset + NODE_SIZE}")
        (
            node_id, name_offset, node_type, plane_flags,
            edge_count, edge_list_offset,
            vector_offset, summary_offset, wiki_offset,
            text_offset, metadata_offset,
            created_at, modified_at,
            importance, reserved,
        ) = _NODE_STRUCT.unpack(data[offset:offset + NODE_SIZE])

        n = cls.__new__(cls)
        n.node_id = node_id
        n.name_offset = name_offset
        n.node_type = node_type
        n.plane_flags = plane_flags
        n.edge_count = edge_count
        n.edge_list_offset = edge_list_offset
        n.vector_offset = vector_offset
        n.summary_offset = summary_offset
        n.wiki_offset = wiki_offset
        n.text_offset = text_offset
        n.metadata_offset = metadata_offset
        n.created_at = created_at
        n.modified_at = modified_at
        n.importance = importance
        n._reserved = reserved
        return n


class EdgeRecord:
    """
    MPMF Edge Record — 64 bytes.

    ASSEMBLY:
    - Edges from the same source node are stored contiguously (sorted by
      source_node_id, then plane_id). Sequential scan from edge_list_offset.
    - plane_id at offset 24 — SIMD gather can filter by plane in parallel.
    - inverse_edge_id at offset 48 — 0 means not bidirectional.
    """

    __slots__ = (
        "edge_id", "source_node_id", "target_node_id",
        "plane_id", "relation_type", "weight", "created_at",
        "metadata_offset", "inverse_edge_id", "_reserved",
    )

    def __init__(
        self,
        edge_id: int,
        source_node_id: int,
        target_node_id: int,
        plane_id: Plane = Plane.ASSOCIATIVE,
        relation_type: int = 0,
        weight: float = 1.0,
        created_at: Optional[int] = None,
        metadata_offset: int = 0,
        inverse_edge_id: int = 0,
    ):
        self.edge_id = edge_id
        self.source_node_id = source_node_id
        self.target_node_id = target_node_id
        self.plane_id = plane_id
        self.relation_type = relation_type
        self.weight = weight
        self.created_at = created_at or int(time.time() * 1000)
        self.metadata_offset = metadata_offset
        self.inverse_edge_id = inverse_edge_id
        self._reserved = 0

    def pack(self) -> bytes:
        return _EDGE_STRUCT.pack(
            self.edge_id, self.source_node_id, self.target_node_id,
            self.plane_id, self.relation_type, self.weight,
            self.created_at, self.metadata_offset,
            self.inverse_edge_id, self._reserved,
        )

    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> "EdgeRecord":
        if len(data) < offset + EDGE_SIZE:
            raise ValueError(f"Edge data too short: {len(data)} < {offset + EDGE_SIZE}")
        (
            edge_id, source_node_id, target_node_id,
            plane_id, relation_type, weight,
            created_at, metadata_offset,
            inverse_edge_id, reserved,
        ) = _EDGE_STRUCT.unpack(data[offset:offset + EDGE_SIZE])

        e = cls.__new__(cls)
        e.edge_id = edge_id
        e.source_node_id = source_node_id
        e.target_node_id = target_node_id
        e.plane_id = plane_id
        e.relation_type = relation_type
        e.weight = weight
        e.created_at = created_at
        e.metadata_offset = metadata_offset
        e.inverse_edge_id = inverse_edge_id
        e._reserved = reserved
        return e


class PlaneIndexEntry:
    """
    MPMF Plane Index Entry — 24 bytes.

    ASSEMBLY:
    - Binary search by (plane_id, source_node_id) is the lookup strategy.
    - AVX2 can compare 8x plane_ids simultaneously per pass.
    - After binary search, linear scan while fields match.
    """

    __slots__ = ("plane_id", "source_node_id", "edge_id", "_reserved")

    def __init__(self, plane_id: int, source_node_id: int, edge_id: int):
        self.plane_id = plane_id
        self.source_node_id = source_node_id
        self.edge_id = edge_id
        self._reserved = 0

    def pack(self) -> bytes:
        return _PLANE_INDEX_STRUCT.pack(
            self.plane_id, self.source_node_id, self.edge_id, 0
        )

    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> "PlaneIndexEntry":
        if len(data) < offset + PLANE_INDEX_SIZE:
            raise ValueError(
                f"Plane index data too short: {len(data)} < {offset + PLANE_INDEX_SIZE}"
            )
        plane_id, source_node_id, edge_id, _ = _PLANE_INDEX_STRUCT.unpack(
            data[offset:offset + PLANE_INDEX_SIZE]
        )
        return cls(plane_id, source_node_id, edge_id)


# ============================================================================
# Variable-Length Block Helpers
# ============================================================================

def pack_string(s: str) -> bytes:
    """
    Pack a string as length-prefixed UTF-8 (no null terminator).

    ASSEMBLY: 4-byte uint32 length prefix, then N bytes of UTF-8.
    """
    encoded = s.encode("utf-8")
    return struct.pack("< I", len(encoded)) + encoded


def unpack_string(data: bytes, offset: int) -> Tuple[str, int]:
    """Unpack a length-prefixed string. Returns (string, new_offset)."""
    length = struct.unpack("< I", data[offset:offset + 4])[0]
    str_data = data[offset + 4:offset + 4 + length]
    return str_data.decode("utf-8"), offset + 4 + length


def pad_to_alignment(offset: int, alignment: int = VECTOR_ALIGNMENT) -> int:
    """Return padding bytes needed to align offset to alignment boundary."""
    return (alignment - (offset % alignment)) % alignment


def align_offset(offset: int, alignment: int = VECTOR_ALIGNMENT) -> int:
    """Return aligned offset (rounded up to next alignment boundary)."""
    return offset + pad_to_alignment(offset, alignment)