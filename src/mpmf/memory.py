"""
mpmf/memory.py — Memory Manager for MPMF (complete storage layer)

Implements the full lifecycle described in the spec:
- On Load: verify magic + checksum, parse header, parse all sections into an
  in-memory working set (nodes, edges, and their attached content).
- During operation: create_node / create_edge mutate the in-memory working set.
- On Save: serialize the entire working set to a byte buffer with a correct,
  deduplicated String Pool and all variable blocks (vector / summary / wiki /
  text / metadata), then write atomically (temp file -> fsync -> rename).

Unlike the original skeleton, this version actually PERSISTS node content
(text, summary, wiki, vector) and rebuilds the name index from the String Pool
on load, so memories survive a save/reload cycle.

The on-disk layout is fully spec-compliant: any reader following SPEC v0.1
can mmap() the file and access fixed records by arithmetic offset.
"""

import os
import struct
import hashlib
import shutil
import tempfile
import time
import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:  # numpy: zero-copy float32 row views. Optional, like the retrieval layer.
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

from .vectors import VectorStore
from . import segments as _seg

try:  # POSIX advisory locks; the store is single-host by design.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None


class StaleImageError(RuntimeError):
    """Raised when a save would overwrite a file this instance no longer matches.

    The store is a whole-file image (periodic_save() serialises everything and
    os.replace()s it). A writer that loaded at T0 and saves at T2 silently
    destroys anything written between T0 and T2. This error makes that loud.
    """


class LockTimeoutError(RuntimeError):
    """Raised when the cross-process store lock cannot be taken in time."""

from .format import (
    FileHeader,
    NodeRecord,
    EdgeRecord,
    PlaneIndexEntry,
    Plane,
    NodeType,
    MAGIC,
    HEADER_SIZE,
    NODE_SIZE,
    EDGE_SIZE,
    PLANE_INDEX_SIZE,
    align_offset,
    DEFAULT_VECTOR_DIM,
)

# CAS version key — persisted in each node's wiki block as a monotonic counter.
_VERSION_KEY = "_ver"


# ============================================================================
# String Pool (deduplicating, length-prefixed UTF-8)
# ============================================================================

class StringPool:
    """
    Append-only, deduplicating string pool.

    Layout per entry: [uint32 length][N bytes UTF-8].
    Offset 0 is reserved to mean "no string" (a node/edge with name_offset==0
    has no interned name), so the pool always begins with a single empty
    string sentinel at offset 0.
    """

    def __init__(self):
        self._buf = bytearray()
        self._offsets: Dict[str, int] = {}
        # Reserve offset 0 as the empty-string sentinel.
        self.intern("")

    def intern(self, s: str) -> int:
        if s in self._offsets:
            return self._offsets[s]
        offset = len(self._buf)
        encoded = s.encode("utf-8")
        self._buf.extend(struct.pack("<I", len(encoded)))
        self._buf.extend(encoded)
        self._offsets[s] = offset
        return offset

    def bytes(self) -> bytes:
        return bytes(self._buf)

    @staticmethod
    def read(data: bytes, offset: int) -> str:
        """Read a single interned string from a packed pool at `offset`."""
        if offset < 0 or offset + 4 > len(data):
            return ""
        length = struct.unpack_from("<I", data, offset)[0]
        start = offset + 4
        return data[start:start + length].decode("utf-8")

    @staticmethod
    def iter_all(data: bytes):
        """Yield (offset, string) for every entry in a packed pool."""
        pos = 0
        n = len(data)
        while pos + 4 <= n:
            length = struct.unpack_from("<I", data, pos)[0]
            start = pos + 4
            s = data[start:start + length].decode("utf-8")
            yield pos, s
            pos = start + length


# ============================================================================
# In-memory content holder for a node
# ============================================================================

class _NodeContent:
    """Holds the variable-length content attached to a node while in memory."""
    # Round 77: no `vector` slot. Vectors live in MemoryManager._vectors (one
    # float32 matrix); a node only carries a boolean "has a vector" flag.
    __slots__ = ("name", "text", "summary", "wiki", "has_vector", "relation_labels")

    def __init__(self, name="", text="", summary="", wiki=None, has_vector=False):
        self.name = name
        self.text = text
        self.summary = summary
        self.wiki: Dict[str, str] = wiki or {}
        self.has_vector = bool(has_vector)


# ============================================================================
# Memory Manager
# ============================================================================

DEFAULT_COMPACT_AFTER = 5000


def _append_segments_default() -> bool:
    """P17 flag: config key / env ``append_segments`` (MPMF_APPEND_SEGMENTS). Default OFF."""
    try:
        from mpmf_plus import paths as _paths
    except ImportError:   # core used without the plus layer: env only
        return os.environ.get("MPMF_APPEND_SEGMENTS", "").strip().lower() in (
            "1", "true", "yes", "on")
    v = _paths.load_config().get("append_segments")
    if isinstance(v, str):      # env override that is not JSON, e.g. MPMF_APPEND_SEGMENTS=on
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _compact_after_default() -> int:
    try:
        from mpmf_plus import paths as _paths
        n = int(_paths.load_config().get("compact_after_segments") or DEFAULT_COMPACT_AFTER)
    except Exception:
        n = DEFAULT_COMPACT_AFTER
    return n if n > 0 else DEFAULT_COMPACT_AFTER


class MemoryManager:
    """
    MPMF Memory Manager — full read/write implementation.

    In-memory model:
      _nodes:   node_id -> NodeRecord       (fixed record fields)
      _content: node_id -> _NodeContent     (attached text/summary/wiki/vector/name)
      _edges:   list[EdgeRecord]
      _edge_labels: edge_id -> relation label str
      _name_index: name -> node_id
    """

    def __init__(self, path: str, vector_dimensions: int = DEFAULT_VECTOR_DIM,
                 append_segments: Optional[bool] = None,
                 compact_after_segments: Optional[int] = None):
        self.path = Path(path)
        self.vector_dimensions = vector_dimensions
        self.header: Optional[FileHeader] = None

        self._nodes: Dict[int, NodeRecord] = {}
        self._content: Dict[int, _NodeContent] = {}
        self._edges: List[EdgeRecord] = []
        self._edge_labels: Dict[int, str] = {}
        self._name_index: Dict[str, int] = {}
        # Round 77: ONE float32 matrix + node_id -> row, shared by every reader.
        self._vectors = VectorStore(self.vector_dimensions)

        self._next_node_id = 1
        self._next_edge_id = 1

        # --- Round 48c: cross-process write safety -------------------------
        # (mtime_ns, size) of the image this instance is holding. Set on load
        # and after every save; used to detect that another writer has
        # replaced the file underneath us.
        self._stat_key: Optional[Tuple[int, int, int]] = None
        self._in_txn: bool = False

        # --- P17: append-only segments (default OFF) -----------------------
        self._segments_on: bool = (_append_segments_default()
                                   if append_segments is None else bool(append_segments))
        self._compact_after = int(_compact_after_default()
                                  if compact_after_segments is None else compact_after_segments)
        self._seg_base_end: Optional[int] = None   # None => file is a legacy image
        self._seg_end: int = 0                     # end of the last good frame
        self._seg_count: int = 0
        self._node_sigs: Optional[Dict[int, int]] = None
        self._edge_sigs: Optional[Dict[int, int]] = None
        self._dirty_vec: set = set()
        self.last_save_kind: str = ""              # "full" | "segment" | "noop"
        self.last_save_bytes: int = 0

    # ------------------------------------------------------------------------
    # Round 48c: write transactions
    # ------------------------------------------------------------------------

    def _current_stat(self) -> Optional[Tuple[int, int, int]]:
        try:
            st = os.stat(str(self.path))
            # P17: the inode joins the key. Full saves rename a new file into
            # place (new inode); appends change the size. Two saves that land in
            # the same coarse mtime tick with the same size used to look identical.
            return (st.st_mtime_ns, st.st_size, st.st_ino)
        except OSError:
            return None

    def _lock_path(self) -> str:
        return str(self.path) + ".lock"

    def _acquire_lock(self, timeout: float = 10.0) -> Optional[int]:
        """Take an exclusive cross-process lock, or raise LockTimeoutError."""
        if fcntl is None:
            return None
        fd = os.open(self._lock_path(), os.O_CREAT | os.O_RDWR, 0o644)
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    try:
                        os.close(fd)
                    except Exception:
                        pass
                    raise LockTimeoutError(
                        "store lock %s not acquired within %.1fs"
                        % (self._lock_path(), timeout))
                time.sleep(0.02)

    def _release_lock(self, fd: Optional[int]) -> None:
        if fd is None or fcntl is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            os.close(fd)
        except Exception:
            pass

    def _reload_under_lock(self) -> None:
        """Re-read the image from disk, discarding this instance's state."""
        self.header = None
        self._nodes = {}
        self._content = {}
        self._edges = []
        self._edge_labels = {}
        self._name_index = {}
        self._next_node_id = 1
        self._next_edge_id = 1
        self.load()

    def _refresh_under_lock(self) -> None:
        """Bring this instance level with the file: apply only the segments
        another writer appended (P17) if the base image is unchanged, else a
        full reload. Caller holds the lock."""
        if not self._refresh_incremental():
            self._reload_under_lock()

    @contextlib.contextmanager
    def write_txn(self, timeout: float = 10.0):
        """Exclusive write transaction: lock -> reload if stale -> mutate -> save.

        The mutation must happen INSIDE the with-block. On entry the instance is
        guaranteed to match the on-disk image, so a save cannot clobber a
        concurrent writer's nodes.
        """
        fd = self._acquire_lock(timeout)
        try:
            if self._stat_key is not None and self._current_stat() != self._stat_key:
                self._refresh_under_lock()
            self._in_txn = True
            try:
                yield self
            finally:
                self._in_txn = False
            self._save_locked()
        finally:
            self._release_lock(fd)

    # ------------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------------

    def load(self) -> None:
        self._stat_key = self._current_stat()
        self._reset_segment_state()
        if not self.path.exists():
            self._create_new()
            return

        with open(self.path, "rb") as f:
            data = f.read()

        if len(data) < HEADER_SIZE:
            raise ValueError(f"File too short: {len(data)} < {HEADER_SIZE}")

        self.header = FileHeader.unpack(data)
        if self.header.vector_dimensions and self.header.vector_dimensions != self.vector_dimensions:
            # r83b: this used to be a silent assignment, which is how a config
            # saying 1024 kept writing 1536-d vectors into Ayu's store (the
            # header won, nobody was told). The file is the truth, but the
            # mismatch is always a bug: either the config or the store is wrong.
            if os.environ.get("MPMF_ALLOW_DIM_MISMATCH") != "1":
                raise ValueError(
                    "vector dimension mismatch: %s has %d-d vectors but this "
                    "process was configured for %d-d (config is the only source; "
                    "the header is not allowed to override it silently). Run the "
                    "reembed migration to convert the store, or fix the config. "
                    "Set MPMF_ALLOW_DIM_MISMATCH=1 to open read-only anyway."
                    % (self.path, self.header.vector_dimensions, self.vector_dimensions))
            import logging
            logging.getLogger("mpmf").warning(
                "dim mismatch on %s: header=%d config=%d (allowed by MPMF_ALLOW_DIM_MISMATCH)",
                self.path, self.header.vector_dimensions, self.vector_dimensions)
        elif self.header.vector_dimensions:
            self.vector_dimensions = self.header.vector_dimensions

        # Verify checksum (computed over everything after byte 256).
        stored = self.header.checksum
        base_end = _seg.read_base_end(self.header)
        if base_end is not None and not (HEADER_SIZE <= base_end <= len(data)):
            raise ValueError("segment base_end out of range — file corrupted")
        # P17: with segments the checksum covers the base image only; a legacy
        # image (base_end None) is checksummed to EOF exactly as before.
        computed = hashlib.sha256(memoryview(data)[HEADER_SIZE:base_end]).digest()
        if stored != computed:
            raise ValueError("Checksum mismatch — file corrupted")

        self._parse(data, base_end)
        if base_end is not None:
            self._seg_base_end = base_end
            self._seg_header_bytes = bytes(data[:HEADER_SIZE])
            self._apply_tail(data, base_end, fresh=True)
            if self._segments_on:
                self._init_sigs()
        elif self._segments_on:
            self._init_sigs()

    def _create_new(self) -> None:
        self._reset_segment_state()
        self.header = FileHeader(vector_dimensions=self.vector_dimensions)
        self._nodes = {}
        self._content = {}
        self._edges = []
        self._edge_labels = {}
        self._name_index = {}
        self._next_node_id = 1
        self._next_edge_id = 1

    def _parse(self, data: bytes, base_end: Optional[int] = None) -> None:
        h = self.header
        self._nodes.clear()
        self._content.clear()
        self._edges.clear()
        self._edge_labels.clear()
        self._name_index.clear()
        # Round 77: a reload always starts from a fresh matrix.
        self._vectors = VectorStore(self.vector_dimensions)
        pending_vectors: List[Tuple[int, int]] = []

        # Slice the variable blocks. We need the boundaries; the string pool is
        # last so it runs to EOF. The other blocks run to the next block start.
        sp_off = h.string_pool_offset
        string_pool_bytes = data[sp_off:base_end] if sp_off else b""

        # --- Nodes ---
        base = h.node_table_offset
        for i in range(h.node_count):
            node = NodeRecord.unpack(data, base + i * NODE_SIZE)
            if node.node_id == 0:
                continue
            self._nodes[node.node_id] = node

            name = StringPool.read(string_pool_bytes, node.name_offset) if node.name_offset else ""
            content = _NodeContent(name=name)
            if name:
                self._name_index[name] = node.node_id

            if node.text_offset:
                content.text = self._read_len_prefixed(data, node.text_offset)
            if node.summary_offset:
                content.summary = self._read_len_prefixed(data, node.summary_offset)
            if node.wiki_offset:
                content.wiki = self._read_wiki(data, node.wiki_offset, string_pool_bytes)
            if node.vector_offset:
                content.has_vector = True
                pending_vectors.append((node.node_id, node.vector_offset))

            self._content[node.node_id] = content
            self._next_node_id = max(self._next_node_id, node.node_id + 1)

        # Round 77: fill the single float32 matrix in one pass, then validate it
        # in bulk (one C-level pass) instead of per node.
        if pending_vectors:
            self._vectors.reserve(len(pending_vectors))
            for _nid, _off in pending_vectors:
                self._vectors.append(
                    _nid, self._read_vector(data, _off), validate=False)
            self._vectors.validate_all()

        # --- Edges ---
        base = h.edge_table_offset
        for i in range(h.edge_count):
            edge = EdgeRecord.unpack(data, base + i * EDGE_SIZE)
            self._edges.append(edge)
            label = StringPool.read(string_pool_bytes, edge.relation_type) if edge.relation_type else ""
            self._edge_labels[edge.edge_id] = label
            self._next_edge_id = max(self._next_edge_id, edge.edge_id + 1)

    @staticmethod
    def _read_len_prefixed(data: bytes, offset: int) -> str:
        length = struct.unpack_from("<I", data, offset)[0]
        start = offset + 4
        return data[start:start + length].decode("utf-8")

    def _read_vector(self, data: bytes, offset: int):
        """Zero-copy view of an on-disk float32 row (copied into the matrix)."""
        dim = self.vector_dimensions
        if _np is not None:
            return _np.frombuffer(data, dtype="<f4", count=dim, offset=offset)
        return list(struct.unpack_from(f"<{dim}f", data, offset))  # pragma: no cover

    def _bytes_to_vector(self, vector):
        """Decode a packed little-endian float32 row (no Python float list)."""
        dim = self.vector_dimensions
        if len(vector) != dim * 4:
            raise ValueError(
                "vector must be %d bytes (%d float32), got %d"
                % (dim * 4, dim, len(vector)))
        if _np is not None:
            return _np.frombuffer(vector, dtype="<f4", count=dim)
        return list(struct.unpack(f"<{dim}f", vector))  # pragma: no cover

    @staticmethod
    def _read_wiki(data: bytes, offset: int, string_pool: bytes) -> Dict[str, str]:
        # [uint32 total_len][uint32 kv_count] then for each:
        # [uint32 key_offset_into_pool][uint32 val_len][val bytes]
        kv_count = struct.unpack_from("<I", data, offset + 4)[0]
        pos = offset + 8
        wiki: Dict[str, str] = {}
        for _ in range(kv_count):
            key_off = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            val_len = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            val = data[pos:pos + val_len].decode("utf-8")
            pos += val_len
            key = StringPool.read(string_pool, key_off)
            wiki[key] = val
        return wiki

    # ------------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------------

    def _require_write_txn(self, what: str) -> None:
        """r69: refuse in-memory-only mutations.

        Before this guard, update_node_wiki()/create_node()/create_edge() called
        outside write_txn() mutated the loaded image and returned successfully,
        but nothing ever saved it — Round 67e lost 144 entity fixes that way and
        reported success. A raise is the only honest answer.
        """
        if not getattr(self, "_in_txn", False):
            raise RuntimeError(
                f"{what}() called outside write_txn(): the edit would live only in "
                "memory and vanish on the next load. Wrap the mutation in "
                "`with mem.write_txn():` or use store()/link()."
            )

    def create_node(
        self,
        name: str,
        node_type: int,
        plane_flags: int,
        text: str = "",
        summary: str = "",
        wiki: Optional[Dict[str, str]] = None,
        vector: Optional[bytes] = None,
        importance: float = 0.5,
    ) -> NodeRecord:
        self._require_write_txn("create_node")
        node_id = self._next_node_id
        self._next_node_id += 1
        now = int(time.time() * 1000)

        node = NodeRecord(
            node_id=node_id,
            node_type=node_type,
            plane_flags=plane_flags,
            importance=importance,
            created_at=now,
            modified_at=now,
        )
        self._nodes[node_id] = node

        if vector is not None:
            self._vectors.append(node_id, self._bytes_to_vector(vector))
            self._dirty_vec.add(node_id)

        self._content[node_id] = _NodeContent(
            name=name, text=text, summary=summary, wiki=wiki,
            has_vector=vector is not None,
        )
        if name:
            self._name_index[name] = node_id
        return node

    def create_edge(
        self,
        source_node_id: int,
        target_node_id: int,
        plane: Plane,
        relation_label: str = "",
        weight: float = 1.0,
    ) -> EdgeRecord:
        self._require_write_txn("create_edge")
        edge_id = self._next_edge_id
        self._next_edge_id += 1
        now = int(time.time() * 1000)

        edge = EdgeRecord(
            edge_id=edge_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            plane_id=plane,
            weight=weight,
            created_at=now,
        )
        self._edges.append(edge)
        self._edge_labels[edge_id] = relation_label

        src = self._nodes.get(source_node_id)
        if src:
            src.edge_count += 1
        return edge

    # ------------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------------

    def get_node(self, node_id: int) -> Optional[NodeRecord]:
        return self._nodes.get(node_id)

    def node_exists(self, node_id: int) -> bool:
        """True if a node with this id is present. Cheap in-memory check."""
        return node_id in self._nodes

    def get_node_id_by_name(self, name: str) -> Optional[int]:
        return self._name_index.get(name)

    def get_name(self, node_id: int) -> str:
        c = self._content.get(node_id)
        return c.name if c else ""

    def get_text(self, node_id: int) -> str:
        c = self._content.get(node_id)
        return c.text if c else ""

    def get_summary(self, node_id: int) -> str:
        c = self._content.get(node_id)
        return c.summary if c else ""

    def get_wiki(self, node_id: int) -> Dict[str, str]:
        c = self._content.get(node_id)
        return dict(c.wiki) if c else {}

    def get_vector(self, node_id: int):
        """Row view straight into the store's float32 matrix, or None.

        Round 77: zero-copy. The old behaviour (a fresh list[float] per call)
        allocated ~32 KB per 1024-d node and made every read a copy; use
        get_vector_list() when a mutable Python list is genuinely needed.
        """
        return self._vectors.get(node_id)

    def get_vector_list(self, node_id: int) -> Optional[List[float]]:
        return self._vectors.get_list(node_id)

    def has_vector(self, node_id: int) -> bool:
        return self._vectors.has(node_id)

    def vector_rows(self):
        """(ids, matrix, norms, version) — zero-copy view of every vector.

        `matrix` is the store's own (N, dim) float32 array: scoring it does not
        build a second copy. Cache on `version`, which bumps on any add,
        update or drop (node_count does not).
        """
        return self._vectors.snapshot()

    def get_edges_from(self, source_node_id: int) -> List[EdgeRecord]:
        return [e for e in self._edges if e.source_node_id == source_node_id]

    def get_edge_label(self, edge_id: int) -> str:
        return self._edge_labels.get(edge_id, "")

    def nodes_in_plane(self, plane: Plane) -> List[NodeRecord]:
        mask = plane.value
        return [n for n in self._nodes.values() if n.plane_flags & mask]

    def all_node_ids(self) -> List[int]:
        return list(self._nodes.keys())

    # ------------------------------------------------------------------------
    # Mutators (used by the auto-ingestion layer for consolidation/decay)
    # ------------------------------------------------------------------------

    def update_node_wiki(self, node_id: int, updates: Dict[str, str]) -> None:
        """Merge key/value pairs into a node's wiki block (used to persist
        ingestion metadata: _qid, _base_weight, _reinforced_at, etc.)."""
        self._require_write_txn("update_node_wiki")
        c = self._content.get(node_id)
        if c is None:
            return
        c.wiki.update({str(k): str(v) for k, v in updates.items()})
        node = self._nodes.get(node_id)
        if node:
            node.modified_at = int(time.time() * 1000)

    def set_node_summary(self, node_id: int, summary: str) -> None:
        c = self._content.get(node_id)
        if c is not None:
            c.summary = summary

    def set_node_text(self, node_id: int, text: str) -> None:
        c = self._content.get(node_id)
        if c is not None:
            c.text = text

    def set_node_vector(self, node_id: int, vector) -> None:
        """vector may be bytes (packed float32) or a sequence of floats.

        Round 77: the row goes into the float32 matrix (upsert), never into a
        per-node Python list. Empty, wrong-length and all-zero vectors raise
        ValueError instead of silently producing a NaN cosine.
        """
        c = self._content.get(node_id)
        if c is None:
            return
        if isinstance(vector, (bytes, bytearray, memoryview)):
            vector = self._bytes_to_vector(vector)
        self._vectors.append(node_id, vector)
        self._dirty_vec.add(node_id)
        c.has_vector = True

    def set_node_importance(self, node_id: int, importance: float) -> None:
        node = self._nodes.get(node_id)
        if node:
            node.importance = max(0.0, min(1.0, importance))

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    def _save_locked(self) -> None:
        """Serialize + atomically write the whole image. Caller holds the lock."""
        if self._segments_on and self._can_append():
            if self._save_segment():
                return
        data = self._serialize()
        self._atomic_write(data)
        self._stat_key = self._current_stat()
        self.last_save_kind = "full"
        self.last_save_bytes = len(data)
        if self._segments_on:
            self._after_full_save(data)
        else:
            self._reset_segment_state()

    # ------------------------------------------------------------------------
    # P17: append-only segments
    # ------------------------------------------------------------------------

    def _reset_segment_state(self) -> None:
        self._seg_base_end = None
        self._seg_header_bytes = b""
        self._seg_end = 0
        self._seg_count = 0
        self._torn = False
        self._node_sigs = None
        self._edge_sigs = None
        self._dirty_vec = set()

    def _node_sig(self, nid: int) -> int:
        n = self._nodes[nid]
        c = self._content.get(nid)
        if c is None:
            return hash((n.node_type, n.plane_flags, n.edge_count, n.created_at,
                         n.modified_at, n.importance))
        return hash((n.node_type, n.plane_flags, n.edge_count, n.created_at,
                     n.modified_at, n.importance, c.name, c.text, c.summary,
                     tuple(c.wiki.items()), c.has_vector))

    def _edge_sig(self, e: EdgeRecord) -> int:
        return hash((e.source_node_id, e.target_node_id, int(e.plane_id), e.weight,
                     e.created_at, e.inverse_edge_id,
                     self._edge_labels.get(e.edge_id, "")))

    def _init_sigs(self) -> None:
        self._node_sigs = {nid: self._node_sig(nid) for nid in self._nodes}
        self._edge_sigs = {e.edge_id: self._edge_sig(e) for e in self._edges}
        self._dirty_vec = set()

    def _can_append(self) -> bool:
        return (self._seg_base_end is not None
                and self._node_sigs is not None
                and self._seg_count < self._compact_after
                and self.path.exists())

    def compact(self, keep_segments: Optional[bool] = None, timeout: float = 120.0) -> dict:
        """Fold every segment into a fresh base image (atomic rename, under the lock).

        keep_segments: True -> the new base is still segment-capable; False ->
        a plain legacy image (readable by pre-P17 code); None -> keep whatever
        this instance is set to. Returns before/after sizes and segment count.
        """
        fd = self._acquire_lock(timeout)
        try:
            if self._stat_key is not None and self._current_stat() != self._stat_key:
                self._refresh_under_lock()
            before = self._current_stat()[1] if self._current_stat() else 0
            segs = self._seg_count
            if keep_segments is not None:
                self._segments_on = bool(keep_segments)
            data = self._serialize()
            self._atomic_write(data)
            self._stat_key = self._current_stat()
            self.last_save_kind, self.last_save_bytes = "full", len(data)
            if self._segments_on:
                self._after_full_save(data)
            else:
                self._reset_segment_state()
            return {"segments_folded": segs, "bytes_before": before,
                    "bytes_after": len(data), "segment_capable": self._segments_on}
        finally:
            self._release_lock(fd)

    def _after_full_save(self, data: bytes) -> None:
        self._seg_base_end = _seg.read_base_end(self.header)
        self._seg_header_bytes = bytes(data[:HEADER_SIZE])
        self._seg_end = len(data)
        self._seg_count = 0
        self._torn = False
        self._init_sigs()

    def _save_segment(self) -> bool:
        """Append the delta since the last save as ONE segment. Caller holds the lock.

        Returns False (caller falls back to a full image save) only when the
        file is not in the state this instance expects.
        """
        try:
            size = os.stat(str(self.path)).st_size
        except OSError:
            return False
        if size < self._seg_end:
            return False
        cur_nodes = {nid: self._node_sig(nid) for nid in self._nodes}
        cur_edges = {e.edge_id: self._edge_sig(e) for e in self._edges}
        old_n, old_e = self._node_sigs, self._edge_sigs
        ch_nodes = [nid for nid, sg in cur_nodes.items()
                    if old_n.get(nid) != sg or nid in self._dirty_vec]
        del_nodes = [nid for nid in old_n if nid not in cur_nodes]
        ch_edges = [e for e in self._edges if old_e.get(e.edge_id) != cur_edges[e.edge_id]]
        del_edges = [eid for eid in old_e if eid not in cur_edges]
        if not (ch_nodes or del_nodes or ch_edges or del_edges):
            self._stat_key = self._current_stat()
            self.last_save_kind = "noop"
            self.last_save_bytes = 0
            return True
        dim = self.vector_dimensions
        node_rows = []
        for nid in sorted(ch_nodes):
            c = self._content.get(nid) or _NodeContent()
            vec = None
            if nid in self._dirty_vec or nid not in old_n:
                row = self._vectors.get(nid)
                if row is not None:
                    vec = (_np.ascontiguousarray(row, dtype="<f4").tobytes()
                           if _np is not None else struct.pack("<%df" % dim, *row))
            node_rows.append((self._nodes[nid], c.name, c.text, c.summary,
                              c.wiki, bool(c.has_vector), vec))
        edge_rows = [(e, self._edge_labels.get(e.edge_id, "")) for e in ch_edges]
        frame = _seg.encode(node_rows, del_nodes, edge_rows, del_edges, dim)
        try:
            _free = shutil.disk_usage(os.path.dirname(str(self.path)) or ".").free
            if _free < len(frame) * 2:
                raise OSError("MPMF_DISK_GUARD: refusing to append %d bytes to %s: "
                              "%.0f MB free" % (len(frame), self.path, _free / 1e6))
        except FileNotFoundError:
            pass
        with open(str(self.path), "r+b") as f:
            f.truncate(self._seg_end)       # drop a torn tail from a crashed append
            f.seek(self._seg_end)
            f.write(frame)
            f.flush()
            os.fsync(f.fileno())
        self._seg_end += len(frame)
        self._seg_count += 1
        self._torn = False
        self._node_sigs, self._edge_sigs = cur_nodes, cur_edges
        self._dirty_vec = set()
        self.header.node_count = len(self._nodes)
        self.header.edge_count = len(self._edges)
        self.header.plane_index_count = len(self._edges)
        self._stat_key = self._current_stat()
        self.last_save_kind = "segment"
        self.last_save_bytes = len(frame)
        return True

    def _apply_segment(self, payload, edge_index) -> None:
        nodes, dnodes, edges, dedges = _seg.decode(payload, self.vector_dimensions)
        for rec, name, text, summary, wiki, has_vector, vec in nodes:
            nid = rec.node_id
            old = self._content.get(nid)
            if old is not None and old.name and self._name_index.get(old.name) == nid \
                    and old.name != name:
                del self._name_index[old.name]
            self._nodes[nid] = rec
            self._content[nid] = _NodeContent(name=name, text=text, summary=summary,
                                              wiki=wiki, has_vector=has_vector)
            if name:
                self._name_index[name] = nid
            if vec is not None:
                self._vectors.append(nid, vec)
            elif not has_vector and self._vectors.has(nid):
                self._vectors.drop(nid)
            self._next_node_id = max(self._next_node_id, nid + 1)
        for nid in dnodes:
            c = self._content.pop(nid, None)
            self._nodes.pop(nid, None)
            if c is not None and c.name and self._name_index.get(c.name) == nid:
                del self._name_index[c.name]
            self._vectors.drop(nid)
        for rec, label in edges:
            i = edge_index.get(rec.edge_id)
            if i is None:
                edge_index[rec.edge_id] = len(self._edges)
                self._edges.append(rec)
            else:
                self._edges[i] = rec
            self._edge_labels[rec.edge_id] = label
            self._next_edge_id = max(self._next_edge_id, rec.edge_id + 1)
        if dedges:
            gone = set(dedges)
            self._edges = [e for e in self._edges if e.edge_id not in gone]
            for eid in gone:
                self._edge_labels.pop(eid, None)
            edge_index.clear()
            edge_index.update({e.edge_id: i for i, e in enumerate(self._edges)})

    def _apply_tail(self, data, start: int, fresh: bool) -> None:
        """Replay every good segment in ``data[start:]`` onto the loaded base."""
        payloads, end, torn = _seg.scan(data, start)
        edge_index = {e.edge_id: i for i, e in enumerate(self._edges)}
        for pl in payloads:
            self._apply_segment(pl, edge_index)
        self._seg_end = end
        self._seg_count = len(payloads) if fresh else self._seg_count + len(payloads)
        self._torn = torn
        self.header.node_count = len(self._nodes)
        self.header.edge_count = len(self._edges)
        self.header.plane_index_count = len(self._edges)

    def _refresh_incremental(self) -> bool:
        """Replay only the segments another writer appended since we last looked."""
        if self._seg_base_end is None or not self.path.exists():
            return False
        try:
            with open(str(self.path), "rb") as f:
                head = f.read(HEADER_SIZE)
                if head != self._seg_header_bytes:
                    return False            # base image was rewritten (compaction/full save)
                f.seek(0, os.SEEK_END)
                if f.tell() < self._seg_end:
                    return False
                f.seek(self._seg_end)
                tail = f.read()
        except OSError:
            return False
        base = self._seg_end
        self._apply_tail(tail, 0, fresh=False)
        self._seg_end += base                 # _apply_tail measured within `tail`
        self._stat_key = self._current_stat()
        if self._segments_on:
            self._init_sigs()
        return True

    def periodic_save(self, timeout: float = 10.0) -> None:
        """Save the whole image — never silently overwriting a newer one.

        Inside write_txn() the lock is already held and the image was reloaded
        on entry, so this is a plain save. Outside a transaction the lock is
        taken here and the on-disk (mtime_ns, size) must still match the image
        this instance loaded or last saved, otherwise StaleImageError is raised.
        """
        if self._in_txn:
            self._save_locked()
            return
        fd = self._acquire_lock(timeout)
        try:
            cur = self._current_stat()
            if self._stat_key is not None and cur != self._stat_key:
                raise StaleImageError(
                    "refusing to save %s: the on-disk image changed since this "
                    "instance loaded it (now %r, loaded %r). Another writer's "
                    "nodes would be destroyed. Reopen, or mutate inside "
                    "write_txn()." % (self.path, cur, self._stat_key))
            self._save_locked()
        finally:
            self._release_lock(fd)

    def _serialize(self) -> bytes:
        if self.header is None:
            self.header = FileHeader(vector_dimensions=self.vector_dimensions)
        header = self.header
        header.vector_dimensions = self.vector_dimensions

        sp = StringPool()

        # Order nodes by id for deterministic layout. Edges sorted by
        # (source_node_id, plane_id) so each node's edges are contiguous (spec).
        node_ids = sorted(self._nodes.keys())
        edges = sorted(self._edges, key=lambda e: (e.source_node_id, int(e.plane_id), e.edge_id))

        total_nodes = len(node_ids)
        total_edges = len(edges)

        node_table_size = total_nodes * NODE_SIZE
        edge_table_size = total_edges * EDGE_SIZE
        plane_index_size = total_edges * PLANE_INDEX_SIZE

        # --- Build variable blocks first; we need their content to compute
        #     offsets, but offsets stored in node records are ABSOLUTE file
        #     offsets, so we compute block base offsets up front. ---

        node_table_offset = HEADER_SIZE
        edge_table_offset = node_table_offset + node_table_size
        plane_index_offset = edge_table_offset + edge_table_size

        # Vector block must be 32-byte aligned.
        vector_block_offset = align_offset(plane_index_offset + plane_index_size)

        # Build vector block (fixed stride per node that HAS a vector; we store
        # a vector for every node that has one, padded to alignment per spec).
        vec_block = bytearray()
        vec_offsets: Dict[int, int] = {}
        dim = self.vector_dimensions
        vec_stride = align_offset(dim * 4)  # pad each vector to 32B boundary
        for nid in node_ids:
            row = self._vectors.get(nid)
            if row is not None:
                vec_offsets[nid] = vector_block_offset + len(vec_block)
                # float32 read -> float32 written: the bytes are identical to
                # what a list-of-floats path produced before Round 77.
                if _np is not None:
                    packed = _np.ascontiguousarray(row, dtype="<f4").tobytes()
                else:  # pragma: no cover
                    packed = struct.pack(f"<{dim}f", *row)
                vec_block.extend(packed)
                pad = vec_stride - len(packed)
                vec_block.extend(bytes(pad))

        summary_block_offset = vector_block_offset + len(vec_block)
        summary_block = bytearray()
        summary_offsets: Dict[int, int] = {}
        for nid in node_ids:
            c = self._content.get(nid)
            if c and c.summary:
                summary_offsets[nid] = summary_block_offset + len(summary_block)
                enc = c.summary.encode("utf-8")
                summary_block.extend(struct.pack("<I", len(enc)))
                summary_block.extend(enc)

        wiki_block_offset = summary_block_offset + len(summary_block)
        wiki_block = bytearray()
        wiki_offsets: Dict[int, int] = {}
        for nid in node_ids:
            c = self._content.get(nid)
            if c and c.wiki:
                wiki_offsets[nid] = wiki_block_offset + len(wiki_block)
                # placeholder header; fill total_len after building body
                body = bytearray()
                body.extend(struct.pack("<I", len(c.wiki)))  # kv_count
                for k, v in c.wiki.items():
                    key_off = sp.intern(str(k))
                    vbytes = str(v).encode("utf-8")
                    body.extend(struct.pack("<I", key_off))
                    body.extend(struct.pack("<I", len(vbytes)))
                    body.extend(vbytes)
                total_len = 4 + len(body)  # includes the leading total_len field
                wiki_block.extend(struct.pack("<I", total_len))
                wiki_block.extend(body)

        text_block_offset = wiki_block_offset + len(wiki_block)
        text_block = bytearray()
        text_offsets: Dict[int, int] = {}
        for nid in node_ids:
            c = self._content.get(nid)
            if c and c.text:
                text_offsets[nid] = text_block_offset + len(text_block)
                enc = c.text.encode("utf-8")
                text_block.extend(struct.pack("<I", len(enc)))
                text_block.extend(enc)

        metadata_block_offset = text_block_offset + len(text_block)
        metadata_block = bytearray()  # unused for now (per-node metadata optional)

        # Intern all names + edge relation labels into the string pool.
        name_offsets: Dict[int, int] = {}
        for nid in node_ids:
            c = self._content.get(nid)
            if c and c.name:
                name_offsets[nid] = sp.intern(c.name)
        label_offsets: Dict[int, int] = {}
        for e in edges:
            lbl = self._edge_labels.get(e.edge_id, "")
            label_offsets[e.edge_id] = sp.intern(lbl) if lbl else 0

        string_pool_offset = metadata_block_offset + len(metadata_block)
        string_pool_bytes = sp.bytes()

        # --- Now patch node records with their resolved absolute offsets ---
        # Recompute each node's edge_list_offset (absolute offset of first edge).
        first_edge_offset: Dict[int, int] = {}
        for idx, e in enumerate(edges):
            if e.source_node_id not in first_edge_offset:
                first_edge_offset[e.source_node_id] = edge_table_offset + idx * EDGE_SIZE

        # Assemble the file.
        out = bytearray()
        out.extend(bytes(HEADER_SIZE))  # header placeholder

        for nid in node_ids:
            node = self._nodes[nid]
            node.name_offset = name_offsets.get(nid, 0)
            node.vector_offset = vec_offsets.get(nid, 0)
            node.summary_offset = summary_offsets.get(nid, 0)
            node.wiki_offset = wiki_offsets.get(nid, 0)
            node.text_offset = text_offsets.get(nid, 0)
            node.metadata_offset = 0
            node.edge_list_offset = first_edge_offset.get(nid, 0)
            out.extend(node.pack())

        for e in edges:
            e.relation_type = label_offsets.get(e.edge_id, 0)
            out.extend(e.pack())

        # Plane index: one entry per edge, sorted by (plane_id, source_node_id).
        plane_entries = sorted(
            ((int(e.plane_id), e.source_node_id, e.edge_id) for e in edges)
        )
        for plane_id, src, eid in plane_entries:
            out.extend(PlaneIndexEntry(plane_id, src, eid).pack())

        # Pad to vector alignment.
        out.extend(bytes(vector_block_offset - len(out)))
        out.extend(vec_block)
        out.extend(summary_block)
        out.extend(wiki_block)
        out.extend(text_block)
        out.extend(metadata_block)
        out.extend(string_pool_bytes)

        # Fill header.
        header.node_count = total_nodes
        header.edge_count = total_edges
        header.plane_index_count = total_edges
        header.modified_at = int(time.time())
        header.node_table_offset = node_table_offset
        header.edge_table_offset = edge_table_offset
        header.plane_index_offset = plane_index_offset
        header.vector_block_offset = vector_block_offset
        header.summary_block_offset = summary_block_offset
        header.wiki_block_offset = wiki_block_offset
        header.text_block_offset = text_block_offset
        header.metadata_block_offset = metadata_block_offset
        header.string_pool_offset = string_pool_offset

        # P17: mark (or un-mark) the file as segment-capable. base_end is the
        # end of this image; later segments are appended after it and are NOT
        # covered by the checksum below.
        if self._segments_on:
            header.flags = int(header.flags) | _seg.FLAG_SEGMENTS
            header._reserved = _seg.pack_reserved(len(out))
        elif _seg.read_base_end(header) is not None:
            header.flags = int(header.flags) & ~_seg.FLAG_SEGMENTS
            header._reserved = bytes(80)

        header.checksum = hashlib.sha256(memoryview(out)[HEADER_SIZE:]).digest()
        out[:HEADER_SIZE] = header.pack()
        return bytes(out)

    def _atomic_write(self, data: bytes) -> None:
        path = str(self.path)
        dir_path = os.path.dirname(path) or "."
        # MPMF_DISK_GUARD: writing a full store copy needs ~2x its size free
        # (temp file + rename). Round 89e died with Errno 28 mid-write.
        try:
            _free = shutil.disk_usage(dir_path).free
            _need = len(data) * 2
            if _free < _need:
                raise OSError(
                    "MPMF_DISK_GUARD: refusing to save %s: need %.0f MB free "
                    "(2x store), have %.0f MB. Free space or the write would "
                    "fail with ENOSPC."
                    % (path, _need / 1e6, _free / 1e6))
        except FileNotFoundError:
            pass
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, prefix=".mpmf_tmp_")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def __repr__(self) -> str:
        return f"MemoryManager(nodes={len(self._nodes)}, edges={len(self._edges)})"
