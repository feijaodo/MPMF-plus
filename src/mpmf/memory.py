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
import tempfile
import time
import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    __slots__ = ("name", "text", "summary", "wiki", "vector", "relation_labels")

    def __init__(self, name="", text="", summary="", wiki=None, vector=None):
        self.name = name
        self.text = text
        self.summary = summary
        self.wiki: Dict[str, str] = wiki or {}
        self.vector: Optional[List[float]] = vector


# ============================================================================
# Memory Manager
# ============================================================================

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

    def __init__(self, path: str, vector_dimensions: int = DEFAULT_VECTOR_DIM):
        self.path = Path(path)
        self.vector_dimensions = vector_dimensions
        self.header: Optional[FileHeader] = None

        self._nodes: Dict[int, NodeRecord] = {}
        self._content: Dict[int, _NodeContent] = {}
        self._edges: List[EdgeRecord] = []
        self._edge_labels: Dict[int, str] = {}
        self._name_index: Dict[str, int] = {}

        self._next_node_id = 1
        self._next_edge_id = 1

        # --- Round 48c: cross-process write safety -------------------------
        # (mtime_ns, size) of the image this instance is holding. Set on load
        # and after every save; used to detect that another writer has
        # replaced the file underneath us.
        self._stat_key: Optional[Tuple[int, int]] = None
        self._in_txn: bool = False

    # ------------------------------------------------------------------------
    # Round 48c: write transactions
    # ------------------------------------------------------------------------

    def _current_stat(self) -> Optional[Tuple[int, int]]:
        try:
            st = os.stat(str(self.path))
            return (st.st_mtime_ns, st.st_size)
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
                self._reload_under_lock()
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
        if not self.path.exists():
            self._create_new()
            return

        with open(self.path, "rb") as f:
            data = f.read()

        if len(data) < HEADER_SIZE:
            raise ValueError(f"File too short: {len(data)} < {HEADER_SIZE}")

        self.header = FileHeader.unpack(data)
        if self.header.vector_dimensions:
            self.vector_dimensions = self.header.vector_dimensions

        # Verify checksum (computed over everything after byte 256).
        stored = self.header.checksum
        computed = hashlib.sha256(data[HEADER_SIZE:]).digest()
        if stored != computed:
            raise ValueError("Checksum mismatch — file corrupted")

        self._parse(data)

    def _create_new(self) -> None:
        self.header = FileHeader(vector_dimensions=self.vector_dimensions)
        self._nodes = {}
        self._content = {}
        self._edges = []
        self._edge_labels = {}
        self._name_index = {}
        self._next_node_id = 1
        self._next_edge_id = 1

    def _parse(self, data: bytes) -> None:
        h = self.header
        self._nodes.clear()
        self._content.clear()
        self._edges.clear()
        self._edge_labels.clear()
        self._name_index.clear()

        # Slice the variable blocks. We need the boundaries; the string pool is
        # last so it runs to EOF. The other blocks run to the next block start.
        sp_off = h.string_pool_offset
        string_pool_bytes = data[sp_off:] if sp_off else b""

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
                content.vector = self._read_vector(data, node.vector_offset)

            self._content[node.node_id] = content
            self._next_node_id = max(self._next_node_id, node.node_id + 1)

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

    def _read_vector(self, data: bytes, offset: int) -> List[float]:
        dim = self.vector_dimensions
        return list(struct.unpack_from(f"<{dim}f", data, offset))

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

        vec_list = None
        if vector is not None:
            vec_list = list(struct.unpack(f"<{self.vector_dimensions}f", vector))

        self._content[node_id] = _NodeContent(
            name=name, text=text, summary=summary, wiki=wiki, vector=vec_list,
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

    def get_vector(self, node_id: int) -> Optional[List[float]]:
        c = self._content.get(node_id)
        return list(c.vector) if c and c.vector is not None else None

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
        """vector may be bytes (packed float32) or a list[float]."""
        import struct as _struct
        c = self._content.get(node_id)
        if c is None:
            return
        if isinstance(vector, (bytes, bytearray)):
            c.vector = list(_struct.unpack(f"<{self.vector_dimensions}f", vector))
        else:
            c.vector = list(vector)

    def set_node_importance(self, node_id: int, importance: float) -> None:
        node = self._nodes.get(node_id)
        if node:
            node.importance = max(0.0, min(1.0, importance))

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    def _save_locked(self) -> None:
        """Serialize + atomically write the whole image. Caller holds the lock."""
        self._data = self._serialize()
        self._atomic_write(self._data)
        self._stat_key = self._current_stat()

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
            c = self._content.get(nid)
            if c and c.vector is not None:
                vec_offsets[nid] = vector_block_offset + len(vec_block)
                packed = struct.pack(f"<{dim}f", *c.vector)
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

        header.checksum = hashlib.sha256(bytes(out[HEADER_SIZE:])).digest()
        out[:HEADER_SIZE] = header.pack()
        return bytes(out)

    def _atomic_write(self, data: bytes) -> None:
        path = str(self.path)
        dir_path = os.path.dirname(path) or "."
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
