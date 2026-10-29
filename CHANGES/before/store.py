"""
mpmf_plus.store — high-level store/recall API with a write-lock and an intake gate.

The store file is a whole-file image (periodic_save() serialises everything and
os.replace()s it). Two writers that load at T0 and save at T2 would silently
destroy each other's nodes, so every mutation runs inside an exclusive
cross-process transaction:

    * a POSIX advisory lock (fcntl.flock) on ``<store>.lock`` serialises
      writers ACROSS processes,
    * a process-local threading lock serialises writers WITHIN the process,
    * on entry the instance reloads if the file moved on disk, so a save cannot
      clobber a concurrent writer's nodes.

The intake gate sits in front of the write: a fact whose text hash already
exists is merged instead of duplicated, so parallel writes of the same fact
converge to exactly one node.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None

from mpmf import MPMFMemory, NodeType, Plane

from .config import RecallConfig, StoreConfig
from .embedder import Embedder, get_embedder
from .recall import Hit, RecallEngine, render_block
from .reranker import Reranker, get_reranker


class LockTimeoutError(RuntimeError):
    pass


#: One process-local lock per store path, shared by every Store instance and
#: every thread in the process. This makes intra-process writers queue instead
#: of all racing for the core (cross-process) flock at once.
_WRITE_LOCKS: Dict[str, threading.Lock] = {}
_WRITE_LOCKS_GUARD = threading.Lock()


def _process_write_lock(path: Path) -> threading.Lock:
    key = str(Path(path).resolve())
    with _WRITE_LOCKS_GUARD:
        lock = _WRITE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _WRITE_LOCKS[key] = lock
        return lock


class WriteLock:
    """Process-local write mutex.

    Cross-process exclusion is provided by
    :meth:`mpmf.memory.MemoryManager.write_txn` (a POSIX ``flock`` on
    ``<store>.lock`` with a reload-if-stale on entry and an atomic rename on
    save). This lock only serialises writers WITHIN the process so that at most
    one thread ever contends for that flock — otherwise many threads would each
    retry the flock and could exhaust the timeout.
    """

    def __init__(self, store_path: Path, timeout: float = 10.0):
        self.store_path = store_path
        self.lock_path = str(store_path) + ".lock"
        self.timeout = timeout
        # Shared per-path lock: every Store/WriteLock instance on the same file
        # must contend for ONE mutex, otherwise threads clobber each other.
        self._thread_lock = _process_write_lock(Path(store_path))

    def __enter__(self):
        acquired = self._thread_lock.acquire(timeout=max(0.0, self.timeout))
        if not acquired:
            raise LockTimeoutError(
                f"process-local store lock {self.lock_path} not acquired "
                f"within {self.timeout:.1f}s")
        return self

    def __exit__(self, *exc):
        self._thread_lock.release()
        return False


def _text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


class Store:
    """A single MPMF+ store: writes are locked, reads use a fresh snapshot."""

    def __init__(self, config: Optional[StoreConfig] = None,
                 recall_config: Optional[RecallConfig] = None,
                 embedder: Optional[Embedder] = None,
                 reranker: Optional[Reranker] = None,
                 path: Optional[str] = None, **kwargs: Any):
        if config is None:
            config = StoreConfig.load(path=path, **kwargs)
        self.config = config
        self.recall_config = recall_config or RecallConfig.load()
        self.path = Path(self.config.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder  # lazily built per snapshot if None
        self._reranker = reranker
        self._read: Optional[MPMFMemory] = None

    # ---- snapshots --------------------------------------------------------
    def _snapshot(self) -> MPMFMemory:
        """A memoized read image, invalidated by mtime+size."""
        key = None
        try:
            st = os.stat(self.path)
            key = (st.st_mtime_ns, st.st_size)
        except OSError:
            key = None
        if getattr(self, "_read_key", None) == key and self._read is not None:
            return self._read
        mem = MPMFMemory(str(self.path), vector_dimensions=self.config.vector_dim,
                         embed_model=self.config.embed_model)
        self._read = mem
        self._read_key = key
        return mem

    def _fresh_rw(self) -> MPMFMemory:
        return MPMFMemory(str(self.path), vector_dimensions=self.config.vector_dim,
                          embed_model=self.config.embed_model)

    def _embedder_for(self, mem: MPMFMemory) -> Embedder:
        if self._embedder is not None:
            return self._embedder
        return get_embedder(mem.vector_dimensions, model=self.config.embed_model)

    # ---- write ------------------------------------------------------------
    def store(self, text: str, name: str, *, namespace: str = "memory",
              summary: str = "", wiki: Optional[Dict[str, str]] = None,
              importance: float = 0.5, node_type: int = NodeType.FACT,
              planes: int = Plane.ASSOCIATIVE.value) -> Dict[str, Any]:
        """Store a fact under the write-lock, with intake-gate dedup.

        Returns ``{"node_id", "name", "created", "merged"}``.
        """
        with WriteLock(self.path, self.config.write_timeout):
            mem = self._fresh_rw()
            mm = mem.mem
            with mm.write_txn(timeout=self.config.write_timeout):
                digest = _text_hash(text)
                existing = self._find_by_digest(mm, digest)
                if existing is not None:
                    return {"node_id": existing, "name": name,
                            "created": False, "merged": True}
                w = dict(wiki or {})
                w.setdefault("_ns", namespace)
                w["_digest"] = digest
                if not summary:
                    summary = text[:120]
                emb = self._embedder_for(mem)
                vec = emb.pack(summary or text)
                node = mm.create_node(name=name, node_type=node_type,
                                      plane_flags=planes, text=text,
                                      summary=summary, wiki=w, vector=vec,
                                      importance=importance)
                nid = node.node_id
            self._read = None  # invalidate snapshot
            return {"node_id": nid, "name": name, "created": True, "merged": False}

    @staticmethod
    def _find_by_digest(mm, digest: str) -> Optional[int]:
        for nid in mm.all_node_ids():
            if (mm.get_wiki(nid) or {}).get("_digest") == digest:
                return nid
        return None

    def link(self, a: str, b: str, *, relation: str = "RELATED_TO",
             plane: Plane = Plane.ASSOCIATIVE) -> None:
        with WriteLock(self.path, self.config.write_timeout):
            mem = self._fresh_rw()
            mm = mem.mem
            with mm.write_txn(timeout=self.config.write_timeout):
                sid = mm.get_node_id_by_name(a)
                tid = mm.get_node_id_by_name(b)
                if sid is None or tid is None:
                    raise KeyError(f"unknown node(s): {a!r} / {b!r}")
                mm.create_edge(sid, tid, plane, relation)
            self._read = None

    # ---- read -------------------------------------------------------------
    def recall(self, query: str, top_n: Optional[int] = None,
               namespaces: Optional[Set[str]] = None) -> List[Hit]:
        mem = self._snapshot()
        eng = RecallEngine(mem, embedder=self._embedder_for(mem),
                           reranker=self._reranker, config=self.recall_config)
        return eng.recall(query, top_n=top_n, namespaces=namespaces)

    def recall_block(self, query: str, top_n: Optional[int] = None,
                     namespaces: Optional[Set[str]] = None) -> str:
        hits = self.recall(query, top_n=top_n, namespaces=namespaces)
        return render_block(hits, self.recall_config, query=query)

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        mem = self._snapshot()
        return mem.get(name)

    def stats(self) -> Dict[str, Any]:
        mem = self._snapshot()
        return mem.stats()
