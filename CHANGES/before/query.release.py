"""
mpmf/query.py — High-level agent-facing API for MPMF.

This is the interface an agent actually uses. It hides records/offsets/planes
behind two verbs: remember() and recall().

    mem = MPMFMemory("~/.mpmf/store.mpmf")
    mem.remember("User prefers concise answers",
                 name="pref_concise", node_type=NodeType.FACT,
                 wiki={"topic": "preference"})
    mem.link("pref_concise", "comms_style", Plane.ASSOCIATIVE, "RELATED_TO")
    result = mem.recall("how does the user like answers?")
    for nid in result.top_nodes:
        print(result.raw_texts[nid])
    mem.save()

Embeddings are generated automatically (Ollama nomic-embed-text, hash fallback)
so the vector plane works without the caller managing vectors by hand.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from .format import Plane, NodeType, NodeRecord
from .memory import MemoryManager
from .embedding import EmbeddingProvider
from .retrieval import RetrievalPipeline, RetrievalResult


class MPMFMemory:
    def __init__(
        self,
        path: str,
        vector_dimensions: int = 768,
        use_ollama: bool = True,
        embed_model: str = "nomic-embed-text",
    ):
        self.path = str(Path(path).expanduser())
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.mem = MemoryManager(self.path, vector_dimensions=vector_dimensions)
        self.mem.load()
        self.embedder = EmbeddingProvider(
            vector_dimensions, model=embed_model, use_ollama=use_ollama
        )
        self._dirty = False

    # ---- write --------------------------------------------------------
    def remember(
        self,
        text: str,
        name: str,
        node_type: int = NodeType.FACT,
        planes: int = Plane.ASSOCIATIVE.value,
        summary: str = "",
        wiki: Optional[Dict[str, str]] = None,
        importance: float = 0.5,
        embed_from: Optional[str] = None,
    ) -> NodeRecord:
        """Store a memory. Embedding is auto-generated from embed_from or
        (summary or text). Returns the created NodeRecord."""
        if not summary:
            summary = text[:120]
        emb_src = embed_from or summary or text
        vec = self.embedder.pack(emb_src)
        node = self.mem.create_node(
            name=name, node_type=node_type, plane_flags=planes,
            text=text, summary=summary, wiki=wiki, vector=vec,
            importance=importance,
        )
        self._dirty = True
        return node

    def link(
        self,
        source_name: str,
        target_name: str,
        plane: Plane = Plane.ASSOCIATIVE,
        relation: str = "RELATED_TO",
        weight: float = 1.0,
    ):
        """Create an edge between two named nodes."""
        sid = self.mem.get_node_id_by_name(source_name)
        tid = self.mem.get_node_id_by_name(target_name)
        if sid is None or tid is None:
            raise KeyError(f"Unknown node(s): {source_name!r} / {target_name!r}")
        self._dirty = True
        return self.mem.create_edge(sid, tid, plane, relation, weight)

    def save(self):
        self.mem.periodic_save()
        self._dirty = False

    # ---- read ---------------------------------------------------------
    def recall(
        self,
        query: str,
        depth: int = 2,
        top_k: int = 20,
        top_n: int = 10,
        max_iterations: int = 2,
    ) -> RetrievalResult:
        pipeline = RetrievalPipeline(
            self.mem, embedder=self.embedder,
            depth=depth, top_k=top_k, max_iterations=max_iterations,
        )
        return pipeline.query(query, top_n=top_n)

    def get(self, name: str) -> Optional[Dict]:
        nid = self.mem.get_node_id_by_name(name)
        if nid is None:
            return None
        node = self.mem.get_node(nid)
        return {
            "id": nid,
            "name": name,
            "text": self.mem.get_text(nid),
            "summary": self.mem.get_summary(nid),
            "wiki": self.mem.get_wiki(nid),
            "importance": node.importance,
            "node_type": int(node.node_type),
            "plane_flags": node.plane_flags,
        }

    def stats(self) -> Dict:
        return {
            "path": self.path,
            "nodes": len(self.mem.all_node_ids()),
            "edges": len(self.mem._edges),
            "vector_dim": self.mem.vector_dimensions,
            "embed_backend": self.embedder.backend,
            "dirty": self._dirty,
        }

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._dirty:
            self.save()
        return False


# Backwards-compatible convenience functions.
def open_memory(path: str, **kwargs) -> MPMFMemory:
    return MPMFMemory(path, **kwargs)


# -- mpmf-plus compatibility extensions on core MPMFMemory --------------------
# The harness MpmfVersionedStore expects list_names / read_version /
# compare_and_swap on MPMFMemory. These methods used to live only in the
# mpmf-plus agent_memory layer; they are first-class on core after the
# Android/Termux port so bare-pipeline installs work without mpmf-plus CAS.
# Create-only CAS (expected_version=None) MUST fail if the name already
# exists — remember() does not enforce uniqueness and would otherwise race.

def _mpmf_list_names(self, prefix=""):
    """List all node names, optionally filtered by prefix."""
    result = []
    for nid in self.mem.all_node_ids():
        name = self.mem.get_name(nid)
        if name and (not prefix or name.startswith(prefix)):
            result.append(name)
    return result


def _mpmf_read_version(self, name):
    """Read the version of a node (for CAS). Returns None if missing."""
    nid = self.mem.get_node_id_by_name(name)
    if nid is None:
        return None
    node = self.mem.get_node(nid)
    if node is None:
        return None
    wiki = self.mem.get_wiki(nid)
    return wiki.get("_mpmf_version", str(node.modified_at))


def _mpmf_compare_and_swap(self, name, expected_version=None, text="", wiki=None):
    """Optimistic compare-and-swap. Thread-safe via instance lock.

    Returns (success: bool, version|None).
    expected_version=None means create-only: succeeds only if name is new.
    """
    import threading as _threading
    import time as _time

    if not hasattr(self, "_cas_lock"):
        self._cas_lock = _threading.Lock()
    with self._cas_lock:
        nid = self.mem.get_node_id_by_name(name)
        if expected_version is None:
            # Create-only: succeed only if node does not exist
            if nid is not None:
                return False, None
            self.remember(text, name=name, wiki=wiki or {})
            nid = self.mem.get_node_id_by_name(name)
            new_ver = str(int(_time.time() * 1000))
            self.mem.update_node_wiki(nid, {"_mpmf_version": new_ver})
            self._dirty = True
            return True, new_ver
        # Update mode: check version match
        if nid is None:
            return False, None
        node = self.mem.get_node(nid)
        if node is None:
            return False, None
        current_wiki = self.mem.get_wiki(nid)
        current_ver = current_wiki.get("_mpmf_version", str(node.modified_at))
        if current_ver != expected_version:
            return False, current_ver
        self.mem.set_node_text(nid, text)
        if wiki:
            for k, v in wiki.items():
                current_wiki[k] = v
            self.mem.update_node_wiki(nid, current_wiki)
        new_ver = str(int(_time.time() * 1000))
        self.mem.update_node_wiki(nid, {"_mpmf_version": new_ver})
        self._dirty = True
        return True, new_ver


MPMFMemory.list_names = _mpmf_list_names
MPMFMemory.read_version = _mpmf_read_version
MPMFMemory.compare_and_swap = _mpmf_compare_and_swap
