"""
MPMF: Multi-Plane Memory Format — Python Reference Implementation (v0.1)

A binary graph memory format for LLM agents, storing information across
five associative planes simultaneously with attached raw-text / summary /
wiki / vector content per node.

Quick start:

    from mpmf import MPMFMemory, Plane, NodeType
    mem = MPMFMemory("~/.mpmf/store.mpmf")
    mem.remember("Fact text", name="fact1", wiki={"k": "v"})
    mem.save()
    res = mem.recall("query text")
"""

from mpmf.format import (
    Plane,
    NodeType,
    NodeRecord,
    EdgeRecord,
    PlaneIndexEntry,
    FileHeader,
    MAGIC,
    HEADER_SIZE,
    NODE_SIZE,
    EDGE_SIZE,
    PLANE_INDEX_SIZE,
)
from mpmf.memory import MemoryManager, StringPool
from mpmf.embedding import EmbeddingProvider
from mpmf.retrieval import RetrievalPipeline, RetrievalResult
from mpmf.query import MPMFMemory, open_memory

__version__ = "1.1.0"  # 5-plane core with iterative ingestion support
__all__ = [
    # high-level API
    "MPMFMemory",
    "open_memory",
    # core
    "MemoryManager",
    "StringPool",
    "EmbeddingProvider",
    "RetrievalPipeline",
    "RetrievalResult",
    # format
    "Plane",
    "NodeType",
    "NodeRecord",
    "EdgeRecord",
    "PlaneIndexEntry",
    "FileHeader",
    "MAGIC",
    "HEADER_SIZE",
    "NODE_SIZE",
    "EDGE_SIZE",
    "PLANE_INDEX_SIZE",
]
