"""
mpmf/embedding.py — Pluggable embedding provider for MPMF.

The vector plane needs text -> fixed-dim float vector. Three tiers (in order):

1. Ollama (when use_ollama=True): POSTs to http://localhost:11434/api/embeddings
   using a local embedding model (default: nomic-embed-text / mxbai path).
2. OpenAI-compatible HTTP (OpenRouter etc.): used when Ollama is unavailable
   or disabled. Reads OPENROUTER_API_KEY / OPENAI_API_KEY, OPENAI_BASE_URL,
   AYU_EMBED_MODEL. Added for remote-only Termux/Linux installs.
3. Hash fallback: deterministic bag-of-words hashing embedder. Not semantically
   rich, but lets MPMF run anywhere with no model or network.

The provider exposes embed(text) -> list[float] of exactly `dim` floats, and
pack(text) -> bytes (little-endian float32) ready for MemoryManager.create_node.
"""

import json
import math
import struct
import urllib.request
import urllib.error
from typing import List, Optional

OLLAMA_URL = "http://localhost:11434/api/embeddings"
DEFAULT_MODEL = "nomic-embed-text"
NOMIC_DIM = 768
# Canonical vector dimensions — must match the embedder's output size.
# mxbai-embed-large → 1024, nomic-embed-text → 768
DEFAULT_VECTOR_DIM = 1024  # matches mxbai-embed-large (used by ingest.py)


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


class EmbeddingProvider:
    def __init__(self, dim: int = DEFAULT_VECTOR_DIM, model: str = DEFAULT_MODEL, use_ollama: bool = True):
        self.dim = dim
        self.model = model
        self.use_ollama = use_ollama
        self._ollama_ok: Optional[bool] = None
        self._ollama_failures = 0

    # ------------------------------------------------------------------
    def _try_ollama(self, text: str) -> Optional[List[float]]:
        if not self.use_ollama:
            return None
        payload = json.dumps({"model": self.model, "prompt": text}).encode()
        req = urllib.request.Request(
            OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            emb = data.get("embedding")
            if not emb:
                return None
            self._ollama_ok = True
            self._ollama_failures = 0
            # Resize to self.dim if the model's native dim differs.
            if len(emb) != self.dim:
                emb = self._resize(emb, self.dim)
            return _l2_normalize(emb)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
            self._ollama_failures += 1
            # Re-probe Ollama after 3 consecutive failures (e.g. service restart)
            if self._ollama_failures >= 3:
                self._ollama_ok = None
            return None

    @staticmethod
    def _resize(vec: List[float], target: int) -> List[float]:
        if len(vec) == target:
            return vec
        if len(vec) > target:
            return vec[:target]
        return vec + [0.0] * (target - len(vec))

    # ------------------------------------------------------------------
    def _try_openai_compat(self, text: str) -> Optional[List[float]]:
        """OpenRouter / OpenAI-compatible embeddings (remote, stdlib only)."""
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        base_url = os.environ.get(
            "OPENAI_BASE_URL", "https://openrouter.ai/api/v1"
        ).rstrip("/")
        model = os.environ.get("AYU_EMBED_MODEL", "openai/text-embedding-3-small")
        payload = json.dumps({"model": model, "input": text}).encode()
        req = urllib.request.Request(
            f"{base_url}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            emb = data["data"][0]["embedding"]
            if len(emb) != self.dim:
                emb = self._resize(emb, self.dim)
            self._ollama_ok = True
            self._ollama_failures = 0
            return _l2_normalize(emb)
        except Exception:
            self._ollama_failures += 1
            return None

    # ------------------------------------------------------------------
    def _hash_embed(self, text: str) -> List[float]:
        """Deterministic bag-of-words hashing embedder (fallback)."""
        vec = [0.0] * self.dim
        tokens = [t for t in text.lower().split() if t]
        for tok in tokens:
            h = 2166136261
            for ch in tok.encode("utf-8"):  # FNV-1a
                h ^= ch
                h = (h * 16777619) & 0xFFFFFFFF
            idx = h % self.dim
            sign = 1.0 if (h >> 16) & 1 else -1.0
            vec[idx] += sign
        return _l2_normalize(vec)

    # ------------------------------------------------------------------
    def _prefix(self, text: str, kind: str) -> str:
        """Embedding models trained with task-instruction prefixes need them
        applied for good retrieval separation. Different families use different
        schemes:

        - nomic-embed-text: symmetric. 'search_document:' for stored text,
          'search_query:' for queries.
        - mxbai-embed-large: asymmetric. Documents get NO prefix; queries get
          'Represent this sentence for searching relevant passages: '.
        - others / hash fallback: raw text.
        """
        m = self.model.lower()
        if "nomic" in m:
            return f"search_query: {text}" if kind == "query" else f"search_document: {text}"
        if "mxbai" in m or "mixedbread" in m:
            if kind == "query":
                return f"Represent this sentence for searching relevant passages: {text}"
            return text
        return text

    def embed(self, text: str, kind: str = "document") -> List[float]:
        if not text:
            return [0.0] * self.dim
        emb = self._try_ollama(self._prefix(text, kind))
        if emb is not None:
            return emb
        emb = self._try_openai_compat(text)
        if emb is not None:
            return emb
        return self._hash_embed(text)

    def pack(self, text: str, kind: str = "document") -> bytes:
        return struct.pack(f"<{self.dim}f", *self.embed(text, kind))

    @property
    def backend(self) -> str:
        if self._ollama_ok is True:
            return f"ollama:{self.model}"
        if self._ollama_ok is False:
            return "hash-fallback"
        return "unprobed"
