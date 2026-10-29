"""OpenAI-compatible embedding provider for MPMF.

Picks up OPENAI_API_KEY (or OPENROUTER_API_KEY as fallback) and an optional
OPENAI_BASE_URL so the same provider routes to OpenRouter's OpenAI-compatible
embeddings endpoint. Truncates the native embedding to embed_dim if needed.
"""
from __future__ import annotations

import os
import struct
import urllib.error
import urllib.request
import json
import math
from typing import List, Optional


def _l2_normalize(vec: List[float]) -> List[float]:
    n = math.sqrt(sum(v * v for v in vec))
    if n == 0:
        return vec
    return [v / n for v in vec]


class OpenAICompatEmbedder:
    def __init__(
        self,
        dim: int,
        model: str = "openai/text-embedding-3-small",
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.dim = dim
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OpenAICompatEmbedder: set OPENAI_API_KEY or pass api_key= (OpenRouter key works)."
            )
        self.timeout = timeout

    def _call(self, text: str) -> List[float]:
        body = json.dumps({"model": self.model, "input": text}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode())
        return data["data"][0]["embedding"]

    @staticmethod
    def _resize(vec: List[float], target: int) -> List[float]:
        if len(vec) == target:
            return vec
        if len(vec) > target:
            return vec[:target]
        return vec + [0.0] * (target - len(vec))

    def embed(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.dim
        emb = self._call(text)
        emb = self._resize(emb, self.dim)
        return _l2_normalize(emb)

    def pack(self, text: str) -> bytes:
        return struct.pack(f"<{self.dim}f", *self.embed(text))

    @property
    def backend(self) -> str:
        return f"openai-compat:{self.model}@{self.base_url}"
