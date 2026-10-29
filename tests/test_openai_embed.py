"""Tests for the OpenAI-compatible embedding provider."""
import math
import os
import sys

import pytest

from mpmf.openai_embed import OpenAICompatEmbedder  # noqa: E402


def _has_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"))


@pytest.mark.skipif(not _has_key(), reason="No OPENAI_API_KEY/OPENROUTER_API_KEY in env")
def test_openai_compat_embedder_returns_vector_at_requested_dim():
    emb = OpenAICompatEmbedder(dim=1536, model="openai/text-embedding-3-small")
    vec = emb.embed("the user lives in Austin, Texas")
    assert len(vec) == 1536
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-3


@pytest.mark.skipif(not _has_key(), reason="No OPENAI_API_KEY/OPENROUTER_API_KEY in env")
def test_openai_compat_embedder_pack_roundtrip():
    emb = OpenAICompatEmbedder(dim=1536)
    packed = emb.pack("hello world")
    assert len(packed) == 1536 * 4  # float32


def test_openai_compat_embedder_resize_logic_truncation():
    # Pure unit test of the static method, no network.
    vec = list(range(2000))
    out = OpenAICompatEmbedder._resize(vec, 1536)
    assert len(out) == 1536
    assert out[-1] == 1535
