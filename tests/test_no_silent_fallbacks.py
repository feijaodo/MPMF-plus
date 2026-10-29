"""Round 59d step 3 — the package must never fail silently.

Four failure modes, each must raise a typed error:
  * store path missing          -> StoreNotFoundError
  * store path unreadable       -> StoreNotFoundError
  * recall config missing/bad    -> ConfigError
  * embedder dead/empty          -> EmbedderError

The point is that none of these may degrade into "empty result" or "another file".
"""
import json
import os

import pytest

from mpmf_plus import block, memory
from mpmf_plus.errors import ConfigError, EmbedderError, StoreNotFoundError


# ---- store -----------------------------------------------------------------
def test_missing_store_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "PATH", str(tmp_path / "does_not_exist.mpmf"))
    with pytest.raises(StoreNotFoundError) as e:
        memory._open()
    assert "not found" in str(e.value)


def test_unreadable_store_raises(tmp_path, monkeypatch):
    p = tmp_path / "unreadable.mpmf"
    p.write_bytes(b"not a store")
    p.chmod(0o000)
    monkeypatch.setattr(memory, "PATH", str(p))
    try:
        with pytest.raises(StoreNotFoundError):
            memory._open()
    finally:
        p.chmod(0o600)


# ---- config ----------------------------------------------------------------
def test_missing_config_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(block, "RECALL_CFG_PATH", tmp_path / "nope.json")
    with pytest.raises(ConfigError):
        block.render_cfg()


def test_invalid_json_config_raises(tmp_path, monkeypatch):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    monkeypatch.setattr(block, "RECALL_CFG_PATH", p)
    with pytest.raises(ConfigError):
        block.render_cfg()


def test_unknown_render_mode_raises(tmp_path, monkeypatch):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"memory_render": "fulltext"}))
    monkeypatch.setattr(block, "RECALL_CFG_PATH", p)
    with pytest.raises(ConfigError) as e:
        block.render_cfg()
    assert "memory_render" in str(e.value)


def test_valid_config_still_works(tmp_path, monkeypatch):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"memory_render": "full", "memory_item_chars": 1200,
                             "memory_block_chars": 3000}))
    monkeypatch.setattr(block, "RECALL_CFG_PATH", p)
    assert block.render_cfg()["mode"] == "full"


# ---- embedder --------------------------------------------------------------
def test_embedder_exception_raises(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("ollama down")
    monkeypatch.setattr(memory, "_embed_floats_local", boom)
    with pytest.raises(EmbedderError):
        memory.embed_query("hello")


def test_embedder_empty_vector_raises(monkeypatch):
    monkeypatch.setattr(memory, "_embed_floats_local", lambda *a, **k: [])
    with pytest.raises(EmbedderError) as e:
        memory.embed_query("hello")
    assert "empty vector" in str(e.value)
