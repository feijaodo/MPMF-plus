"""r84: mpmf_plus.provider is the moved Athena provider.

The move is mechanical, so the tests pin the two things that could drift:
the configure() contract and behavioural equality with the module it came from
(when that module is present on this box).
"""
import importlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.expanduser("~/.mpmf"))

from mpmf_plus import provider  # noqa: E402


def test_public_api_surface():
    for fn in ("store", "recall", "get", "stats", "link", "embed_query",
               "recall_cognition", "recall_cognition_cosine", "re_enrich_pending",
               "configure", "_provider_config"):
        assert hasattr(provider, fn), "provider lost %s in the move" % fn


def test_configure_sets_and_returns_previous():
    prev = provider.configure(VECTOR_DIM=1024, DEFAULT_K=5)
    assert prev["VECTOR_DIM"] == 1024
    provider.configure(DEFAULT_K=7)
    assert provider.DEFAULT_K == 7
    provider.configure(DEFAULT_K=prev["DEFAULT_K"])


def test_configure_rejects_unknown_key():
    with pytest.raises(KeyError):
        provider.configure(NOT_A_SETTING=1)


def test_config_file_drives_paths(tmp_path, monkeypatch):
    cfg = tmp_path / "provider.config.json"
    cfg.write_text(json.dumps({"PATH": str(tmp_path / "custom.mpmf"),
                               "STORE_LOG": str(tmp_path / "store.log")}))
    monkeypatch.setenv("MPMF_PROVIDER_CONFIG", str(cfg))
    mod = importlib.reload(provider)
    try:
        assert mod.PATH == str(tmp_path / "custom.mpmf")
        assert mod.STORE_LOG == str(tmp_path / "store.log")
        assert mod.PATH_LEGACY.endswith(".v1-legacy")
    finally:
        monkeypatch.delenv("MPMF_PROVIDER_CONFIG", raising=False)
        importlib.reload(provider)


def test_entry_module_reexports(tmp_path, monkeypatch):
    entry = os.path.expanduser("~/.mpmf/provider_entry.py")
    if not os.path.exists(entry):
        pytest.skip("Athena entry module not on this box")
    import importlib.util
    spec = importlib.util.spec_from_file_location("provider_entry_t", entry)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert callable(m.recall) and callable(m.store)
    assert m.CODE_REV


def test_same_results_as_athena_memory_on_a_small_store(tmp_path, monkeypatch):
    """Both modules, same store, same query vector -> same ranked names."""
    old_path = os.path.expanduser("~/.mpmf/athena_memory.py")
    if not os.path.exists(old_path):
        pytest.skip("legacy module not on this box")
    import importlib.util
    from mpmf.memory import MemoryManager

    store = str(tmp_path / "tiny.mpmf")
    m = MemoryManager(store, vector_dimensions=1024)
    vec = [0.0] * 1024
    vec[0] = 1.0
    with m.write_txn():
        for i, t in enumerate(["boat purchase", "car repair", "boat trailer", "garden hose"]):
            v = list(vec)
            v[i + 1] = 1.0
            m.create_node(name="n%d" % i, node_type=1, plane_flags=1, text=t,
                          summary=t, wiki={"_ns": "memory"},
                          vector=__import__("numpy").asarray(v, dtype="<f4").tobytes())

    spec = importlib.util.spec_from_file_location("am_r84", old_path)
    old = importlib.util.module_from_spec(spec)
    sys.modules["am_r84"] = old
    spec.loader.exec_module(old)

    qv = [0.0] * 1024
    qv[0] = 1.0
    qv[1] = 0.5
    out = {}
    for label, mod in (("old", old), ("new", provider)):
        mod.configure(PATH=store) if label == "new" else setattr(mod, "PATH", store)
        if hasattr(mod, "_invalidate_open_cache"):
            mod._invalidate_open_cache()
        monkeypatch.setattr(mod, "embed_query", lambda text, _m=mod: list(qv), raising=False)
        monkeypatch.setattr(mod, "_embed_query_cap", lambda: 1200, raising=False)
        hits = mod.recall("boat", top_n=4, use_rerank=False)
        out[label] = [h.get("name") for h in hits]
    assert out["old"] == out["new"], out
