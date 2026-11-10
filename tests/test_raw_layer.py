"""P8: raw-message layer — isolation, privacy, search, fusion, flag-off identity.

Throwaway store, no Ollama, no network: vectors are hash-derived and the reranker is
switched off in config.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mpmf.memory import MemoryManager        # noqa: E402
from mpmf_plus import raw_layer as R          # noqa: E402
from mpmf_plus import paths                   # noqa: E402
from mpmf_plus import provider as P           # noqa: E402

DIM = 1024


def vec(text):
    """Deterministic unit vector; texts sharing words share direction."""
    v = np.zeros(DIM, dtype=np.float32)
    for w in text.lower().split():
        h = int(hashlib.sha256(w.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0
    n = float(np.linalg.norm(v)) or 1.0
    return (v / n).tolist()


@pytest.fixture()
def store(tmp_path, monkeypatch):
    p = str(tmp_path / "t.mpmf")
    m = MemoryManager(p, vector_dimensions=DIM)
    with m.write_txn():
        m.create_node(name="memfact_a_0_0", node_type=1, plane_flags=1,
                      text="The user drives a green Subaru.", summary="x",
                      wiki={"_ns": "memory", "entity": "car"}, vector=R._pack(vec("drives green subaru")))
    cfg = tmp_path / "recall.config.json"
    cfg.write_text(json.dumps({"raw_layer": True, "raw_rerank": False, "raw_min_cos": 0.2,
                               "store_path": p, "store_mode": "attach"}))
    monkeypatch.setenv("MPMF_RECALL_CONFIG", str(cfg))
    monkeypatch.setenv("MPMF_HOME", str(tmp_path))
    monkeypatch.setenv("MPMF_STORE_PATH", p)
    prev = P.configure(PATH=p)
    P._invalidate_open_cache()
    yield p
    P.configure(**prev)
    P._invalidate_open_cache()


def test_default_is_off():
    assert paths.DEFAULTS["raw_layer"] is False


def test_storable_filters():
    assert R.storable("I bought a green Subaru Outback last spring and love it")
    assert not R.storable("too short")
    assert not R.storable("my key is sk-or-v1-" + "a" * 40 + " please keep it safe, thanks")
    assert not R.storable("email me at someone@example.com about the Subaru purchase soon")
    assert not R.storable("[System note: this is host injected text that is long enough to pass]")


def test_add_turn_isolated_dedup_and_search(store):
    msg = "I bought a green Subaru Outback last spring and I really love driving it"
    assert R.add_turn(msg, conv_id="s1", msg_index=3, embed_fn=vec) == "stored"
    assert R.add_turn(msg, conv_id="s1", msg_index=3, embed_fn=vec) == "dup"
    assert R.add_turn("short", embed_fn=vec) == "skipped"
    m = MemoryManager(store, vector_dimensions=DIM); m.load()
    ns = [(m.get_wiki(n) or {}).get("_ns") for n in m.all_node_ids()]
    assert ns.count("raw") == 1 and ns.count("memory") == 1
    # the default recall asks for namespaces ("memory",); the raw node carries _ns == "raw"
    rawid = [n for n in m.all_node_ids() if (m.get_wiki(n) or {}).get("_ns") == "raw"][0]
    assert m.get_name(rawid).startswith("rawmsg_") and "entity" not in (m.get_wiki(rawid) or {})
    hits = R.search("green Subaru Outback", vec("green Subaru Outback"), mem=m, k=3, min_cos=0.1, rerank=False)
    assert len(hits) == 1 and hits[0]["text"] == msg and hits[0]["conv"] == "s1"
    assert R.search("zzz qqq", vec("zzz qqq"), mem=m, k=3, min_cos=0.5, rerank=False) == []


def test_backfill_refuses_live_and_writes_elsewhere(store, tmp_path, monkeypatch):
    live = tmp_path / "live.mpmf"
    live.write_bytes(b"")
    monkeypatch.setenv("MPMF_STORE_PATH", str(live))
    with pytest.raises(PermissionError):
        R.backfill([{"text": "x" * 50}], [vec("x")], str(live))
    monkeypatch.setenv("MPMF_STORE_PATH", str(tmp_path / "other.mpmf"))
    items = [{"text": "I moved to Lisbon in March 2023 for a new job at a small studio", "date": "2023-03-01", "conv": "c", "idx": 0},
             {"text": "I moved to Lisbon in March 2023 for a new job at a small studio", "date": "2023-03-01", "conv": "c", "idx": 0},
             {"text": "short", "date": "2023-03-01"}]
    r = R.backfill(items, [vec("a")] * 3, store)
    assert r == {"written": 1, "dup": 1, "skipped": 1}


def test_render_is_separate_and_capped():
    hits = [{"text": "a" * 900, "date": "2024-01-02"}, {"text": "b" * 900, "date": "2024-01-03"}]
    out = R.render(hits, item_chars=100, block_chars=1500)
    assert out.startswith("# Raw messages") and "2024-01-02: " in out and out.count("…") == 2
    small = R.render(hits, 100, 200)       # header + first item fit, the second is dropped
    assert "2024-01-02" in small and "2024-01-03" not in small and len(small) <= 200


def test_prefetch_section_flag_off_is_empty(store, tmp_path):
    cfg = json.loads(open(os.environ["MPMF_RECALL_CONFIG"]).read())
    cfg["raw_layer"] = False
    open(os.environ["MPMF_RECALL_CONFIG"], "w").write(json.dumps(cfg))
    assert R.prefetch_section("green subaru", vec("green subaru")) == ""
    assert not R.enabled()


def test_adapter_flag_off_leaves_prefetch_and_sync_untouched(store, monkeypatch):
    import inspect
    from mpmf_plus import hermes_adapter as H
    cfg = json.loads(open(os.environ["MPMF_RECALL_CONFIG"]).read())
    cfg["raw_layer"] = False
    open(os.environ["MPMF_RECALL_CONFIG"], "w").write(json.dumps(cfg))
    assert H._raw_flag() is False
    called = []
    monkeypatch.setattr(R, "prefetch_section", lambda *a, **k: called.append(1) or "X")
    monkeypatch.setattr(R, "add_turn", lambda *a, **k: called.append(2) or "stored")
    monkeypatch.setattr(P, "set_ingest_session", lambda *a, **k: None)
    monkeypatch.setattr(P, "_ingest_turn", lambda *a, **k: "n")
    prov = H.MPMFPlusProvider()
    prov.sync_turn("u" * 80, "a" * 80, session_id="s")
    assert called == []
    cfg["raw_layer"] = True
    open(os.environ["MPMF_RECALL_CONFIG"], "w").write(json.dumps(cfg))
    prov = H.MPMFPlusProvider()
    prov.sync_turn("u" * 80, "a" * 80, session_id="s")
    assert called == [2]
