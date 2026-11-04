"""R87b E2E: plugin.sync_turn -> ingest -> prefetch must surface the fact.

Uses a throwaway store and a canned extraction reply (no network).
"""
import importlib.util
import json
import sys

import pytest

from mpmf_plus import memory as M
from mpmf_plus import provider as P


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        "mpmf_plus_user_r87b", "/root/.hermes/plugins/mpmf_plus_user/__init__.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mpmf_plus_user_r87b"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_sync_turn_ingests_full_user_text_to_memory(tmp_path, monkeypatch):
    store = str(tmp_path / "e2e.mpmf")
    M.configure_store(path=store, mode="create")
    P.configure(PATH=store)
    monkeypatch.setattr(P, "STORE_LOG", str(tmp_path / "store.log"))

    reply = json.dumps({"facts": [{
        "fact_sentence": "The user set the Round 87b end-to-end test cap to 30 dollars.",
        "entity": "user", "msg_index": 0,
        "evidence_sentence": "the round 87b end-to-end test cap is 30 dollars",
        "kind": "have", "mode": "actual", "fact_date": "2026-11-04"}]})
    import mpmf_plus.ingest as I
    monkeypatch.setattr(I, "_default_call", lambda *a, **k: reply)

    plugin = _load_plugin()
    monkeypatch.setattr(plugin, "_get_wrapper_module", lambda: P)
    cfg = {"store_path": store, "auto_write": True, "top_k": 8, "max_chars": 3500}
    prov = plugin.MPMFPlusProvider(cfg)
    prov.initialize("r87b-e2e")

    user = "The round 87b end-to-end test cap is 30 dollars. " + ("filler " * 200)
    prov.sync_turn(user, "Athena acknowledgement that must not become a user fact.", session_id="r87b_e2e")

    mm = M._open().mem
    facts = []
    for nid in mm.all_node_ids():
        w = mm.get_wiki(nid) or {}
        if w.get("_source") == "ingest":
            facts.append((int(nid), mm.get_name(nid), w, mm.get_text(nid) or ""))
    assert facts, "sync_turn wrote no ingest facts"
    nid, name, wiki, text = facts[-1]
    assert name.startswith("memfact_"), name
    assert wiki.get("_ns") == "memory", wiki.get("_ns")
    assert "87b" in text.lower()
    assert "Athena acknowledgement" not in text

    block = prov.prefetch("What did the user set the Round 87b end-to-end test cap to?")
    assert "30 dollars" in block or "87b" in block.lower(), block[:400]
