"""Round 83 — mpmf_plus.ingest: one turn, one txn, one log line, full provenance."""
import hashlib
import json
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mpmf.memory import MemoryManager                     # noqa: E402
from mpmf_plus import ingest as ing                        # noqa: E402
from mpmf_plus import store as store_mod                   # noqa: E402

DIM = 16


def fake_vec(text, dim=DIM):
    """Deterministic unit-ish vector, no Ollama."""
    h = hashlib.sha256(text.encode()).digest()
    vals = [((h[i % len(h)] / 255.0) - 0.5) for i in range(dim)]
    n = sum(v * v for v in vals) ** 0.5 or 1.0
    return [v / n for v in vals]


class FakeMem:
    """MPMFMemory-shaped shim over a real MemoryManager (so write_txn is real)."""

    def __init__(self, path):
        self.mem = MemoryManager(path, vector_dimensions=DIM)
        self.mem.load()

    def write_txn(self, timeout=60):
        return self.mem.write_txn(timeout=timeout)

    def embed(self, text):
        return fake_vec(text)


def reply(*facts):
    # r83c: msg_index is the index inside the window the extractor saw; these
    # fixtures pass no context, so the current (and only) message is 0. A fact
    # labelled for any other message is dropped as context.
    return json.dumps({"facts": [dict(
        msg_index=f.get("i", 0), entity=f.get("entity", "car"),
        fact_sentence=f["s"], subject="user", predicate="drives",
        object=f.get("obj", ""), evidence_sentence=f.get("ev", f["s"]),
        fact_date=f.get("date")) for f in facts]})


@pytest.fixture()
def store(tmp_path):
    return FakeMem(str(tmp_path / "in.mpmf"))


def test_ingest_writes_facts_with_full_provenance(store, tmp_path):
    log = str(tmp_path / "store.log")
    res = ing.ingest("I drive a 2024 Volvo V60.", said_at="2026-11-03T01:00:00Z",
                     conv_id="conv1234567890", msg_index=7, store=store,
                     call_model=lambda body: reply({"s": "The user drives a 2024 Volvo V60.",
                                                    "entity": "car", "ev": "I drive a 2024 Volvo V60."}),
                     embed_fn=fake_vec, log_path=log, embed_dim=DIM)
    assert res["status"] == "ok" and len(res["written"]) == 1
    w = res["written"][0]
    assert w["kind"] in ("have", "change", "intent", "interest")
    mm = MemoryManager(str(tmp_path / "in.mpmf"), vector_dimensions=DIM)
    mm.load()
    nid = mm.get_node_id_by_name(w["name"])
    wiki = mm.get_wiki(nid)
    for f in ing.FACT_FIELDS:
        assert wiki.get(f) not in (None, ""), "missing %s" % f
    assert wiki["_conv_id"] == "conv1234567890"
    assert wiki["_msg_index"] == "7"
    assert wiki["_evidence"].startswith("I drive")
    assert wiki["_fact_date"] == "2026-11-03"          # said_at date when the extractor has none
    assert wiki["_said_at"] == "2026-11-03T01:00:00Z"
    assert len(mm.get_vector(nid)) == DIM
    lines = [l for l in open(log).read().splitlines() if l.strip()]
    assert len(lines) == 1 and "ingest_ms=" in lines[0] and "facts=1" in lines[0]


def test_ingest_dedups_a_repeated_fact(store, tmp_path):
    same = "The user drives a 2024 Volvo V60."
    call = lambda body: reply({"s": same, "entity": "car"})     # noqa: E731
    log = str(tmp_path / "store.log")
    first = ing.ingest("I drive a 2024 Volvo V60.", store=store, conv_id="c1",
                       call_model=call, embed_fn=fake_vec, log_path=log, embed_dim=DIM)
    second = ing.ingest("I drive a 2024 Volvo V60.", store=store, conv_id="c2",
                        call_model=call, embed_fn=fake_vec, log_path=log, embed_dim=DIM)
    assert len(first["written"]) == 1
    assert second["written"] == [] and second["status"] == "empty"
    assert second["skipped"][0]["action"] in ("exact_match", "near_dup")
    mm = MemoryManager(str(tmp_path / "in.mpmf"), vector_dimensions=DIM)
    mm.load()
    assert len([i for i in mm.all_node_ids()
                if (mm.get_text(i) or "").startswith("The user drives")]) == 1


def test_ingest_stamps_embedder_and_refuses_a_foreign_stamp(store, tmp_path):
    call = lambda body: reply({"s": "The user drives a 2024 Volvo V60."})   # noqa: E731
    ing.ingest("I drive a Volvo.", store=store, call_model=call, embed_fn=fake_vec,
               log_path="", embed_dim=DIM, embed_model="fake-embed")
    mm = MemoryManager(str(tmp_path / "in.mpmf"), vector_dimensions=DIM)
    mm.load()
    assert store_mod.read_embedder(mm)["embed_model"] == "fake-embed"
    with pytest.raises(Exception):
        ing.ingest("I drive a Volvo.", store=store, call_model=call, embed_fn=fake_vec,
                   log_path="", embed_dim=DIM, embed_model="other-embed")


def test_ingest_rejects_a_zero_vector_loudly(store, tmp_path):
    call = lambda body: reply({"s": "The user drives a 2024 Volvo V60."})   # noqa: E731
    with pytest.raises(ValueError, match="all-zero"):
        ing.ingest("x", store=store, call_model=call,
                   embed_fn=lambda t: [0.0] * 16, log_path="", embed_dim=DIM,
                   embed_model="fake-embed")


def test_context_narrows_mode_to_hypothetical(store, tmp_path):
    # index 2 = the current message (2 context messages precede it)
    call = lambda body: reply({"s": "The user wants a boat.", "i": 2})      # noqa: E731
    res = ing.ingest("I'd love a boat, if I had the money.",
                     context=[{"role": "user", "text": "Suppose I win the lottery"},
                              {"role": "assistant", "text": "Then a boat?"}],
                     store=store, call_model=call, embed_fn=fake_vec,
                     log_path="", embed_dim=DIM, embed_model="fake-embed")
    assert res["written"][0]["mode"] == "hypothetical"
