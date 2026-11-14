"""x1 — JSON parse failure, retry once with higher max_tokens; replay dead-letter."""
import json
import os
import sys
import tempfile
import time

import pytest

sys_path_insert = 0
sys_path = ["src"]

def _prepend_sys_path():
    global sys_path_insert
    if sys_path_insert:
        return
    sys_path_insert = 1
    sys.path[:0] = sys_path

_prepend_sys_path()

from mpmf_plus import ingest as ing
from mpmf_plus import reingest as rei
from mpmf_plus import paths as _paths
from test_r83_ingest import FakeMem, fake_vec, reply, DIM


BAD_JSON = '{"facts": [{"fact_sentence": "x", ]}'  # missing closing brace + extra comma
GOOD = lambda: reply({"s": "The user drives a 2024 Volvo V60.", "entity": "car",
                      "ev": "I drive a 2024 Volvo V60."})
TEXT = "I drive a 2024 Volvo V60."


def _write_dead_letter(dl_path, conv="convX1", msg=0):
    row = {"conv": conv, "msg": msg, "text": TEXT,
           "error": "JSONDecodeError: Expecting value", "ts": "2026-01-01T00:00:00Z"}
    with open(dl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_retry_higher_max_tokens_succeeds(tmp_path, monkeypatch):
    """First call returns broken JSON; second call (with higher max_tokens) works."""
    calls = []

    def call(body, max_tokens=None):
        calls.append((body, max_tokens))
        if len(calls) == 1:
            return BAD_JSON   # first attempt fails
        return GOOD()         # retry succeeds

    store = FakeMem(str(tmp_path / "x1.mpmf"))
    dl = str(tmp_path / "dl.jsonl")
    res = ing.ingest(TEXT, conv_id="convX1", msg_index=0, store=store,
                     call_model=call, embed_fn=fake_vec, embed_dim=DIM,
                     log_path=str(tmp_path / "store.log"), dead_letter_path=dl)
    # exactly two calls, second with higher max_tokens
    assert len(calls) == 2
    assert calls[0][1] is None   # first call got None -> uses FIRST_MAX_TOKENS (2000)
    assert calls[1][1] == 4000   # retry got the config default
    assert res["status"] == "ok"
    assert len(res["written"]) == 1
    # no dead-letter written
    assert not os.path.exists(dl)


def test_retry_still_fails_goes_to_dead_letter(tmp_path, monkeypatch):
    """Both attempts return broken JSON -> dead-letter."""
    calls = []

    def call(body, max_tokens=None):
        calls.append((body, max_tokens))
        return BAD_JSON   # always fails

    store = FakeMem(str(tmp_path / "x2.mpmf"))
    dl = str(tmp_path / "dl.jsonl")
    with pytest.raises(Exception, match="unparseable"):
        ing.ingest(TEXT, conv_id="convX2", msg_index=0, store=store,
                   call_model=call, embed_fn=fake_vec, embed_dim=DIM,
                   log_path=str(tmp_path / "store.log"), dead_letter_path=dl)
    assert len(calls) == 2   # exactly one retry
    assert calls[0][1] is None
    assert calls[1][1] == 4000
    rows = [json.loads(l) for l in open(dl)]
    assert len(rows) == 1
    r = rows[0]
    assert r["conv"] == "convX2" and r["msg"] == 0 and r["text"] == TEXT
    assert "JSONDecodeError" in r["error"] and "retry max_tokens=4000" in r["error"]


def test_retry_override_via_ingest(tmp_path):
    """Caller can pass retry_max_tokens to override the config."""
    calls = []

    def call(body, max_tokens=None):
        calls.append((body, max_tokens))
        if len(calls) == 1:
            return BAD_JSON
        return GOOD()

    store = FakeMem(str(tmp_path / "x3.mpmf"))
    dl = str(tmp_path / "dl.jsonl")
    ing.ingest(TEXT, conv_id="convX3", msg_index=0, store=store,
               call_model=call, embed_fn=fake_vec, embed_dim=DIM,
               log_path=str(tmp_path / "store.log"), dead_letter_path=dl,
               retry_max_tokens=8000)
    assert len(calls) == 2
    assert calls[0][1] is None
    assert calls[1][1] == 8000   # overridden
    assert not os.path.exists(dl)


def test_retry_unchanged_when_budget_zero(tmp_path):
    """ingest_retry_max_tokens <= 0 means retry with same budget."""
    calls = []

    def call(body, max_tokens=None):
        calls.append((body, max_tokens))
        if len(calls) == 1:
            return BAD_JSON
        return GOOD()

    store = FakeMem(str(tmp_path / "x4.mpmf"))
    dl = str(tmp_path / "dl.jsonl")
    # set config to 0 -> retry unchanged
    (tmp_path / "mpmf.config.json").write_text('{"ingest_retry_max_tokens": 0}')
    os.environ["MPMF_RECALL_CONFIG"] = str(tmp_path / "mpmf.config.json")
    ing.ingest(TEXT, conv_id="convX4", msg_index=0, store=store,
               call_model=call, embed_fn=fake_vec, embed_dim=DIM,
               log_path=str(tmp_path / "store.log"), dead_letter_path=dl)
    assert len(calls) == 2
    assert calls[0][1] is None        # 2000
    assert calls[1][1] is None        # retry also got None -> 2000
    assert not os.path.exists(dl)


def test_reingest_idempotent_and_counts(tmp_path, monkeypatch):
    """Re-ingest writes facts, marks resolved, second run is a no-op."""
    store = FakeMem(str(tmp_path / "x5.mpmf"))
    dl = str(tmp_path / "dl.jsonl")
    log = str(tmp_path / "store.log")
    # one unresolved dead-letter row
    _write_dead_letter(dl, conv="rebel", msg=5)
    assert len(rei.unresolved(dead_letter_path=dl)) == 1

    def good_call(body, max_tokens=None):
        return GOOD()

    # first run
    out = rei.reingest(store, dl, log_path=log, call_model=good_call,
                       embed_fn=fake_vec, embed_dim=DIM)
    assert len(out) == 1 and out[0]["status"] == "ok" and out[0]["facts"] == 1
    assert rei.unresolved(dead_letter_path=dl) == []   # now resolved
    # resolved file should have one line
    resolved = [json.loads(l) for l in open(rei.resolved_file(dl))]
    assert len(resolved) == 1
    r = resolved[0]
    assert r["conv"] == "rebel" and r["msg"] == 5 and r["facts"] == 1

    # second run (idempotent)
    out2 = rei.reingest(store, dl, log_path=log, call_model=good_call,
                        embed_fn=fake_vec, embed_dim=DIM)
    assert out2 == []   # no unresolved rows -> nothing to do
    # resolved file still has one line (no duplicate)
    resolved2 = [json.loads(l) for l in open(rei.resolved_file(dl))]
    assert len(resolved2) == 1