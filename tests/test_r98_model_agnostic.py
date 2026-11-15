"""r98: model-agnostic audit guards.

Nothing in the shipped core may decide behaviour from a model NAME. These tests pin the
three capabilities MPMF actually needs to the reply: which field carries the reasoning
trace, what the call cost, and whether the endpoint accepted a request option.
"""
import json
import re
import sqlite3
import sys
import types
from pathlib import Path

import pytest

from mpmf_plus import llm_caps

SRC = Path(__file__).resolve().parents[1] / "src" / "mpmf_plus"

#: name-based model logic: `"slug" in model`, `model.startswith("slug")`, `model == "slug"`.
NAME_LOGIC = re.compile(
    r"""if\s+["'][\w.:/+-]+["']\s+in\s+(?:model|mdl|args\.model)\b"""
    r"""|(?:model|mdl)\.startswith\(\s*["']"""
    r"""|(?:model|mdl)\s*==\s*["'][\w.:/+-]+["']""")


# ── llm_caps: capability from the reply ────────────────────────────────────
def test_reasoning_text_reads_every_field_name_a_provider_uses():
    assert llm_caps.reasoning_text({"reasoning": "a"}) == "a"
    assert llm_caps.reasoning_text({"reasoning_content": "b"}) == "b"
    assert llm_caps.reasoning_text({"reasoning_text": "c"}) == "c"
    assert llm_caps.reasoning_text({"thinking": "d"}) == "d"
    # OpenRouter's list-of-parts shape
    assert llm_caps.reasoning_text(
        {"reasoning_details": [{"type": "reasoning.text", "text": "e1"},
                               {"type": "reasoning.text", "text": "e2"}]}) == "e1\ne2"
    # dict shape
    assert llm_caps.reasoning_text({"reasoning": {"content": "f"}}) == "f"
    # no trace at all
    assert llm_caps.reasoning_text({"content": "x"}) == ""
    assert llm_caps.reasoning_text({}) == ""
    assert llm_caps.reasoning_text(None) == ""


def test_usage_cost_prefers_what_the_provider_reported():
    assert llm_caps.usage_cost({"usage": {"cost": 0.0}}) == 0.0     # free model, still reported
    assert llm_caps.usage_cost({"usage": {"cost": 0.0123}}) == 0.0123
    assert llm_caps.usage_cost({"usage": {"total_cost": 2}}) == 2.0
    assert llm_caps.usage_cost({"usage": {"prompt_tokens": 10}}) is None
    assert llm_caps.usage_cost({}) is None


import json as _json
import requests as _requests


class _Resp:
    """Minimal requests.Response stand-in."""

    def __init__(self, status, body):
        self.status_code, self.text, self._body = status, body, body
        self.headers = {}

    def json(self):
        return _json.loads(self._body)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _requests.HTTPError("%d: %s" % (self.status_code, self.text))


def test_option_rejected_only_for_body_errors_that_name_the_option():
    assert llm_caps.option_rejected(400, '{"error":"Unsupported parameter: reasoning"}', "reasoning")
    assert llm_caps.option_rejected(422, "reasoning: unknown field", "reasoning")
    assert not llm_caps.option_rejected(429, "rate limited", "reasoning")
    assert not llm_caps.option_rejected(400, "invalid api key", "reasoning")
    assert not llm_caps.option_rejected(500, "reasoning backend exploded", "reasoning")


def test_post_drops_a_rejected_option_retries_and_remembers_it():
    llm_caps.reset_cache()
    seen = []

    def post(url, json=None, headers=None, timeout=None):
        seen.append(dict(json))
        if "reasoning" in json:
            return _Resp(400, '{"error":{"message":"Unsupported parameter: reasoning"}}')
        return _Resp(200, '{"choices":[]}')

    payload = {"model": "whatever", "reasoning": {"enabled": False}, "max_tokens": 1}
    r, sent, dropped = llm_caps.post_with_option_fallback(
        post, "http://x/c", payload, timeout=5, endpoint="http://x")
    assert r.status_code == 200 and dropped is True
    assert "reasoning" not in sent and len(seen) == 2
    # learned: the next call does not send it at all (one round-trip)
    r2, sent2, _ = llm_caps.post_with_option_fallback(
        post, "http://x/c", payload, timeout=5, endpoint="http://x")
    assert "reasoning" not in sent2 and len(seen) == 3
    assert llm_caps.option_supported("http://x", "reasoning") is False
    llm_caps.reset_cache()
    assert llm_caps.option_supported("http://x", "reasoning") is True


def test_post_keeps_the_option_when_the_endpoint_accepts_it():
    llm_caps.reset_cache()
    seen = []

    def post(url, json=None, headers=None, timeout=None):
        seen.append(dict(json))
        return _Resp(200, "{}")

    r, sent, dropped = llm_caps.post_with_option_fallback(
        post, "http://y/c", {"reasoning": {"enabled": False}}, endpoint="http://y")
    assert r.status_code == 200 and dropped is False and "reasoning" in seen[0]


# ── call sites now use it ──────────────────────────────────────────────────
def test_llm_complete_retries_without_the_option_on_an_unsupported_parameter(monkeypatch):
    from mpmf_plus import memory
    llm_caps.reset_cache()
    calls = []

    def post(url, json=None, headers=None, timeout=None):
        calls.append(dict(json))
        if "reasoning" in json:
            return _Resp(422, '{"error":"reasoning: unsupported parameter"}')
        return _Resp(200, _json.dumps({"choices": [{"message": {"content": "ok"},
                                                   "finish_reason": "stop"}],
                                      "usage": {"cost": 0.0}, "model": "m"}))
    monkeypatch.setattr(memory.requests, "post", post)
    monkeypatch.setattr(memory, "_api_key", lambda: "k")
    out = memory._llm_complete("prompt", reasoning=False)
    assert out["content"] == "ok"
    assert "reasoning" in calls[0] and "reasoning" not in calls[1]


def test_bench_chat_switches_reasoning_off_only_after_a_trace_only_reply(monkeypatch):
    from mpmf_plus.bench import llm as bench
    llm_caps.reset_cache()
    calls = []

    def post(url, json=None, headers=None, timeout=None):
        calls.append(dict(json))
        if len(calls) == 1:
            # content empty, trace present -> the reply itself says where the budget went
            return _Resp(200, _json.dumps({"choices": [{"message": {
                "content": None, "reasoning": "long trace"}, "finish_reason": "length"}],
                "usage": {"cost": 0.0}, "model": "zzz"}))
        return _Resp(200, _json.dumps({"choices": [{"message": {"content": "answer"},
                                                   "finish_reason": "stop"}],
                                      "usage": {"cost": 0.0}, "model": "zzz"}))
    monkeypatch.setattr(_requests, "post", post)
    monkeypatch.setattr(bench, "_api_key", lambda: "k")
    out = bench.chat("q", model="some-vendor/unknown-model")
    assert out["text"] == "answer"
    assert "reasoning" not in calls[0]                       # no name-based pre-emptive switch
    assert calls[1]["reasoning"] == {"enabled": False}       # learned from the reply


def test_extractor_survives_an_endpoint_without_a_reasoning_option(monkeypatch):
    from mpmf_plus import provider
    llm_caps.reset_cache()
    calls = []

    def post(url, json=None, headers=None, timeout=None):
        calls.append(dict(json))
        if "reasoning" in json:
            return _Resp(400, '{"error":"Unsupported parameter: reasoning"}')
        return _Resp(200, _json.dumps({"choices": [{"message": {"content": '{"facts": []}'},
                                                   "finish_reason": "stop"}],
                                      "usage": {}}))
    monkeypatch.setattr(provider.requests, "post", post)
    monkeypatch.setattr(provider, "_api_key", lambda: "k")
    out = provider.call_extractor("a fact", model="vendor/some-model")
    assert out["facts"] == []
    assert "reasoning" in calls[0] and "reasoning" not in calls[1]


def test_capture_reads_reasoning_from_any_field(tmp_path, monkeypatch):
    """A model that writes reasoning_content (not reasoning) is no longer 'no_reasoning'."""
    from mpmf_plus import capture
    stored = []
    fake = types.ModuleType("athena_thinking")
    fake.store_thinking = lambda **kw: stored.append(kw)
    monkeypatch.setitem(sys.modules, "athena_thinking", fake)
    db = tmp_path / "state.db"
    c = sqlite3.connect(db)
    c.execute("create table sessions (id text primary key, source text, model text)")
    c.execute("create table messages (id integer primary key, session_id text, role text, "
              "content text, tool_calls text, reasoning text, reasoning_content text, "
              "reasoning_details text, timestamp real)")
    c.execute("insert into sessions values ('s','telegram','anthropic/claude-sonnet-5.5')")
    now = __import__("time").time()
    c.execute("insert into messages (session_id, role, content, timestamp) values ('s','user','q',?)",
              (now - 100,))
    c.execute("insert into messages (session_id, role, content, reasoning_content, timestamp) "
              "values ('s','assistant','a',?,?)", ("R" * 300, now - 50))
    c.commit()
    c.close()
    monkeypatch.setattr(capture, "STATE_DB", str(db))
    monkeypatch.setattr(capture, "STATE_PATH", str(tmp_path / "st.json"))
    cfg = dict(capture.DEFAULTS, thinking_capture_enabled=True,
               backup_before_first_write=False, since="2000-01-01T00:00:00")
    out = capture.write_once(cfg)
    assert out["written"] == 1 and out["skipped"] == {}
    assert stored[0]["model"] == "anthropic/claude-sonnet-5.5"


def test_store_trace_tolerates_an_older_shim_without_model(monkeypatch):
    from mpmf_plus import capture
    got = []

    def legacy(**kw):
        if "model" in kw:
            raise TypeError("store_thinking() got an unexpected keyword argument 'model'")
        got.append(kw)

    fake = types.ModuleType("athena_thinking")
    fake.store_thinking = legacy
    monkeypatch.setitem(sys.modules, "athena_thinking", fake)
    capture._store_trace(context="c", thinking="t", conclusion="x", model="m")
    assert got and "model" not in got[0]


def test_no_name_based_model_logic_left_in_the_core():
    hits = []
    for p in sorted(SRC.rglob("*.py")):
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if NAME_LOGIC.search(line):
                hits.append("%s:%d: %s" % (p.relative_to(SRC.parent.parent), i, line.strip()))
    assert hits == [], "model-name logic in the core:\n" + "\n".join(hits)
