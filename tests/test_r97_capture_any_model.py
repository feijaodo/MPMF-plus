"""r97: thinking capture has no model allowlist.

Any model's FINAL reply that carries reasoning >= min chars is captured; everything else is
skipped with a reason, counted per model, and the daily check prints the counts.
Throwaway state.db / state file / fake store_thinking; nothing live is touched.
"""
import json
import sqlite3
import sys
import time
import types

import pytest

from mpmf_plus import capture, health

REASON = "R" * 300          # >= min_reasoning_chars (200)
SONNET = "anthropic/claude-sonnet-5.5"
FABLE = "anthropic/claude-fable-5.2"
DSFLASH = "deepseek/deepseek-v4.1-flash-1017"


def _mk_db(path, replies):
    """replies: list of (model, reasoning); one session each: user turn then a final reply."""
    c = sqlite3.connect(path)
    c.execute("create table sessions (id text primary key, source text, model text)")
    c.execute("create table messages (id integer primary key, session_id text, role text, "
              "content text, tool_calls text, reasoning text, timestamp real)")
    now = time.time()
    for i, (model, rz) in enumerate(replies):
        sid = f"s{i}"
        c.execute("insert into sessions values (?,?,?)", (sid, "telegram", model))
        c.execute("insert into messages (session_id, role, content, timestamp) values (?,?,?,?)",
                  (sid, "user", f"question {i}", now - 100))
        c.execute("insert into messages (session_id, role, content, tool_calls, reasoning, timestamp) "
                  "values (?,?,?,?,?,?)", (sid, "assistant", f"answer {i}", None, rz, now - 50))
    c.commit()
    c.close()


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    stored = []
    fake = types.ModuleType("athena_thinking")
    fake.store_thinking = lambda **kw: stored.append(kw)
    monkeypatch.setitem(sys.modules, "athena_thinking", fake)
    monkeypatch.setenv("MPMF_HOME", str(tmp_path))
    monkeypatch.setenv("MPMF_HEALTH_DIR", str(tmp_path / "health"))
    monkeypatch.setattr(capture, "STATE_DB", str(db))
    monkeypatch.setattr(capture, "STATE_PATH", str(tmp_path / "health" / "thinking_capture_state.json"))
    cfg = dict(capture.DEFAULTS, thinking_capture_enabled=True, backup_before_first_write=False,
               since="2000-01-01T00:00:00")
    return types.SimpleNamespace(db=db, stored=stored, cfg=cfg, tmp=tmp_path)


def test_no_model_allowlist_in_defaults():
    assert "models" not in capture.DEFAULTS


def test_stale_models_key_in_config_file_is_ignored(tmp_path, monkeypatch):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"models": ["only-this"], "daily_cap": 7}))
    monkeypatch.setattr(capture, "CONFIG_PATH", str(p))
    cfg = capture.load_config()
    assert "models" not in cfg and cfg["daily_cap"] == 7


def test_sonnet_fable_and_deepseek_replies_with_reasoning_are_all_captured(env):
    # distinct reasoning per reply so md5 dedup keeps all three
    _mk_db(env.db, [(SONNET, "s" + REASON), (FABLE, "f" + REASON), (DSFLASH, "d" + REASON)])
    out = capture.write_once(env.cfg)
    assert out["written"] == 3
    assert out["written_by_model"] == {SONNET: 1, FABLE: 1, DSFLASH: 1}
    assert {s["thinking"][0] for s in env.stored} == {"s", "f", "d"}
    assert out["skipped"] == {}


def test_reply_without_reasoning_is_skipped_with_reason_and_model(env):
    _mk_db(env.db, [(SONNET, None), (FABLE, ""), (DSFLASH, "short"), (SONNET, "ok" + REASON)])
    out = capture.write_once(env.cfg)
    assert out["written"] == 1
    assert out["skipped"]["no_reasoning"] == {SONNET: 1, FABLE: 1}
    assert out["skipped"]["reasoning_too_short"] == {DSFLASH: 1}
    # persisted per day, and logged with the model name
    st = capture.load_state()
    day = next(iter(st["skips"].values()))
    assert day["no_reasoning"] == {SONNET: 1, FABLE: 1}
    log = (env.tmp / "health" / "thinking_capture.log").read_text()
    assert "no_reasoning=" in log and f"{SONNET}:1" in log and f"{FABLE}:1" in log


def test_duplicate_reasoning_is_skipped_and_counted(env):
    _mk_db(env.db, [(SONNET, REASON), (FABLE, REASON)])
    out = capture.write_once(env.cfg)
    assert out["written"] == 1
    assert out["skipped"] == {"duplicate_reasoning": {FABLE: 1}}


def test_cron_source_is_excluded_with_reason(env):
    _mk_db(env.db, [(SONNET, REASON)])
    c = sqlite3.connect(env.db)
    c.execute("update sessions set source='cron'")
    c.commit()
    c.close()
    out = capture.write_once(env.cfg)
    assert out["written"] == 0 and out["skipped"] == {"excluded_source": {SONNET: 1}}


def test_daily_check_reports_skip_counts(env):
    _mk_db(env.db, [(SONNET, None), (FABLE, "f" + REASON)])
    capture.write_once(env.cfg)
    lines = health.capture_skip_report(state=capture.STATE_PATH)
    assert len(lines) == 1
    assert "captured=1" in lines[0]
    assert "no_reasoning=1" in lines[0] and SONNET in lines[0]
    assert health.daily_info.__name__ == "daily_info"


def test_daily_check_with_no_state_file_says_so(tmp_path):
    lines = health.capture_skip_report(state=tmp_path / "nope.json")
    assert len(lines) == 1 and "no state file" in lines[0]
