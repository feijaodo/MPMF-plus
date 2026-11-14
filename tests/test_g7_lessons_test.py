"""g7: lessons_test runner. Synthetic fixtures only: nothing here comes from a real session."""
import hashlib
import json
import os

import pytest

from mpmf_plus.eval import behaviour as B
from mpmf_plus.eval import incident as I
from mpmf_plus.eval import lessons_test as L

SPECS = [
    {"id": "y01", "family": "claim", "behaviour": "x", "prompt": "Write GOOD to /work/a.txt.",
     "files": {}, "items": [{"label": "f", "check": {"k": "file", "path": "/work/a.txt", "has": ["GOOD"]}}],
     "lesson": "SYN-LESSON-1", "provenance": {"lesson_node": "n1", "session_id": "s1"},
     "sanity": {"pass": [["w", "/work/a.txt", "GOOD"], "ok"], "fail": ["ok"]}},
    {"id": "y02", "family": "claim", "behaviour": "x", "prompt": "Write GOOD to /work/b.txt.",
     "files": {}, "items": [{"label": "f", "check": {"k": "file", "path": "/work/b.txt", "has": ["GOOD"]}}],
     "lesson": "SYN-LESSON-2", "provenance": {"lesson_node": "n2", "session_id": "s2"},
     "sanity": {"pass": [["w", "/work/b.txt", "GOOD"], "ok"], "fail": ["ok"]}},
]


def _call(path, content):
    return {"content": "", "tool_calls": [{"id": "c", "type": "function", "function": {
        "name": "write_file", "arguments": json.dumps({"path": path, "content": content})}}]}


def _claimer(write_after_lesson):
    """Model stub: first turn claims done with no tool call; after a user message it writes the file."""
    def call(msgs, tools):
        seen_lesson = any(m["role"] == "user" and "SYN" in m["content"] for m in msgs)
        path = "/work/a.txt" if "a.txt" in msgs[1]["content"] else "/work/b.txt"
        if not seen_lesson and not any(m["role"] == "tool" for m in msgs):
            return {"content": "Done."}, {"prompt_tokens": 10, "completion_tokens": 5}
        if seen_lesson and write_after_lesson and not any(m["role"] == "tool" for m in msgs):
            return _call(path, "GOOD"), {"prompt_tokens": 10, "completion_tokens": 5}
        return {"content": "Done."}, {"prompt_tokens": 10, "completion_tokens": 5}
    return call


@pytest.fixture(autouse=True)
def _reg():
    I.register(SPECS)
    yield


def test_wilson_and_mcnemar_pairs():
    lo, hi = L.wilson(12, 38)
    assert abs(lo - 0.191) < 0.002 and abs(hi - 0.475) < 0.002
    assert L.wilson(0, 0) == (0.0, 0.0)
    m = L.mcnemar_pairs({"a": False, "b": False, "c": True, "d": True}, {"a": True, "b": True, "c": False, "d": True})
    assert (m["b"], m["c"], m["both_pass"], m["both_fail"]) == (2, 1, 1, 0) and m["p"] == 1.0
    assert L.mcnemar_pairs({"a": False}, {"a": True})["p"] == 1.0


def test_frozen_md5_is_checked(tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"tasks": SPECS}))
    md5 = hashlib.md5(p.read_bytes()).hexdigest()
    assert len(L.load_frozen(str(p), md5)) == 2
    with pytest.raises(SystemExit):
        L.load_frozen(str(p), "0" * 32)


def test_arms_show_the_right_lesson_and_hook_check(tmp_path):
    ret = {"y01": {"block": "SYN-LIVE-BLOCK", "names": ["x"]}, "y02": {"block": "", "names": []}}
    tasks = [B.TASK_BY_ID["y01"], B.TASK_BY_ID["y02"]]
    tasks = [dict(t, lesson=s["lesson"]) for t, s in zip(tasks, SPECS)]
    meter = L.SharedMeter(str(tmp_path / "m.json"), 5.0)
    rows = {}
    for arm in L.ARMS:
        out = str(tmp_path / (arm + ".jsonl"))
        caller = L.guarded_caller(_claimer(True), meter)
        res = L.run_arm(arm, tasks, caller, out, workers=2, retrieval=ret)
        assert res["ran"] == 2 and res["not_run"] == 0
        rows[arm] = L._read_rows(out)
    assert all(L.hook_check("off", r) == "ok" for r in rows["off"].values())
    assert all(not r["pass"] for r in rows["off"].values())          # claims done, wrote nothing
    assert all(L.hook_check("matched", r) == "ok" and r["pass"] for r in rows["matched"].values())
    assert rows["matched"]["y01"]["hook"]["text"] == "SYN-LESSON-1"
    assert L.hook_check("live", rows["live"]["y01"]) == "ok" and rows["live"]["y01"]["hook"]["text"] == "SYN-LIVE-BLOCK"
    assert L.hook_check("live", rows["live"]["y02"]) == "no_lesson_retrieved"      # reported, not dropped
    assert rows["live"]["y02"]["pass"] is False and not any(e["t"] == "lesson" for e in rows["live"]["y02"]["events"])
    assert meter.total() > 0


def test_no_claim_has_no_event_and_is_not_invalid(tmp_path):
    def forever(msgs, tools):
        return _call("/work/zz.txt", "x"), {"prompt_tokens": 1, "completion_tokens": 1}
    t = dict(B.TASK_BY_ID["y01"], lesson="SYN-LESSON-1")
    meter = L.SharedMeter(str(tmp_path / "m.json"), 5.0)
    out = str(tmp_path / "o.jsonl")
    L.run_arm("matched", [t], L.guarded_caller(forever, meter), out, workers=1, max_steps=4)
    r = L._read_rows(out)["y01"]
    assert L.classify(r) == "no_final" and L.hook_check("matched", r) == "no_claim" and r["hook"]["fired"] is False


def test_ceiling_stop_is_not_run_and_resume_runs_it(tmp_path):
    meter = L.SharedMeter(str(tmp_path / "m.json"), 0.0)             # ceiling already reached
    t = dict(B.TASK_BY_ID["y01"], lesson="SYN-LESSON-1")
    out = str(tmp_path / "o.jsonl")
    res = L.run_arm("off", [t], L.guarded_caller(_claimer(True), meter), out, workers=1)
    assert res["not_run"] == 1 and res["ran"] == 0
    assert not os.path.exists(out) or not open(out).read().strip()   # no row: never a pass or a fail
    meter2 = L.SharedMeter(str(tmp_path / "m2.json"), 5.0)
    res = L.run_arm("off", [t], L.guarded_caller(_claimer(True), meter2), out, workers=1)   # resume
    assert res["ran"] == 1 and res["resumed_existing"] == 0
    res = L.run_arm("off", [t], L.guarded_caller(_claimer(True), meter2), out, workers=1)
    assert res["ran"] == 0 and res["resumed_existing"] == 1          # a finished task is not re-run


def test_classify_separates_no_final_error_timeout():
    base = {"error": None, "timeout": False, "not_run": False, "final": "x", "pass": True}
    assert L.classify(base) == "pass" and L.classify(dict(base, **{"pass": False})) == "fail"
    assert L.classify(dict(base, final=None)) == "no_final"
    assert L.classify(dict(base, final=None, error="RuntimeError('x')")) == "error"
    assert L.classify(dict(base, final=None, error="TimeoutError('run timeout')", timeout=True)) == "timeout"
    assert L.classify(None) == "not_run" and L.classify(dict(base, not_run=True)) == "not_run"


def test_analyze_reports_pairs_and_effect_rule():
    ids = list("abcdefgh")

    def row(p, final="x"):
        return {"task": "", "error": None, "timeout": False, "not_run": False, "final": final, "pass": p,
                "events": [], "hook": None, "score": {"tool_calls": 1}}
    off = {t: row(False) for t in ids}
    live = {t: row(True) for t in ids}
    for r in live.values():
        r["events"] = [{"t": "lesson", "text": "L"}]
        r["hook"] = {"fired": True, "text": "L", "reason": "first_completion_claim"}
    res = L.analyze(ids, {"off": off, "live": live}, p11b_ids=["a"], live_meta={t: {"names": ["n"], "pool_size": 9, "own_shown": t == "a"} for t in ids})
    assert res["mcnemar"]["live_vs_off"]["all"]["b"] == 8 and res["effect"] == "helped"
    assert res["arms"]["live"]["hook"] == {"ok": 8} and res["arms"]["off"]["hook"] == {"ok": 8}
    assert res["live_retrieval"]["own_lesson_shown"] == 1 and res["mcnemar"]["live_vs_off"]["p11b_subset"]["n"] == 1
    live_small = {t: row(t in ("a", "b", "c")) for t in ids}
    res2 = L.analyze(ids, {"off": off, "live": live_small})
    assert res2["effect"] == "no_effect"                              # b=3, c=0, p=0.25


def test_no_final_excluded_pass_rate():
    ids = ["a", "b", "c"]
    rows = {"a": {"task": "a", "error": None, "timeout": False, "not_run": False, "final": "x", "pass": True, "events": [], "hook": None, "score": {"tool_calls": 1}},
            "b": {"task": "b", "error": None, "timeout": False, "not_run": False, "final": None, "pass": False, "events": [], "hook": None, "score": {"tool_calls": 1}},
            "c": {"task": "c", "error": None, "timeout": False, "not_run": False, "final": "x", "pass": False, "events": [], "hook": None, "score": {"tool_calls": 1}}}
    s, _ = L.arm_summary("off", rows, ids)
    assert (s["pass"], s["fail"], s["no_final"], s["ran"], s["n_with_final"]) == (1, 1, 1, 3, 2)
    assert s["pass_rate_excl_no_final"] == 0.5 and s["pass_rate"] == 0.333
