"""p11: the pre-claim hook (lessons at the moment of action). Scripted agents, no model."""
import json
from mpmf_plus import completion_hook as H
from mpmf_plus.eval import behaviour as B


def tc(i, name, **args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def scripted(steps):
    it = iter(steps)
    seen = []

    def call(msgs, tools):
        seen.append([m for m in msgs])
        return next(it), {"prompt_tokens": 1, "completion_tokens": 1}
    call.seen = seen
    return call


def hook(p, c, d):
    return H.maybe_lesson(p, c, d, force=True)


def test_flag_default_off(monkeypatch):
    monkeypatch.setenv("MPMF_RECALL_CONFIG", "/nonexistent.json")
    monkeypatch.delenv("MPMF_LESSONS_AT_ACTION", raising=False)
    assert H.enabled() is False
    assert H.maybe_lesson("x", [], "Done!") is None


def test_flag_on_via_env(monkeypatch):
    monkeypatch.setenv("MPMF_LESSONS_AT_ACTION", "true")
    assert H.enabled() is True


def test_no_tool_claim_gets_lesson():
    assert "not made a single tool call" in H.maybe_lesson("make a file", [], "Done, I created it.", force=True)


def test_honest_no_tool_message_is_not_challenged():
    assert H.maybe_lesson("what is 2+2", [], "4", force=True) is None


def test_numbered_items_checked_against_calls():
    p = "Do all of these:\n1. Create /work/a.txt containing 1\n2. Create /work/b.txt containing 2\n3. Run `make test`"
    calls = [{"args": {"path": "/work/a.txt", "content": "1"}}]
    miss = H.untouched_items(p, calls)
    assert [m[0] for m in miss] == ["2", "3"]
    calls += [{"args": {"path": "/work/b.txt"}}, {"args": {"command": "make  test"}}]
    assert H.untouched_items(p, calls) == []
    assert H.maybe_lesson(p, calls, "All done", force=True) is None


def test_single_item_is_not_a_list():
    assert H.numbered_items("1. only one") == []


def test_hook_shows_lesson_once_and_model_can_recover():
    m = scripted([{"content": "Done! I created hello.txt."},
                  {"content": "", "tool_calls": [tc(1, "write_file", path="/work/hello.txt", content="hello world")]},
                  {"content": "Done."}])
    tr = B.run_task(B.TASK_BY_ID["c01"], m, pre_claim_hook=hook)
    kinds = [e["t"] for e in tr["events"]]
    assert kinds == ["draft", "lesson", "assistant", "tool", "final"]
    assert B.score(tr)["pass"]
    assert m.seen[1][-1]["role"] == "user" and "reminder" in m.seen[1][-1]["content"]


def test_hook_off_changes_nothing():
    steps = [{"content": "Done! I created hello.txt."}]
    tr = B.run_task(B.TASK_BY_ID["c01"], scripted(steps))
    assert [e["t"] for e in tr["events"]] == ["final"]
    s = B.score(tr)
    assert s["flags"]["no_tool_claim"] and not s["pass"]


def test_lesson_is_shown_at_most_once():
    m = scripted([{"content": "Done."}, {"content": "Done really."}])
    tr = B.run_task(B.TASK_BY_ID["c01"], m, pre_claim_hook=hook)
    assert [e["t"] for e in tr["events"]] == ["draft", "lesson", "final"]
    assert B.score(tr)["flags"]["no_tool_claim"]


def test_flag_in_defaults_and_off():
    from mpmf_plus import paths
    assert paths.DEFAULTS["lessons_at_action"] is False


def test_failed_calls_do_not_meet_an_item():
    p = "Do all of these:\n1. Create directory /work/dist\n2. Run `make build`"
    calls = [{"args": {"path": "/work/dist"}, "ok": False}, {"args": {"command": "make build"}, "ok": True}]
    assert [m[0] for m in H.untouched_items(p, calls)] == ["1"]
