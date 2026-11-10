#!/usr/bin/env python3
"""p7: a simulated winning proposal is applied, logged, then reverted on a simulated HOLDOUT drop.

Throwaway MPMF_HOME/EVAL/HEALTH dirs; no model call, no store, nothing outside tmp is touched.
"""
import io, contextlib, json, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def _env():
    tmp = tempfile.mkdtemp(prefix="mpmf-aa-")
    os.environ.update(MPMF_HOME=tmp, MPMF_EVAL_DIR=tmp, MPMF_HEALTH_DIR=tmp,
                      MPMF_RECALL_CONFIG=os.path.join(tmp, "recall.config.json"))
    os.environ.pop("MPMF_AUTO_APPLY", None)
    from mpmf_plus.eval import nightly as N, autoapply as A
    N.BASE = tmp
    N.AR_LOG = os.path.join(tmp, "autorevert_last.txt")
    reg = {"settings": [
        {"id": "d20_k", "apply": "recall_config", "key": "d20_k", "current": 20, "allowed": [10, 40], "step": 5},
        {"id": "top_n", "apply": "plugin_config", "key": "top_k", "current": 8, "allowed": [3, 12], "step": 1},
        {"id": "rerank_budget_ms", "apply": "recall_config", "key": "rerank_skip_after_ms",
         "current": 4500.0, "allowed": [400, 6000], "step": 100}]}
    json.dump(reg, open(os.path.join(tmp, "registry.json"), "w"))
    open(os.path.join(tmp, "PAUSED.md"), "w").write("# PAUSED JOBS\n")
    # reranker/history baselines equal to the candidate's own HOLDOUT, so they never fire here
    for f in ("rerank_baseline.json", "history_baseline.json"):
        json.dump({"acc": 0.6, "rows": {str(i): v for i, v in _rows(0).items()}}, open(os.path.join(tmp, f), "w"))
    return tmp, N, A


def _cfg(tmp, **kw):
    d = {"rerank_skip_after_ms": 4500, "auto_apply": kw.pop("auto_apply", True)}
    d.update(kw)
    json.dump(d, open(os.environ["MPMF_RECALL_CONFIG"], "w"))


def _plan(setting="d20_k", to=25, **kw):
    p = {"setting": setting, "from": 20, "to": to, "outcome": "won_both", "confirmed_on_holdout": True,
         "tune_baseline_acc": 0.40, "tune_candidate_acc": 0.52, "mcnemar_b": 12, "mcnemar_c": 2, "mcnemar_p": 0.013,
         "holdout_baseline_acc": 0.40, "holdout_candidate_acc": 0.47, "mcnemar_b_holdout": 9, "mcnemar_c_holdout": 2,
         "mcnemar_p_holdout": 0.065}
    p.update(kw)
    return p


def _rows(n_wrong=0):
    # candidate HOLDOUT: 40 correct of 64 -> n_wrong of them regress
    return {i: ("wrong" if i <= n_wrong else ("correct" if i <= 40 else "declined")) for i in range(1, 65)}


def _night(N, tmp, cur):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = N.autorevert([{"id": i, "split": "holdout", "verdict": v} for i, v in cur.items()])
    return rc, buf.getvalue()


def _cfg_now():
    return json.load(open(os.environ["MPMF_RECALL_CONFIG"]))


def test_end_to_end_apply_log_revert():
    tmp, N, A = _env()
    _cfg(tmp)
    base = {i: ("correct" if i <= 34 else "declined") for i in range(1, 65)}
    ok, msg = A.maybe_apply(tmp, _plan(), _rows(0), base)
    assert ok, msg
    assert _cfg_now()["d20_k"] == 25 and _cfg_now()["auto_apply"] is True
    reg = json.load(open(os.path.join(tmp, "registry.json")))
    assert reg["settings"][0]["current"] == 25
    paused = open(os.path.join(tmp, "PAUSED.md")).read()
    assert "d20_k auto-apply" in paused and "20 -> 25" in paused and "REVERTED" not in paused
    assert "AUTO-APPLIED d20_k" in msg and "nightly reverts" in msg
    assert A.load_state(tmp)["status"] == "active"
    # night 1: same as candidate -> keep, config untouched, clean night counted
    rc, out = _night(N, tmp, _rows(0))
    assert rc == 0 and out == "" and _cfg_now()["d20_k"] == 25
    assert A.load_state(tmp)["clean_nights"] == 1
    # night 2: 8 correct answers regress (-12.5 pts, p~0.008) -> REVERT
    rc, out = _night(N, tmp, _rows(8))
    assert rc == 0 and "AUTO-APPLY REVERTED d20_k" in out, out
    assert "d20_k" not in _cfg_now() and _cfg_now()["auto_apply"] is True
    assert json.load(open(os.path.join(tmp, "registry.json")))["settings"][0]["current"] == 20
    paused = open(os.path.join(tmp, "PAUSED.md")).read()
    assert "d20_k auto-apply REVERTED" in paused
    assert A.load_state(tmp)["status"] == "reverted"
    ev = [json.loads(l)["event"] for l in open(os.path.join(tmp, "autoapply_log.jsonl"))]
    assert ev == ["applied", "reverted"], ev
    # once reverted the check is gone: nothing more to do, nothing re-fires
    rc, out = _night(N, tmp, _rows(8))
    assert out == "" and A.active_checks(tmp) == []


def test_noise_is_kept_and_change_settles():
    tmp, N, A = _env()
    _cfg(tmp)
    ok, _ = A.maybe_apply(tmp, _plan(), _rows(0), {i: "declined" for i in range(1, 65)})
    assert ok
    rc, out = _night(N, tmp, _rows(2))          # -3 pts: below the 5.4 pt floor
    assert out == "" and _cfg_now()["d20_k"] == 25
    for _ in range(2):
        _night(N, tmp, _rows(0))
    assert A.load_state(tmp)["status"] == "settled" and _cfg_now()["d20_k"] == 25


def test_refusals():
    tmp, N, A = _env()
    base = {i: "declined" for i in range(1, 65)}
    # default OFF
    _cfg(tmp, auto_apply=False)
    ok, msg = A.maybe_apply(tmp, _plan(), _rows(), base)
    assert not ok and "OFF" in msg and "d20_k" not in _cfg_now()
    _cfg(tmp)
    # code/plugin setting
    ok, msg = A.maybe_apply(tmp, _plan("top_n", 9), _rows(), base)
    assert not ok and "never auto-applied" in msg
    # the user's rerank-off rule
    ok, msg = A.maybe_apply(tmp, _plan("rerank_budget_ms", 4600), _rows(), base)
    assert not ok and "never-auto" in msg and _cfg_now()["rerank_skip_after_ms"] == 4500
    # not confirmed / not actually significant / out of range
    assert not A.maybe_apply(tmp, _plan(outcome="won_tune_failed_holdout", confirmed_on_holdout=False), _rows(), base)[0]
    assert not A.maybe_apply(tmp, _plan(mcnemar_b=5, mcnemar_c=4), _rows(), base)[0]
    assert not A.maybe_apply(tmp, _plan(to=99), _rows(), base)[0]
    assert "d20_k" not in _cfg_now() and A.load_state(tmp) is None
    # one at a time
    assert A.maybe_apply(tmp, _plan(), _rows(), base)[0]
    ok, msg = A.maybe_apply(tmp, _plan(to=30), _rows(), base)
    assert not ok and "still active" in msg and _cfg_now()["d20_k"] == 25


def test_revert_never_clobbers_a_hand_edit():
    tmp, N, A = _env()
    _cfg(tmp)
    assert A.maybe_apply(tmp, _plan(), _rows(0), {i: "declined" for i in range(1, 65)})[0]
    c = _cfg_now(); c["d20_k"] = 15; json.dump(c, open(os.environ["MPMF_RECALL_CONFIG"], "w"))
    rc, out = _night(N, tmp, _rows(8))
    assert "NOT REVERTED" in out and _cfg_now()["d20_k"] == 15
    assert A.load_state(tmp)["status"] == "superseded"


def test_proposer_hook_is_wired():
    import inspect
    from mpmf_plus.eval import proposer, nightly
    assert "autoapply.maybe_apply" in inspect.getsource(proposer.main)
    assert "autoapply.active_checks" in inspect.getsource(nightly.autorevert)


if __name__ == "__main__":
    fails = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            try:
                f(); print("ok", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1
    print("FAIL" if fails else "OK test_autoapply")
    sys.exit(1 if fails else 0)
