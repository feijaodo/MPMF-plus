"""p7 — automatic apply of a winning proposal to ONE registry setting, with rollback.

Flow
  proposer: wins on TUNE, confirms on HOLDOUT -> ``maybe_apply`` writes the new value
            into recall.config.json (read per call by the provider), updates the registry
            ``current``, appends a PAUSED.md decision row and returns the message for the user.
  nightly : ``active_checks`` hands the nightly auto-revert two extra checks (HOLDOUT now vs
            the candidate's HOLDOUT, and vs the pre-change HOLDOUT). Past the noise floor
            (same rule as the other auto-reverts: drop >= REVERT_DROP and McNemar p < 0.10)
            ``_revert`` restores the previous value, logs a PAUSED.md row and returns the message.

Hard limits (code, not convention)
  * only registry settings with apply == "recall_config" (a JSON key) can be applied;
    plugin_config / code / unknown ids are refused. Nothing here writes Python.
  * settings in NEVER_AUTO are refused (they undo a deliberate human rule).
  * default OFF: ``auto_apply`` in recall.config.json (or MPMF_AUTO_APPLY=1) must be true.
  * one change at a time: a new apply is refused while the last one is still "active".
  * a revert never clobbers a value someone changed by hand after the apply.
"""
import json
import math
import os
import shutil
import time

from .. import paths as _paths

STATE_NAME = "autoapply_state.json"
LOG_NAME = "autoapply_log.jsonl"
CAND_BASELINE = "autoapply_baseline.json"       # candidate HOLDOUT rows at apply time
PRE_BASELINE = "autoapply_pre_baseline.json"    # HOLDOUT rows before the change
SETTLE_NIGHTS = 3                               # clean nightlies before the change is "settled"

# rerank_budget_ms -> rerank_skip_after_ms: the user's rule 2026-11-05 (PAUSED.md rerank-off-88c)
# keeps the reranker OFF (0); a "winning" budget would silently turn it back on.
NEVER_AUTO = {"rerank_budget_ms"}


def _now():
    return time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())


def _mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def _jwrite(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def _jread(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return default


def _log(base, event, **kw):
    try:
        with open(os.path.join(base, LOG_NAME), "a") as fh:
            fh.write(json.dumps(dict(ts=_now(), event=event, **kw)) + "\n")
    except Exception:
        pass


def enabled():
    """Default OFF. recall.config.json ``auto_apply`` (or MPMF_AUTO_APPLY) turns it on."""
    return bool(_paths.load_config().get("auto_apply", False))


def load_state(base):
    return _jread(os.path.join(base, STATE_NAME), None)


def eligible(plan, setting):
    """(ok, reason). Re-derives the verdict from the raw counts; trusts nothing else."""
    if plan.get("outcome") != "won_both" or not plan.get("confirmed_on_holdout"):
        return False, "not won on TUNE and confirmed on HOLDOUT"
    b, c = plan.get("mcnemar_b", 0), plan.get("mcnemar_c", 0)
    bh, ch = plan.get("mcnemar_b_holdout", 0), plan.get("mcnemar_c_holdout", 0)
    if not (b > c and _mcnemar(b, c) < 0.05):
        return False, "TUNE counts b=%d c=%d do not win at p<0.05" % (b, c)
    if not (bh >= ch and _mcnemar(bh, ch) < 0.10):
        return False, "HOLDOUT counts b=%d c=%d do not confirm at p<0.10" % (bh, ch)
    if setting is None:
        return False, "setting %r not in registry" % plan.get("setting")
    if setting.get("apply") != "recall_config":
        return False, "setting %s apply=%r is not a recall_config registry key (code/plugin changes are never auto-applied)" % (
            setting["id"], setting.get("apply"))
    if setting["id"] in NEVER_AUTO:
        return False, "setting %s is on the never-auto list" % setting["id"]
    lo, hi = setting["allowed"]
    v = plan.get("to")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not (lo <= v <= hi):
        return False, "value %r outside allowed [%s, %s]" % (v, lo, hi)
    return True, "ok"


def maybe_apply(base, plan, cand_rows, base_rows, force_enabled=None):
    """Apply a won_both proposal. Returns (applied: bool, message: str).

    cand_rows / base_rows: {question id: verdict} on HOLDOUT, candidate and pre-change.
    """
    on = enabled() if force_enabled is None else force_enabled
    if not on:
        return False, "auto_apply is OFF (default) -> proposal NOT applied"
    registry_path = os.path.join(base, "registry.json")
    registry = _jread(registry_path)
    if not registry:
        return False, "registry.json unreadable in %s -> NOT applied" % base
    setting = next((s for s in registry.get("settings", []) if s.get("id") == plan.get("setting")), None)
    ok, why = eligible(plan, setting)
    if not ok or setting is None:
        _log(base, "refused", setting=plan.get("setting"), reason=why)
        return False, "auto-apply refused: " + why
    st = load_state(base)
    if st and st.get("status") == "active":
        why = "previous auto-apply %s still active (%d/%d clean nights)" % (
            st["setting"], st.get("clean_nights", 0), SETTLE_NIGHTS)
        _log(base, "refused", setting=plan["setting"], reason=why)
        return False, "auto-apply refused: " + why
    cfg_path = str(_paths.config_path())
    cfg = _jread(cfg_path)
    if not isinstance(cfg, dict):
        return False, "auto-apply refused: %s unreadable" % cfg_path
    key, new = setting["key"], plan["to"]
    if isinstance(new, float) and new == int(new) and isinstance(setting["allowed"][0], int):
        new = int(new)
    had = key in cfg
    old = cfg[key] if had else _paths.DEFAULTS.get(key, setting["current"])
    bdir = os.path.join(base, "autoapply_backups")
    os.makedirs(bdir, exist_ok=True)
    shutil.copy2(cfg_path, os.path.join(bdir, "recall.config.%s.json" % time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())))
    cfg[key] = new
    _jwrite(cfg_path, cfg)
    # keep the registry in step so the next proposal starts from the live value
    registry_prev = setting["current"]
    setting["current"] = new
    _jwrite(registry_path, registry)
    ev = {k: plan.get(k) for k in (
        "tune_baseline_acc", "tune_candidate_acc", "mcnemar_b", "mcnemar_c", "mcnemar_p",
        "holdout_baseline_acc", "holdout_candidate_acc", "mcnemar_b_holdout",
        "mcnemar_c_holdout", "mcnemar_p_holdout")}
    state = {"status": "active", "setting": setting["id"], "key": key, "had_key": had,
             "from": old, "to": new, "registry_from": registry_prev, "applied_ts": _now(),
             "clean_nights": 0, "evidence": ev}
    _jwrite(os.path.join(base, CAND_BASELINE), {"rows": {str(k): v for k, v in cand_rows.items()},
                                                "ts": state["applied_ts"], "acc": ev["holdout_candidate_acc"]})
    _jwrite(os.path.join(base, PRE_BASELINE), {"rows": {str(k): v for k, v in base_rows.items()},
                                               "ts": state["applied_ts"], "acc": ev["holdout_baseline_acc"]})
    _jwrite(os.path.join(base, STATE_NAME), state)
    _log(base, "applied", **{k: state[k] for k in ("setting", "key", "from", "to")})
    _paused_row(
        "%s auto-apply" % setting["id"], "`recall.config.json` `%s`: %s -> %s" % (key, old, new),
        "TUNE %.1f%%->%.1f%% (b=%s c=%s p=%.3f); HOLDOUT %.1f%%->%.1f%% (b=%s c=%s p=%.3f)" % (
            100 * ev["tune_baseline_acc"], 100 * ev["tune_candidate_acc"], ev["mcnemar_b"], ev["mcnemar_c"],
            ev["mcnemar_p"], 100 * ev["holdout_baseline_acc"], 100 * ev["holdout_candidate_acc"],
            ev["mcnemar_b_holdout"], ev["mcnemar_c_holdout"], ev["mcnemar_p_holdout"]),
        "proposer (auto; nightly reverts on HOLDOUT drop)")
    msg = ("AUTO-APPLIED %s: %s %s -> %s (TUNE %+.0f pts, HOLDOUT %+.0f pts, McNemar p=%.3f/%.3f). "
           "The nightly reverts it if HOLDOUT drops >= %.1f pts at p<0.10."
           % (setting["id"], key, old, new,
              100 * (ev["tune_candidate_acc"] - ev["tune_baseline_acc"]),
              100 * (ev["holdout_candidate_acc"] - ev["holdout_baseline_acc"]),
              ev["mcnemar_p"], ev["mcnemar_p_holdout"], 5.4))
    return True, msg


def _paused_row(what, where, evidence, who):
    # same 5-column shape as the nightly's history_mode auto-revert row
    try:
        path = os.path.join(_paths.health_dir(), "PAUSED.md")
        with open(path, "a") as fh:
            fh.write("| %s | %s | %s | %s | %s |\n" % (_now(), what, where, evidence, who))
    except Exception:
        pass


def _revert(base):
    def fn(a_cur, a_base, drop, p, b, c):
        st = load_state(base)
        if not st or st.get("status") != "active":
            return ""
        cfg_path = str(_paths.config_path())
        cfg = _jread(cfg_path)
        if not isinstance(cfg, dict):
            return "AUTO-APPLY REVERT FAILED: %s unreadable (%s still applied)" % (cfg_path, st["setting"])
        key = st["key"]
        if cfg.get(key) != st["to"]:
            st["status"] = "superseded"
            _jwrite(os.path.join(base, STATE_NAME), st)
            _log(base, "revert_skipped", setting=st["setting"], now=cfg.get(key))
            return ("AUTO-APPLY NOT REVERTED: %s is now %r (not the applied %r) — changed by hand since; left alone"
                    % (key, cfg.get(key), st["to"]))
        if st["had_key"]:
            cfg[key] = st["from"]
        else:
            cfg.pop(key, None)
        _jwrite(cfg_path, cfg)
        reg_path = os.path.join(base, "registry.json")
        reg = _jread(reg_path)
        if reg:
            for s in reg.get("settings", []):
                if s.get("id") == st["setting"]:
                    s["current"] = st["registry_from"]
            _jwrite(reg_path, reg)
        st.update(status="reverted", reverted_ts=_now())
        _jwrite(os.path.join(base, STATE_NAME), st)
        _log(base, "reverted", setting=st["setting"], restored=st["from"], a_cur=a_cur, a_base=a_base, p=p)
        _paused_row("%s auto-apply REVERTED" % st["setting"],
                    "`recall.config.json` `%s`: %s -> %s" % (key, st["to"], st["from"] if st["had_key"] else "(key removed)"),
                    "HOLDOUT %.1f%% vs reference %.1f%% (-%.1f pts), McNemar p=%.3f (b=%d c=%d)"
                    % (a_cur * 100, a_base * 100, drop * 100, p, b, c), "nightly")
        return ("AUTO-APPLY REVERTED %s: %s back to %s. HOLDOUT %.3f vs %.3f (-%.1f pts), McNemar p=%.3f (b=%d c=%d)"
                % (st["setting"], key, st["from"] if st["had_key"] else "default", a_cur, a_base, drop * 100, p, b, c))
    return fn


def active_checks(base):
    """Extra (name, baseline_file, revert_fn) tuples for nightly.autorevert; [] unless active."""
    st = load_state(base)
    if not st or st.get("status") != "active":
        return []
    fn = _revert(base)
    return [("autoapply", CAND_BASELINE, fn), ("autoapply-pre", PRE_BASELINE, fn)]


def tick(base, checks_ran_ok):
    """Call once per nightly after the checks. Counts clean nights; settles after SETTLE_NIGHTS."""
    st = load_state(base)
    if not st or st.get("status") != "active" or not checks_ran_ok:
        return ""
    st["clean_nights"] = st.get("clean_nights", 0) + 1
    if st["clean_nights"] >= SETTLE_NIGHTS:
        st["status"] = "settled"
        _log(base, "settled", setting=st["setting"])
    _jwrite(os.path.join(base, STATE_NAME), st)
    return ""
