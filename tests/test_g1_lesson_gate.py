"""g1: the lesson gate's match rule, every-merge-counts occurrences, 2-session promotion.

Synthetic fixtures only (invented names/ids/text). Embeddings are stubbed with
exact unit vectors so a pair's cosine is whatever the test says it is; the vet
panel is stubbed; promotion is recorded, not written.
"""
import json
import math

import pytest

from mpmf.memory import MemoryManager
from mpmf_plus import cognition as cg
from mpmf_plus import provider as A

DIM = 1024


def _unit(i, j=None, cos=None):
    """Unit vector e_i, or one whose cosine with e_i is `cos` (uses e_j)."""
    v = [0.0] * DIM
    if cos is None:
        v[i] = 1.0
    else:
        v[i] = cos
        v[j] = math.sqrt(1.0 - cos * cos)
    return v


def _raw(v):
    import numpy as np
    return np.asarray(v, dtype="<f4").tobytes()


class Env:
    """A temp store + a gate wired to it, no network, no live paths."""

    def __init__(self, tmp_path, monkeypatch):
        self.path = str(tmp_path / "t.mpmf")
        self.vecs = {}          # text prefix -> vector the stub embedder returns
        self.promoted = []
        self.mm = MemoryManager(self.path, vector_dimensions=DIM)
        monkeypatch.setattr(A, "PATH", self.path)
        monkeypatch.setattr(cg, "GATE_LOG", tmp_path / "gate.jsonl")
        monkeypatch.setattr(cg, "VET_PATH", tmp_path / "vet.json")
        monkeypatch.setattr(cg, "LOCK_PATH", tmp_path / "intake.lock")
        monkeypatch.setattr(cg, "_vec", lambda text: self.vecs[text.split(" | ")[0]])
        # r107: the gate embeds through the batched/cached helper now, so a test
        # that stubs the embedding must stub THAT entry point (the logic under
        # test here is the match rule, not the embedder).
        monkeypatch.setattr(cg, "_vecs", lambda texts, dry_run=False:
                            [self.vecs[t.split(" | ")[0]] for t in texts])
        monkeypatch.setattr(A, "_sit_vectors", lambda: (self._pool_ids, self._pool_mat, {}))
        monkeypatch.setattr(cg, "vet_lesson",
                            lambda *a, **k: ("keep", "stub", 0.0, {}))
        monkeypatch.setattr(cg, "_promote",
                            lambda mm, nid, name, text, wiki, sessions, occ:
                            self.promoted.append((name, list(sessions), occ)) or "p")
        self._pool_ids, self._pool_mat = [], None
        self._pool = []

    def _node(self, name, text, sessions, occ, status, created, rule="", vec=None):
        wiki = {"_ns": "cognition_staging" if name.startswith("cog_stage_") else "cognition",
                "_source_sessions": json.dumps(sessions), "_occurrences": str(occ),
                "_created_at": created}
        if sessions:
            wiki["_source_session"] = sessions[0]
        if status:
            wiki["_status"] = status
        if rule:
            wiki["_rule_id"] = rule
        with self.mm.write_txn():
            self.mm.create_node(name=name, node_type=1, plane_flags=1, text=text,
                                summary=text[:50], wiki=wiki, vector=_raw(vec),
                                importance=0.5)

    def staged(self, name, sit, vec, sessions, occ=1, rule="", created="2026-11-10T00:00:00+00:00"):
        self.vecs[sit] = vec
        self._node(name, sit + " | principle | rule", sessions, occ, "held", created, rule, vec)

    def pool(self, name, sit, vec, sessions, occ=1, rule=""):
        self.vecs[sit] = vec
        self._node(name, sit + " | principle | rule", sessions, occ, "hypothesis",
                   "2026-10-01T00:00:00+00:00", rule, vec)
        self._pool.append((name, vec))

    def finish(self):
        import numpy as np
        A._invalidate_open_cache()
        if self._pool:
            mm = A._open().mem
            self._pool_ids = [mm.get_node_id_by_name(n) for n, _ in self._pool]
            self._pool_mat = np.asarray([v for _, v in self._pool], dtype="float32")

    def run(self, *extra):
        self.finish()
        rc = cg.main(["--store", self.path, "--lock-held", "--no-panel"] + list(extra))
        assert rc == 0
        A._invalidate_open_cache()
        return dict(cg.LAST_SUMMARY)

    def wiki(self, name):
        A._invalidate_open_cache()
        mm = A._open().mem
        nid = mm.get_node_id_by_name(name)
        return mm.get_wiki(nid) or {}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    return Env(tmp_path, monkeypatch)


# ---- (a) same rule id, cosine 0.80+, two distinct sessions -> merged and promotable
def test_a_same_rule_id_two_sessions_merges_and_promotes(env):
    env.staged("cog_stage_a1", "alpha situation", _unit(0), ["sess-1"], rule="rule_x",
               created="2026-11-01T00:00:00+00:00")
    env.staged("cog_stage_a2", "alpha again", _unit(0, 1, 0.82), ["sess-2"], rule="rule_x",
               created="2026-11-02T00:00:00+00:00")
    s = env.run()
    assert s["merged"] == 1 and s["merges"][0]["thr"] == cg.COS_SAME_ID
    w = env.wiki("cog_stage_a1")
    assert json.loads(w["_source_sessions"]) == ["sess-1", "sess-2"]
    assert w["_occurrences"] == "2"
    assert env.wiki("cog_stage_a2")["_status"] == "merged"
    assert s["vet_candidates"] == 1 and s["promoted"] == 1
    assert env.promoted == [("cog_stage_a1", ["sess-1", "sess-2"], 2)]


# ---- (b) the same lesson twice from ONE session: merged, occurrences 2, one session, not promoted
def test_b_same_session_twice_counts_but_never_promotes(env):
    env.staged("cog_stage_b1", "beta situation", _unit(2), ["sess-1"], rule="rule_y",
               created="2026-11-01T00:00:00+00:00")
    env.staged("cog_stage_b2", "beta repeated", _unit(2, 3, 0.95), ["sess-1"], rule="rule_y",
               created="2026-11-02T00:00:00+00:00")
    s = env.run()
    assert s["merged"] == 1
    w = env.wiki("cog_stage_b1")
    assert json.loads(w["_source_sessions"]) == ["sess-1"]      # still ONE session
    assert w["_occurrences"] == "2"                              # but the merge counted
    assert s["promoted"] == 0 and env.promoted == [] and s["vet_candidates"] == 0
    assert w["_status"] == "held"
    # and a third repeat from the same session still does not promote
    env.staged("cog_stage_b3", "beta third", _unit(2, 4, 0.97), ["sess-1"], rule="rule_y",
               created="2026-11-03T00:00:00+00:00")
    s = env.run()
    w = env.wiki("cog_stage_b1")
    assert w["_occurrences"] == "3" and json.loads(w["_source_sessions"]) == ["sess-1"]
    assert s["promoted"] == 0 and env.promoted == []


# ---- (c) different rule ids never merge, even at cosine 1.0
def test_c_different_rule_ids_never_merge(env):
    env.staged("cog_stage_c1", "gamma situation", _unit(5), ["sess-1"], rule="rule_one",
               created="2026-11-01T00:00:00+00:00")
    env.staged("cog_stage_c2", "gamma twin", _unit(5), ["sess-2"], rule="rule_two",
               created="2026-11-02T00:00:00+00:00")
    env.pool("cog_rum_c3", "gamma pool twin", _unit(5), ["sess-3"], rule="rule_three")
    s = env.run()
    assert s["merged"] == 0
    assert env.wiki("cog_stage_c1")["_status"] == "held"
    assert env.wiki("cog_stage_c2")["_status"] == "held"
    assert s["promoted"] == 0


def test_c2_match_threshold_table():
    assert cg.match_threshold("a", "a") == cg.COS_SAME_ID == 0.80
    assert cg.match_threshold("a", "b") is None
    assert cg.match_threshold("", "a") == cg.COS_NO_ID == 0.85
    assert cg.match_threshold("a", "") == cg.COS_NO_ID
    assert cg.match_threshold("", "") == cg.COS_NO_ID


def test_c3_rule_id_is_read_from_metadata_or_text():
    assert cg._rule_id({"_rule_id": "Rule_X"}) == "rule_x"
    assert cg._rule_id({"_question_id": "false_x"}) == "false_x"
    assert cg._rule_id({}, "Rule fired: some_rule_id. Evidence: zzz") == "some_rule_id"
    # a prose rule name (spaces) carries no usable id
    assert cg._rule_id({}, "Rule fired: No Chinese in output to D. Evidence: x") == ""
    assert cg._rule_id({}, "no marker here") == ""


# ---- (d) no id: 0.82 does not merge, 0.86 does
@pytest.mark.parametrize("cos,merged", [(0.82, 0), (0.86, 1)])
def test_d_no_id_pair_threshold(env, cos, merged):
    env.staged("cog_stage_d1", "delta situation", _unit(7), ["sess-1"],
               created="2026-11-01T00:00:00+00:00")
    env.staged("cog_stage_d2", "delta sibling", _unit(7, 8, cos), ["sess-2"],
               created="2026-11-02T00:00:00+00:00")
    s = env.run()
    assert s["merged"] == merged
    assert (env.wiki("cog_stage_d2")["_status"] == "merged") == bool(merged)


def test_d2_one_side_without_id_uses_the_stricter_threshold(env):
    env.staged("cog_stage_d3", "eps situation", _unit(9), ["sess-1"], rule="rule_z",
               created="2026-11-01T00:00:00+00:00")
    env.staged("cog_stage_d4", "eps sibling", _unit(9, 10, 0.82), ["sess-2"],
               created="2026-11-02T00:00:00+00:00")                  # no id
    assert env.run()["merged"] == 0


# ---- staged-vs-pool honours the same rule, and a pool survivor records the merge
def test_staged_into_pool_counts_and_adds_session(env):
    env.pool("cog_rum_p1", "zeta pool lesson", _unit(11), ["sess-1"], rule="rule_p")
    env.staged("cog_stage_p2", "zeta staged", _unit(11, 12, 0.83), ["sess-2"], rule="rule_p")
    s = env.run()
    assert s["merged"] == 1 and s["merges"][0]["kind"] == "pool"
    w = env.wiki("cog_rum_p1")
    assert json.loads(w["_source_sessions"]) == ["sess-1", "sess-2"] and w["_occurrences"] == "2"
    assert json.loads(w["_merged_from"]) == ["cog_stage_p2"]


def test_a_merge_is_counted_once_even_if_replayed(env):
    env.pool("cog_rum_q1", "eta pool", _unit(13), ["sess-1"])
    env.staged("cog_stage_q2", "eta staged", _unit(13, 14, 0.9), ["sess-2"])
    env.run()
    # replay the same merge by hand (as a crashed/retried run would)
    out = {}
    cg._rw_update_many([("cog_rum_q1", cg._merge_mut(["sess-2"], "x", "cog_stage_q2", 1, "", out))])
    assert out.get("already") is True
    assert env.wiki("cog_rum_q1")["_occurrences"] == "2"


# ---- held lessons that reach 2 sessions only through the merge are promotable; expiry is post-merge
def test_expiry_is_judged_after_the_merge(env):
    env.staged("cog_stage_x1", "theta situation", _unit(15), ["sess-1"],
               created="2026-09-01T00:00:00+00:00")                   # older than 30 days
    env.staged("cog_stage_x2", "theta sibling", _unit(15, 16, 0.9), ["sess-2"],
               created="2026-11-05T00:00:00+00:00")
    env.staged("cog_stage_x3", "lonely situation", _unit(20), ["sess-9"],
               created="2026-09-01T00:00:00+00:00")
    s = env.run("--as-of", "2026-11-13T00:00:00+00:00")
    assert env.wiki("cog_stage_x1")["_status"] == "promoted"         # merge partner saved it from expiry
    assert s["promoted"] == 1 and s["expired"] == 1
    assert env.wiki("cog_stage_x3")["_status"] == "expired"


# ---- rebuild: sessions from the gate log + node metadata, occurrences from the merges logged
def test_rebuild_reconstructs_sessions_and_occurrences(env, tmp_path):
    env.pool("cog_rum_r1", "iota pool", _unit(17), ["sess-1"], occ=1)
    env.staged("cog_stage_r2", "kappa staged", _unit(18), ["sess-5"], occ=1)
    log = tmp_path / "gate.jsonl"
    lines = [
        {"name": "cog_stage_r2", "decision": "held", "sessions": ["sess-5", "sess-6"]},
        {"name": "cog_stage_r2", "decision": "held", "sessions": ["sess-5", "sess-6"]},
        {"name": "cog_stage_gone", "decision": "merged", "target": "cog_rum_r1",
         "sessions": ["sess-7"], "occurrences": 2},
        {"name": "cog_stage_gone", "decision": "merged", "target": "cog_rum_r1",
         "sessions": ["sess-7"], "occurrences": 2},
        {"name": "cog_stage_dry", "decision": "merged", "target": "cog_rum_r1",
         "sessions": ["sess-8"], "dry_run": True},
    ]
    log.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    info = cg.read_gate_log(str(log))
    assert set(info["names"]) == {"cog_stage_r2", "cog_stage_gone"}      # distinct, no dry run
    env.finish()
    mm = A._open().mem
    items = [cg._item("cog_stage_r2", mm.get_node_id_by_name("cog_stage_r2"), "staged",
                      mm.get_wiki(mm.get_node_id_by_name("cog_stage_r2")), "t"),
             cg._item("cog_rum_r1", mm.get_node_id_by_name("cog_rum_r1"), "pool",
                      mm.get_wiki(mm.get_node_id_by_name("cog_rum_r1")), "t")]
    out = cg.rebuild_held_state(items, log_info=info)
    assert out["cog_stage_r2"]["sessions"] == ["sess-5", "sess-6"] and out["cog_stage_r2"]["occ"] == 2
    assert out["cog_rum_r1"]["sessions"] == ["sess-1", "sess-7"]
    assert out["cog_rum_r1"]["occ"] == 2          # one DISTINCT merged-in name, logged twice
    assert out["cog_rum_r1"]["merged_from"] == ["cog_stage_gone"]


def test_rebuild_flag_writes_nothing_on_dry_run_and_applies_when_real(env, tmp_path):
    env.staged("cog_stage_s1", "lambda staged", _unit(19), ["sess-1"])
    (tmp_path / "gate.jsonl").write_text(json.dumps(
        {"name": "cog_stage_s1", "decision": "held", "sessions": ["sess-1", "sess-2"]}) + "\n")
    s = env.run("--rebuild", "--dry-run")
    assert s["rebuilt"] == 1 and s["dry_run"] is True
    assert json.loads(env.wiki("cog_stage_s1")["_source_sessions"]) == ["sess-1"]
    s = env.run("--rebuild")
    assert json.loads(env.wiki("cog_stage_s1")["_source_sessions"]) == ["sess-1", "sess-2"]
    assert env.wiki("cog_stage_s1")["_occurrences"] == "2"
    assert s["promoted"] == 1                      # 2 distinct sessions + stubbed keep


# ---- vet budget: out of budget is held (vet_budget), never rejected
def test_out_of_vet_budget_is_held_not_rejected(env, monkeypatch):
    for i in range(3):
        env.staged("cog_stage_v%d" % i, "mu situation %d" % i, _unit(30 + i), ["sess-1", "sess-%d" % (2 + i)])
    calls = []

    def vet(text, doc, idx, allow, spend, max_vet):
        calls.append(text)
        if len(calls) > 1:
            return "unvetted", "skipped", 0.0, {}
        return "keep", "stub", 0.0, {}
    monkeypatch.setattr(cg, "vet_lesson", vet)
    env.finish()
    assert cg.main(["--store", env.path, "--lock-held", "--max-vet", "1"]) == 0
    s = cg.LAST_SUMMARY
    assert s["promoted"] == 1 and s["vet_budget_held"] == 2 and s["rejected"] == 0
    held = [env.wiki("cog_stage_v%d" % i) for i in range(3)]
    assert sorted(w["_gate_reason"] for w in held if w["_status"] == "held") == ["vet_budget"] * 2


# ---- (e) the daily rumination staging path writes a cog_stage_ node on a temp store
def test_e_daily_rumination_staging_writes_a_cog_stage_node(tmp_path, monkeypatch):
    import hashlib
    import pathlib
    from mpmf_plus import memory as M
    path = str(tmp_path / "e.mpmf")
    mm = MemoryManager(path, vector_dimensions=DIM)
    with mm.write_txn():                       # create the (empty-ish) store on disk
        mm.create_node(name="seed", node_type=1, plane_flags=1, text="seed", summary="seed",
                       wiki={"_ns": "memory"}, vector=_raw(_unit(0)), importance=0.5)

    def fv(text, dim=1024, cap=None):
        h = hashlib.sha256(text.encode()).digest()
        v = [((h[i % 32] / 255.0) - 0.5) for i in range(dim)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]
    monkeypatch.setattr(M, "_embed_floats_local", fv)
    monkeypatch.setattr(A, "PATH", path)
    monkeypatch.setattr(M, "CACHE_DIR", pathlib.Path(tmp_path) / "cache")
    monkeypatch.setattr(M, "EXTRACTIONS_CACHE", pathlib.Path(tmp_path) / "cache" / "ex.json")
    monkeypatch.setattr(cg, "GATE_LOG", tmp_path / "gate.jsonl")
    prev = object.__getattribute__(M._VIEW, "_settings")     # restore the process-global view after
    M.configure_store(path=path, mode="attach")
    try:
        r = cg.stage_lesson("A made-up problem | made-up trigger | made-up question",
                            session="sess-A", rule_id="false_completion_claim",
                            origin="rumination", source="deterministic", auto_extract=False)
        assert r["action"] == "staged" and r["name"].startswith("cog_stage_")
        A._invalidate_open_cache()
        mm2 = A._open().mem
        nid = mm2.get_node_id_by_name(r["name"])
        w = mm2.get_wiki(nid)
        assert w["_ns"] == "cognition_staging" and w["_source_session"] == "sess-A"
        assert json.loads(w["_source_sessions"]) == ["sess-A"] and w["_occurrences"] == "1"
        assert w["_rule_id"] == "false_completion_claim" and w["_origin"] == "rumination"
        # the same text again from another session is absorbed (every merge counts), not duplicated
        r2 = cg.stage_lesson("A made-up problem | made-up trigger | made-up question",
                             session="sess-B", rule_id="false_completion_claim",
                             auto_extract=False)
        assert r2["action"] == "absorbed" and r2["target"] == r["name"]
        w = cg.A._open().mem.get_wiki(A._open().mem.get_node_id_by_name(r["name"]))
        assert json.loads(w["_source_sessions"]) == ["sess-A", "sess-B"] and w["_occurrences"] == "2"
        assert cg.stage_lesson("   ")["action"] == "skipped"
    finally:
        object.__setattr__(M._VIEW, "_settings", prev)
        M._invalidate_open_cache()


def test_e2_staging_dry_run_writes_nothing(tmp_path, monkeypatch):
    """The rumination --dry-run proof: stage_lesson(dry_run=True) reports, never writes."""
    import hashlib
    import pathlib
    from mpmf_plus import memory as M
    path = str(tmp_path / "e2.mpmf")
    mm = MemoryManager(path, vector_dimensions=DIM)
    with mm.write_txn():
        mm.create_node(name="seed", node_type=1, plane_flags=1, text="seed", summary="seed",
                       wiki={"_ns": "memory"}, vector=_raw(_unit(0)), importance=0.5)
    monkeypatch.setattr(A, "PATH", path)
    monkeypatch.setattr(M, "CACHE_DIR", pathlib.Path(tmp_path) / "cache")
    monkeypatch.setattr(M, "EXTRACTIONS_CACHE", pathlib.Path(tmp_path) / "cache" / "ex.json")
    prev = object.__getattribute__(M._VIEW, "_settings")
    M.configure_store(path=path, mode="attach")
    try:
        before = hashlib.md5(open(path, "rb").read()).hexdigest()
        r = cg.stage_lesson("Another made-up problem | made-up trigger | made-up question",
                            session="sess-Z", rule_id="rule_q", dry_run=True)
        assert r["action"] == "would_stage" and r["name"].startswith("cog_stage_")
        assert hashlib.md5(open(path, "rb").read()).hexdigest() == before
    finally:
        object.__setattr__(M._VIEW, "_settings", prev)
        M._invalidate_open_cache()
