"""b0: mpmf_plus.bench -- stats, text metrics, loaders, judges, summarize on a truncated file, fake end-to-end run.

Synthetic fixtures only; no network, no model, no real data, no absolute paths.
"""
import json
import os
import subprocess
import sys

import pytest

from mpmf_plus.bench import datasets as D
from mpmf_plus.bench import judges as J
from mpmf_plus.bench import llm as L
from mpmf_plus.bench import prompts as PR
from mpmf_plus.bench import report as R
from mpmf_plus.bench import stats as S
from mpmf_plus.bench import textmetrics as T
from mpmf_plus.bench import pipeline as PL
from mpmf_plus.bench import runner as RU

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


# ------------------------------------------------------------------ stats
def test_mcnemar_exact_known_values():
    assert S.mcnemar_exact(0, 0) == 1.0
    assert S.mcnemar_exact(5, 5) == 1.0
    assert abs(S.mcnemar_exact(0, 5) - 0.0625) < 1e-9          # 2 * (1/32)
    assert abs(S.mcnemar_exact(11, 10) - 1.0) < 1e-9


def test_paired_binary_ignores_nulls_and_counts_discordant():
    a = {"q1": 0, "q2": 1, "q3": None, "q4": 0, "q5": 1}
    b = {"q1": 1, "q2": 1, "q3": 1, "q4": 1, "q5": 0}
    r = S.paired_binary(a, b)
    assert (r["n"], r["b"], r["c"]) == (4, 2, 1)
    assert r["both_right"] == 1 and r["both_wrong"] == 0


def test_wilson_bounds():
    lo, hi = S.wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert S.wilson(0, 0) == (0.0, 0.0)
    assert S.wilson(10, 10)[1] == pytest.approx(1.0)


def test_signflip_deterministic_and_sensible():
    d = [0.5] * 20
    p1, p2 = S.signflip_p(d), S.signflip_p(d)
    assert p1 == p2 and p1 < 0.001
    assert S.signflip_p([0.0] * 5) == 1.0
    assert S.signflip_p([0.3, -0.3] * 10) > 0.5
    lo, hi = S.bootstrap_ci([1, 0, 1, 1, 0, 1, 0, 1])
    assert 0 <= lo <= hi <= 1


# ------------------------------------------------------------------ text metrics
def test_locomo_f1_rules():
    assert T.f1_score("the red car", "red car") == 1.0
    assert T.locomo_f1(2, "7 May 2023", "7 May 2023") == 1.0
    assert T.locomo_f1(1, "pottery, camping", "camping, pottery") == 1.0
    assert T.locomo_f1(3, "tennis", "tennis; maybe golf") == 1.0           # only the part before ';'
    assert T.locomo_f1(5, "Not mentioned in the conversation", "") == 1.0
    assert T.locomo_f1(5, "It was a boat", "") == 0.0


def test_rouge_l_recall_and_bm25():
    r = T.rouge_l("oh yes the shell necklace was great", "shell necklace")
    assert r["r"] == 1.0 and 0 < r["p"] < 1
    assert T.rouge_l("", "x")["f"] == 0.0
    bm = T.BM25(["the cat sat", "dogs bark loudly", "a cat and a dog"])
    top = bm.top("cat", 2)
    assert {i for i, _ in top} == {0, 2} and bm.top("zebra", 3) == []


# ------------------------------------------------------------------ judges
def test_parsers():
    assert J.parse_label('{"label": "CORRECT"}') == 1
    assert J.parse_label("The answer matches. WRONG") == 0
    assert J.parse_label("no verdict here") is None
    assert J.parse_beam_score('{"score": 0.5, "reason": "x"}') == 0.5
    assert J.parse_beam_score('{"score": 7}') is None


def test_lme_prompt_per_type_and_abstention():
    p = J.lme_prompt("temporal-reasoning", "q?", "a", "r", False)
    assert "off-by-one" in p
    assert "updated answer" in J.lme_prompt("knowledge-update", "q?", "a", "r", False)
    assert "unanswerable" in J.lme_prompt("multi-session", "q?", "a", "r", True)
    with pytest.raises(NotImplementedError):
        J.lme_prompt("nonsense", "q", "a", "r", False)


def test_judge_failure_is_null_not_wrong():
    def boom(*a, **k):
        raise RuntimeError("down")
    q = D.Question("x", "locomo", "single-hop", "q", "g", "c", {"cat": 4})
    r = J.judge(q, "ans", boom, "m")
    assert r["verdict"] is None and "judge error" in r["raw"]


def test_beam_judge_mean_of_items():
    q = D.Question("x", "beam", "abstention", "q", "g", "c", {"rubric": ["mentions alpha", "mentions zebra"]})
    r = J.judge(q, "alpha only", L.fake_chat, "m")
    assert r["score"] == 0.5 and r["verdict"] == 1


# ------------------------------------------------------------------ loaders (synthetic data)
def _write_locomo(d):
    sess = lambda n: [{"speaker": "Ann" if i % 2 == 0 else "Bob", "dia_id": "D%d:%d" % (n, i + 1),
                       "text": "hello number %d about pottery and camping trips" % i} for i in range(6)]
    conv = {"sample_id": "conv-t1",
            "conversation": {"speaker_a": "Ann", "speaker_b": "Bob", "session_1": sess(1), "session_1_date_time": "1:56 pm on 8 May, 2023",
                             "session_2": sess(2), "session_2_date_time": "9:30 am on 21 June, 2023"},
            "qa": [{"question": "What does Ann like?", "answer": "pottery", "evidence": ["D1:1"], "category": 4},
                   {"question": "When did they talk first?", "answer": "8 May 2023", "evidence": ["D1:2"], "category": 2},
                   {"question": "Anything adversarial?", "adversarial_answer": "x", "evidence": [], "category": 5}]}
    with open(os.path.join(d, "locomo10.json"), "w") as fh:
        json.dump([conv], fh)


def test_locomo_loader_dates_categories_and_cat5_excluded(tmp_path):
    _write_locomo(str(tmp_path))
    corp, qs = D.load("locomo", str(tmp_path))
    assert {q.category for q in qs} == {"single-hop", "temporal"}      # adversarial dropped by default
    assert len(D.load("locomo", str(tmp_path), include_cat5=True)[1]) == 3
    t = corp["conv-t1"].turns
    assert t[0].date == "2023-05-08T13:56:00Z" and t[6].date.startswith("2023-06-21T09:30")


def test_dmr_loader_handles_string_and_list_fields(tmp_path):
    row = {"personas": [], "dialog": [{"text": "hi there", "id": "Speaker 1"}, {"text": "hello", "id": "Speaker 2"}],
           "metadata": str({"initial_data_id": "t_1", "session_id": 4}),
           "previous_dialogs": [{"dialog": [{"text": "I own two dogs", "id": "Speaker 1"}], "time_num": 2, "time_unit": "days",
                                 "time_back": "2 days 8 hours ago"}],
           "self_instruct": {"B": "How many dogs do I own?", "A": "two"}}
    with open(os.path.join(str(tmp_path), "msc_self_instruct.jsonl"), "w") as fh:
        fh.write(json.dumps(row) + "\n")
    corp, qs = D.load("dmr", str(tmp_path))
    assert qs[0].gold == "two" and qs[0].cid == "t_1"
    assert corp["t_1"].turns[0].date < corp["t_1"].turns[-1].date


def test_units_pairing_and_sample_prefix():
    c = D.Corpus("c", [D.Turn(1, "User", "u1", "2024-01-01T00:00:00Z", "user"), D.Turn(1, "Assistant", "a1", "2024-01-01T00:00:01Z", "assistant"),
                       D.Turn(1, "User", "u2", "2024-01-01T00:00:02Z", "user")], "pair")
    us = D.units(c)
    assert [(u["user"], u["assistant"]) for u in us] == [("u1", "a1"), ("u2", None)]
    qs = [D.Question("q%03d" % i, "locomo", "c", "q", "a", "x") for i in range(50)]
    full, pre = D.seeded_order(qs, 3), D.sample(qs, 10, 3)
    assert pre == full[:10] and D.ids_md5(pre) == D.ids_md5(D.sample(qs, 10, 3))
    assert D.ids_md5(D.sample(qs, 10, 4)) != D.ids_md5(pre)


def test_ingest_calls_mirror_sync_turn_digest():
    c = D.Corpus("c1", [D.Turn(1, "A", "x" * 700, "2024-01-01T00:00:00Z"), D.Turn(1, "B", "y", "2024-01-01T00:00:01Z"),
                        D.Turn(2, "A", "z", "2024-01-02T00:00:00Z")], "single")
    calls = PL.ingest_calls(c)
    assert calls[0]["turn_text"].startswith("[session=s001_c1 t=") and "USER: A: " in calls[0]["turn_text"]
    assert len(calls[0]["turn_text"].split("USER: ", 1)[1]) <= 600
    assert calls[1]["context"] and calls[1]["msg_index"] == 1
    assert calls[2]["context"] == [] and calls[2]["msg_index"] == 0     # context never crosses a session


def test_bm25_context_budget():
    c = D.Corpus("c", [D.Turn(1, "A", "pottery class on friday " + "w" * 900, "2024-01-01T00:00:00Z") for _ in range(10)], "single")
    ctx = PL.bm25_context(PL.bm25_index(c), "pottery friday", k=8, budget=3500, item=1200)
    assert 0 < len(ctx) <= 3500 and ctx.startswith("[2024-01-01]")


# ------------------------------------------------------------------ results.jsonl -> summary.md
def _row(i, arm, v, cat="single-hop", sc=None):
    return {"id": i, "category": cat, "arm": arm, "verdict": v, "score": sc, "cost": 0.001}


def test_summarize_on_truncated_results(tmp_path):
    out = str(tmp_path)
    rows = []
    for i in range(12):
        rows += [_row("q%d" % i, "bm25", i % 2, sc=0.2), _row("q%d" % i, "off", 1, sc=0.5), _row("q%d" % i, "on", 1 if i < 9 else 0, sc=0.6)]
    rows.append(_row("q12", "bm25", 1))                                  # only one arm scored: not paired
    with open(os.path.join(out, "results.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.write('{"id": "q13", "category": "single-hop", "arm": "o')     # killed mid-write
    with open(os.path.join(out, "prereg.json"), "w") as fh:
        json.dump({"benchmark": "locomo", "n_planned": 50, "pool": 1000, "seed": 1, "ids_md5": "abc", "reader": "r", "judge": "j"}, fh)
    r = subprocess.run([sys.executable, "-m", "mpmf_plus.bench", "summarize", "locomo", "--out", out],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=SRC))
    assert r.returncode == 0, r.stderr
    text = open(os.path.join(out, "summary.md")).read()
    assert text.splitlines()[0] == "STATUS: IN PROGRESS n=12 of 50"
    assert "McNemar" in text and "sign-flip" in text and "single-hop" in text
    assert "How our reader and judge differ" in text
    A = R.analyse(R._read_jsonl(os.path.join(out, "results.jsonl")))
    assert A["n_paired"] == 12 and A["totals"]["off"] == (12, 12) and A["totals"]["on"] == (9, 12)
    assert A["cmp"]["on vs off (history)"]["c"] == 3 and A["cmp"]["on vs off (history)"]["b"] == 0
    r2 = subprocess.run([sys.executable, "-m", "mpmf_plus.bench", "summarize", "locomo", "--out", out, "--status", "PARTIAL, stopped at spend ceiling"],
                        capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=SRC))
    assert open(os.path.join(out, "summary.md")).read().splitlines()[0] == "STATUS: PARTIAL, stopped at spend ceiling"


def test_summarize_empty_dir_does_not_crash(tmp_path):
    text = R.write_summary(str(tmp_path), "IN PROGRESS")
    assert text.startswith("STATUS: IN PROGRESS n=0 of 0") and "No question has been scored" in text


def test_prereg_is_written_once_and_never_changed(tmp_path):
    class A:                     # minimal args
        reader_tokens, judge, cap, rerank, fake_model, cids = 400, None, 8.0, False, True, None
    qs = [D.Question("q%d" % i, "locomo", "c", "q", "a", "x", {"cat": 4}) for i in range(5)]
    pre = RU.prereg("locomo", qs, qs, 7, 5, A)
    ev = str(tmp_path / "ev.txt")
    RU.write_prereg_once(str(tmp_path), ev, pre)
    first = open(os.path.join(str(tmp_path), "prereg.json")).read()
    RU.write_prereg_once(str(tmp_path), ev, pre)
    assert open(os.path.join(str(tmp_path), "prereg.json")).read() == first
    assert open(ev).read().count("PRE-REGISTRATION") == 1 and pre["ids_md5"] in open(ev).read()
    other = RU.prereg("locomo", qs, qs[:3], 7, 3, A)
    with pytest.raises(SystemExit):
        RU.write_prereg_once(str(tmp_path), ev, other)


# ------------------------------------------------------------------ end to end with the fake model
def test_fake_model_end_to_end_resume_and_files(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    _write_locomo(str(data))
    env = dict(os.environ, PYTHONPATH=SRC)
    cmd = [sys.executable, "-m", "mpmf_plus.bench", "run", "locomo", "--data", str(data), "--out", str(out), "--n", "2", "--seed", "3",
           "--fake-model", "--workers", "2", "--evidence", str(tmp_path / "ev.txt")]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-1500:]
    rows = [json.loads(l) for l in open(out / "results.jsonl")]
    assert len(rows) == 6 and {r_["arm"] for r_ in rows} == {"bm25", "off", "on"}
    assert all(set(r_) == {"id", "category", "arm", "verdict", "score", "cost"} for r_ in rows)
    assert (out / "summary.md").read_text().splitlines()[0] == "STATUS: COMPLETE"
    assert (out / "results.json").exists() and (out / "prereg.json").exists()
    assert not list((out / "work" / "stores").glob("*.mpmf"))             # bench store deleted once scored
    assert "PRE-REGISTRATION" in (tmp_path / "ev.txt").read_text()
    # resume: nothing is re-run, nothing is appended
    r2 = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r2.returncode == 0, r2.stderr[-800:]
    assert len((out / "results.jsonl").read_text().splitlines()) == 6


def test_dry_run_writes_no_results(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    _write_locomo(str(data))
    r = subprocess.run([sys.executable, "-m", "mpmf_plus.bench", "run", "locomo", "--data", str(data), "--out", str(out), "--n", "2", "--dry-run"],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=SRC))
    assert r.returncode == 0, r.stderr[-800:]
    plan = json.loads(r.stdout)
    assert plan["planned"] == 2 and plan["ids_md5"] and not (out / "results.jsonl").exists() and not (out / "prereg.json").exists()


def test_fake_chat_is_deterministic_and_costs_nothing():
    a = L.fake_chat("x\nQuestion: what pottery?")
    assert a == L.fake_chat("x\nQuestion: what pottery?") and a["cost"] == 0.0
    assert PR.reader_prompt(D.Question("q", "beam", "c", "Q?", "g", "x"), "").count("(nothing was retrieved)") == 1
