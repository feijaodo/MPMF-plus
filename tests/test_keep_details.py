"""p9 — keep_details: every proper noun / number / date of the evidence survives into the fact."""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mpmf_plus import keep_details as K            # noqa: E402
from mpmf_plus import paths                         # noqa: E402
from mpmf_plus import ingest as ing                 # noqa: E402
from test_r83_ingest import FakeMem, fake_vec, reply, DIM   # noqa: E402

EV = "I bought a Lenovo ThinkPad X1 for 1,450 euros in March 2024."


def test_flag_default_off():
    assert paths.DEFAULTS["keep_details"] is False


def test_detects_dropped_specifics():
    m = K.missing_specifics(EV, "User bought a laptop.")
    assert {"Lenovo", "ThinkPad", "X1", "March", "1450"} <= set(m)


def test_complete_sentence_passes():
    assert K.missing_specifics(EV, "User bought a Lenovo ThinkPad X1 for 1,450 euros in March 2024.") == []
    assert K.missing_specifics("We met on 2023-05-04 at 14:30.", "User met on 2023-05-04 at 14:30.") == []
    assert K.missing_specifics("I drive a 2024 Volvo V60.", "User drives a 2024 Volvo V60.") == []


def test_sentence_start_and_pronouns_are_not_proper_nouns():
    assert K.specifics("Okay, so I think we should go.")["proper"] == []


def test_fact_date_can_carry_the_date():
    assert K.missing_specifics("It happened on 2024-05-12.", "User had an accident.", "2024-05-12") == []


def _ingest(store, calls, flag, text=EV):
    def call(body):
        calls.append(body)
        if len(calls) == 1:
            return reply({"s": "User bought a laptop.", "ev": EV, "entity": "laptop"})
        return reply({"s": "User bought a Lenovo ThinkPad X1 for 1,450 euros in March 2024.",
                      "ev": EV, "entity": "laptop"})
    return ing.ingest(text, store=store, call_model=call, embed_fn=fake_vec, log_path="",
                      embed_dim=DIM, keep_details=flag)


def test_off_makes_one_call_and_keeps_original(tmp_path):
    calls = []
    res = _ingest(FakeMem(str(tmp_path / "a.mpmf")), calls, False)
    assert len(calls) == 1 and res["written"][0]["text"] == "User bought a laptop."
    assert res["keep_details"]["retried"] is False


def test_on_retries_once_and_replaces(tmp_path):
    calls = []
    res = _ingest(FakeMem(str(tmp_path / "b.mpmf")), calls, True)
    assert len(calls) == 2 and "REVIEW" in calls[1] and "Lenovo" in calls[1]
    assert "ThinkPad X1" in res["written"][0]["text"]
    assert res["keep_details"] == {"bad": 1, "fixed": 1, "retried": True}


def test_on_never_retries_twice_and_never_adds_facts(tmp_path):
    calls = []

    def call(body):
        calls.append(body)
        return json.dumps({"facts": [dict(msg_index=0, entity="laptop", fact_sentence="User bought a laptop.",
                                          evidence_sentence=EV),
                                     dict(msg_index=0, entity="x", fact_sentence="Invented Thing 99.",
                                          evidence_sentence="something else")]})
    res = ing.ingest(EV, store=FakeMem(str(tmp_path / "c.mpmf")), call_model=call, embed_fn=fake_vec,
                     log_path="", embed_dim=DIM, keep_details=True)
    assert len(calls) == 2
    assert [w["text"] for w in res["written"]] == ["User bought a laptop.", "Invented Thing 99."]


def test_retry_failure_keeps_original(tmp_path):
    calls = []

    def call(body):
        calls.append(body)
        if len(calls) > 1:
            raise RuntimeError("boom")
        return reply({"s": "User bought a laptop.", "ev": EV})
    res = ing.ingest(EV, store=FakeMem(str(tmp_path / "d.mpmf")), call_model=call, embed_fn=fake_vec,
                     log_path="", embed_dim=DIM, keep_details=True)
    assert res["written"][0]["text"] == "User bought a laptop."


def test_config_flag_drives_default(tmp_path, monkeypatch):
    cfg = tmp_path / "recall.config.json"
    cfg.write_text(json.dumps({"keep_details": True}))
    monkeypatch.setenv("MPMF_RECALL_CONFIG", str(cfg))
    assert K.enabled() is True
    monkeypatch.setenv("MPMF_RECALL_CONFIG", str(tmp_path / "none.json"))
    assert K.enabled() is False


def test_shared_evidence_is_not_audited():
    ev = "Winifred had a cousin named Clement King, who died in Canada."
    facts = [{"sentence": "Winifred had a cousin named Clement King.", "evidence": ev},
             {"sentence": "Clement King died in Canada.", "evidence": ev}]
    assert K.audit(facts) == []
    assert K.audit(facts[:1]) == [(0, ["Canada"])]       # a sole fact must carry all of it
