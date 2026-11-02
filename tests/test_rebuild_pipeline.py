"""Rebuild pipeline tests on a 3-conversation synthetic fixture (no network)."""
import json

import pytest

from mpmf_plus.rebuild import extraction as X
from mpmf_plus.rebuild import retag as R
from mpmf_plus.rebuild import stage2 as S

FIXTURE = [
    {"id": "convA", "start": "2024-04-21",
     "messages": [{"role": "user", "text": "I leased a Volvo on 2025-12-26."},
                  {"role": "user", "text": "Before that I drove a Subaru Outback for 15 years."}]},
    {"id": "convB", "start": "2024-05-02",
     "messages": [{"role": "user", "text": "If I had a cat I would name it Mittens."},
                  {"role": "user", "text": "My sister is 8."}]},
    {"id": "convC", "start": "2024-06-10",
     "messages": [{"role": "user", "text": "I want to buy a pickup truck someday."}]},
]


def _reply(facts):
    return json.dumps({"facts": facts})


GOOD = {
    "convA": _reply([
        {"fact_sentence": "User leased a Volvo on 2025-12-26.", "entity": "car",
         "msg_index": 0, "evidence": "I leased a Volvo on 2025-12-26.", "fact_date": "2025-12-26"},
        {"fact_sentence": "User drove a Subaru Outback for 15 years.", "entity": "car",
         "msg_index": 1, "evidence": "drove a Subaru Outback for 15 years", "fact_date": "2024-04-21"},
    ]),
    "convB": _reply([
        {"fact_sentence": "User would name a cat Mittens.", "entity": "cat",
         "msg_index": 0, "evidence": "If I had a cat I would name it Mittens.", "fact_date": "2024-05-02"},
        {"fact_sentence": "User has a sister who is 8.", "entity": "sister",
         "msg_index": 1, "evidence": "My sister is 8.", "fact_date": "2024-05-02"},
    ]),
    "convC": _reply([
        {"fact_sentence": "User wants to buy a pickup truck.", "entity": "car",
         "msg_index": 0, "evidence": "I want to buy a pickup truck someday.", "fact_date": "2024-06-10"},
    ]),
}


def test_extraction_statuses_and_failed_is_not_empty():
    calls = {"n": 0}

    def model(prompt):
        if "convC" in prompt or "pickup truck" in prompt:
            calls["n"] += 1
            return "{not json"
        return GOOD["convA"] if "Volvo" in prompt else GOOD["convB"]

    good = X.extract_conversation(FIXTURE[0], model)
    assert good["status"] == "ok" and len(good["facts"]) == 2
    bad = X.extract_conversation(FIXTURE[2], model)
    assert bad["status"] == X.FAILED and bad["facts"] == []      # never "empty"
    assert calls["n"] == 2                                       # one retry happened


def test_empty_reply_is_empty_not_failed():
    res = X.extract_conversation(FIXTURE[0], lambda p: _reply([]))
    assert res["status"] == "empty" and res["facts"] == []


def test_schema_fields_all_persisted():
    stored = []

    def make_node(name, text, summary, wiki):
        stored.append(wiki)
        return len(stored)

    X.persist_facts(object(), FIXTURE[0], X._parse_reply(GOOD["convA"]), make_node)
    required = set(X.REQUIRED_FIELDS)
    for wiki in stored:
        assert required.issubset(wiki.keys()), wiki.keys()


def test_kind_and_mode_rules():
    assert S.classify_text("User leased a Volvo on 2025-12-26.")["kind"] == "have"
    assert S.classify_text("User moved to Springfield.")["kind"] == "change"
    assert S.classify_text("User wants to buy a pickup truck.")["kind"] == "intent"
    assert S.classify_text("User is interested in weightlifting.")["kind"] == "interest"
    assert S.classify_text("User would name a cat Mittens.")["mode"] == "hypothetical"
    assert S.classify_text("User rented a Volvo for the weekend.")["mode"] == "temporary"
    # anchored rule: "are better" inside a question must not make it a 'have'
    assert S.classify_text("User asks whether motors are better.")["kind"] == "interest"


def test_newest_actual_change_have_wins_and_hypothetical_never():
    facts = S.classify_facts([
        {"sentence": "User leased a Volvo on 2025-12-26.", "entity": "car",
         "fact_date": "2025-12-26", "msg_index": 0},
        {"sentence": "User drove a Subaru Outback for 15 years.", "entity": "car",
         "fact_date": "2024-04-21", "msg_index": 1},
        {"sentence": "User wants to buy a pickup truck.", "entity": "car",
         "fact_date": "2024-06-10", "msg_index": 2},
        {"sentence": "User would like a Volvo V60.", "entity": "car",
         "fact_date": "2026-01-01", "msg_index": 3},
    ])
    cur = S.current_state(facts)
    assert "Volvo" in cur["sentence"], cur
    assert all(f["mode"] != "hypothetical" for f in [cur])


def test_states_get_is_a_in_the_same_call():
    links = []
    nodes = []

    def create_node(name, text, summary, wiki):
        nodes.append(name)
        return len(nodes)

    def link(src, dst, label):
        links.append((src, dst, label))

    facts = S.classify_facts([
        {"sentence": "User leased a Volvo on 2025-12-26.", "entity": "car",
         "fact_date": "2025-12-26", "msg_index": 0, "name": 99},
    ])
    out = S.build_states(None, facts, create_node, link)
    assert out["states"] == 1
    assert ("IS_A" in [l[2] for l in links])
    assert [l for l in links if l[2] == "DERIVED_FROM"][0][0] == 99


def test_retag_counts_flips_and_uses_context_window():
    facts = [{"sentence": "User would name a cat Mittens.", "_conv_id": "convB",
              "_msg_index": 1, "mode": "actual", "name": 1}]
    prompts = []

    def model(p):
        prompts.append(p)
        return json.dumps({"mode": "hypothetical", "why": "conditional"})

    msgs = ["hello", "If I had a cat I would name it Mittens.", "ok"]
    out = R.retag_modes(facts, lambda cid: msgs, model)
    assert out["flips"] == {"actual->hypothetical": 1}
    assert "[0]" in prompts[0] and "[1]" in prompts[0]      # two-before window included


def test_cap_is_enforced(monkeypatch):
    convs = FIXTURE
    with pytest.raises(X.BudgetExceeded):
        X.run_extraction(convs, lambda p: _reply([]), cap_usd=0.001, cost_per_call=0.002, workers=1)


def test_cli_help_exits_zero():
    from mpmf_plus.rebuild.cli import main
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
