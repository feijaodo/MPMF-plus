"""Unit tests for the auto-ingestion logic (no model calls)."""
import os, sys, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mpmf.ingestion.template import QuestionTemplate, ACTIVE, PROBATION, RETIRED
from mpmf.ingestion.extract import Extractor, Answer, Relation
from mpmf.ingestion import decay


def test_template_seed_and_persist():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "qt.json")
        t = QuestionTemplate(p)
        assert t.stats()["total"] >= 20
        assert os.path.exists(p)
        # reload
        t2 = QuestionTemplate(p)
        assert t2.stats()["total"] == t.stats()["total"]


def test_template_evolution():
    with tempfile.TemporaryDirectory() as d:
        t = QuestionTemplate(os.path.join(d, "qt.json"), t_min=2)
        qid = next(iter(t.questions))
        # ask 3 times, never retrieved -> probation
        for _ in range(3):
            t.record_asked(qid, 1)
        ch = t.evolve()
        assert t.questions[qid].status == PROBATION
        # retrieval rescues it
        t.record_retrieval(qid, used=True)
        assert t.questions[qid].status == ACTIVE
        assert t.questions[qid].usefulness_score > 0.5


def test_probation_to_retired():
    with tempfile.TemporaryDirectory() as d:
        t = QuestionTemplate(os.path.join(d, "qt.json"), t_min=2)
        qid = next(iter(t.questions))
        for _ in range(3):
            t.record_asked(qid, 0)
        t.evolve()
        assert t.questions[qid].status == PROBATION
        for _ in range(3):
            t.record_asked(qid, 0)
        t.evolve()
        assert t.questions[qid].status == RETIRED


def test_decay_math():
    now = 1_000_000.0
    wiki = {decay.KEY_BASE: "1.0", decay.KEY_REINFORCED: str(int(now))}
    # fresh
    assert abs(decay.effective_weight(wiki, now) - 1.0) < 1e-6
    # one decay period later -> * 0.95
    later = now + decay.DECAY_PERIOD
    assert abs(decay.effective_weight(wiki, later) - 0.95) < 1e-3
    # reinforce resets timer and bumps base
    upd = decay.reinforce(wiki, later)
    assert float(upd[decay.KEY_BASE]) == 1.0  # capped
    assert int(upd[decay.KEY_REINFORCED]) == int(later)


def test_extractor_parse():
    raw = '```json\n{"answer": "User prefers concise replies", "entities": ["User"], "relations": [{"from":"User","to":"replies","type":"PREFERS"}], "confidence": 0.9}\n```'
    a = Extractor._parse(raw)
    assert a is not None and not a.is_none
    assert a.answer.startswith("User prefers")
    assert a.entities == ["User"]
    assert len(a.relations) == 1 and a.relations[0].rtype == "PREFERS"
    assert a.confidence == 0.9


def test_extractor_none():
    a = Extractor._parse('{"answer": "NONE", "entities": [], "relations": [], "confidence": 0.0}')
    assert a is not None and a.is_none


def test_extractor_garbage():
    assert Extractor._parse("the model rambled with no json") is None


def test_concise_name():
    from mpmf.ingestion.ingest import _concise_name
    # prefers a short salient entity over the full sentence
    n = _concise_name(
        "The problem being solved was the need for cost-conscious animations",
        ["cost-conscious animations", "ballerina game"])
    assert len(n) <= 40 and n.count("_") <= 3, n
    assert "the_problem_being_solved" not in n
    # no usable entity -> significant tokens from answer, stopwords dropped
    n2 = _concise_name(
        "The decision was made to use grok-imagine-video for animations", [])
    assert "decision" in n2 and "was" not in n2.split("_"), n2
    assert len(n2.split("_")) <= 4
    # entity that's all stopwords falls through to answer tokens
    n3 = _concise_name("El's animations should use grok-imagine-video", ["the"])
    assert n3 and "the" != n3


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-q"])
