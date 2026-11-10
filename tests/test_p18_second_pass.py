"""P18: second-pass retrieval — flag-off identity, dedupe against the prefetch block, per-turn call
limit, reset by the next prefetch. Fake provider module, no store, no Ollama, no network."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mpmf_plus import block as B                  # noqa: E402
from mpmf_plus import hermes_adapter as H         # noqa: E402
from mpmf_plus import paths                       # noqa: E402
from mpmf_plus import second_pass as SP           # noqa: E402

TEXT = {i: "memory number %d says something specific about topic %d" % (i, i) for i in range(1, 40)}


class FakeMem:
    def get_name(self, n):
        return "n%d" % n

    def get_wiki(self, n):
        return {}


class FakeProv:
    """recall() returns node ids 1..top_n in order (rows shape used by block.wrapper_recall)."""
    calls = []

    @staticmethod
    def embed_query(t):
        return [0.0]

    @staticmethod
    def recall(q, top_n=8, **kw):
        FakeProv.calls.append((q, top_n))
        return [{"node_id": i, "text": TEXT[i], "summary": TEXT[i]} for i in range(1, top_n + 1)]

    @staticmethod
    def _open():
        class O:
            mem = FakeMem()
        return O()


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    def write(**kw):
        p = tmp_path / "recall.config.json"
        p.write_text(json.dumps(kw))
        monkeypatch.setenv("MPMF_RECALL_CONFIG", str(p))
        monkeypatch.setenv("MPMF_HOME", str(tmp_path))
        monkeypatch.delenv("MPMF_SECOND_PASS", raising=False)
    write()
    FakeProv.calls = []
    return write


def test_default_off():
    assert paths.DEFAULTS["second_pass"] is False
    assert paths.DEFAULTS["second_pass_max_calls"] == 2


def test_flag_off_schema_and_handler_are_the_old_ones(cfg, monkeypatch):
    cfg(second_pass=False)
    monkeypatch.setattr(SP, "query", lambda *a, **k: pytest.fail("second_pass.query called with flag off"))
    prov = H.MPMFPlusProvider()
    sch = {s["name"]: s for s in prov.get_tool_schemas()}
    assert sch["mpmf_recall"]["description"].startswith("Ad-hoc deep lookup")
    from mpmf_plus import provider as P
    monkeypatch.setattr(P, "recall", lambda q, top_n=10, **k: [{"id": 7, "text": "x" * 500}])
    out = json.loads(prov.handle_tool_call("mpmf_recall", {"query": "q"}))
    assert out == {"hits": [{"id": 7, "text": "x" * 300}]}


def test_flag_on_changes_description_only_when_on(cfg):
    cfg(second_pass=True)
    d = H.MPMFPlusProvider().get_tool_schemas()[0]["description"]
    assert d == SP.DESCRIPTION and "BEFORE answering" in d


def test_query_excludes_already_shown_and_adds_only_new(cfg):
    cfg(second_pass=True, memory_render="full")
    st = SP.TurnState()
    st.reset([1, 2, 3])                              # prefetch block already carries 1..3
    out = SP.query(st, "something else", 4, provider_mod=FakeProv)
    assert "## [4]" in out and "## [5]" in out and "## [6]" in out and "## [7]" in out
    assert "## [1]" not in out and "## [2]" not in out and "## [3]" not in out
    assert "## [8]" not in out                         # k=4 new memories
    out2 = SP.query(st, "again", 2, provider_mod=FakeProv)
    assert "## [4]" not in out2 and "## [8]" in out2    # earlier second-pass results are not repeated


def test_call_limit_and_reset_by_next_prefetch(cfg):
    cfg(second_pass=True, second_pass_max_calls=2, memory_render="full")
    st = SP.TurnState()
    st.reset([])
    SP.query(st, "a", 2, provider_mod=FakeProv)
    SP.query(st, "b", 2, provider_mod=FakeProv)
    n = len(FakeProv.calls)
    third = SP.query(st, "c", 2, provider_mod=FakeProv)
    assert "limit reached" in third and len(FakeProv.calls) == n    # no recall was run
    st.reset([])                                                    # next turn
    assert "## [1]" in SP.query(st, "d", 2, provider_mod=FakeProv)


def test_nothing_new_is_said_so(cfg):
    cfg(second_pass=True, memory_render="full")
    st = SP.TurnState()
    st.reset(range(1, 40))
    assert SP.query(st, "x", 3, provider_mod=FakeProv).startswith("No new memories")


def test_empty_query_and_failure_are_loud(cfg, monkeypatch):
    cfg(second_pass=True)
    st = SP.TurnState()
    assert "query is required" in SP.query(st, "  ", provider_mod=FakeProv)
    monkeypatch.setattr(B, "wrapper_recall", lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        SP.query(st, "q", provider_mod=FakeProv)


def test_node_id_lookup_unchanged_with_flag_on(cfg, monkeypatch):
    cfg(second_pass=True)
    from mpmf_plus import provider as P

    class M:
        def get_text(self, n):
            return "full text %d" % n
    monkeypatch.setattr(P, "_open", lambda: type("O", (), {"mem": M()})())
    out = json.loads(H.MPMFPlusProvider().handle_tool_call("mpmf_recall", {"node_id": 5}))
    assert out == {"found": True, "node_id": 5, "text": "full text 5"}
