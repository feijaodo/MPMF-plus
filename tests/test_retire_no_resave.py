"""r93: the intake gate must not re-save the whole store for a lesson whose
held state is unchanged (each save rewrites the entire store; ~270 of them
pushed historical_rumination past the cron's 300 s limit)."""
from mpmf_plus import cognition as cg


class _MM:
    def __init__(self, wiki):
        self._w = wiki

    def get_name(self, nid):
        return "cog_stage_x"

    def get_wiki(self, nid):
        return dict(self._w)


def _spy(monkeypatch):
    calls = []
    monkeypatch.setattr(cg, "_rw_update", lambda name, mutate: calls.append(name) or True)
    return calls


def test_unchanged_held_lesson_causes_no_store_save(monkeypatch):
    calls = _spy(monkeypatch)
    mm = _MM({"_status": "held", "_gate_reason": "sessions=1<2"})
    cg._retire(mm, 1, "held", reason="sessions=1<2")
    assert calls == []


def test_held_with_new_reason_still_saves(monkeypatch):
    calls = _spy(monkeypatch)
    mm = _MM({"_status": "held", "_gate_reason": "sessions=1<2"})
    cg._retire(mm, 1, "held", reason="vet_unsure")
    assert calls == ["cog_stage_x"]


def test_first_time_held_still_saves(monkeypatch):
    calls = _spy(monkeypatch)
    cg._retire(_MM({"_status": "hypothesis"}), 1, "held", reason="sessions=1<2")
    assert calls == ["cog_stage_x"]


def test_terminal_status_always_saves(monkeypatch):
    calls = _spy(monkeypatch)
    cg._retire(_MM({"_status": "held", "_gate_reason": "r"}), 1, "merged", target="t", reason="r")
    cg._retire(_MM({"_status": "held", "_gate_reason": "r"}), 1, "expired", reason="r")
    assert len(calls) == 2


def test_long_reason_compared_after_truncation(monkeypatch):
    calls = _spy(monkeypatch)
    long = "x" * 400
    cg._retire(_MM({"_status": "held", "_gate_reason": long[:300]}), 1, "held", reason=long)
    assert calls == []
