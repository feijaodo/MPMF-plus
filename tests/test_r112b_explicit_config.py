"""r112b: explicit rerank/prefetch values, historical_rumination -> stage_lesson,
and the intake-gate "staged" metric rename.

Written BEFORE the r112b change (red first): the zero-literal reads and the
wrapper delegation fail on r112, and pass after r112b.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mpmf_plus import memory as mem  # noqa: E402
from mpmf_plus import provider as prov  # noqa: E402


def _cfg(monkeypatch, mod, cfg):
    monkeypatch.setattr(mod, "_recall_flags", lambda: cfg)


# ---- (1) explicit rerank / prefetch values -------------------------------
def test_prefetch_budget_reads_zero_literally(monkeypatch):
    _cfg(monkeypatch, prov, {"prefetch_budget_ms": 0})
    assert prov._prefetch_budget_ms() == 0.0
    _cfg(monkeypatch, mem, {"prefetch_budget_ms": 0})
    assert mem._prefetch_budget_ms() == 0.0


def test_rerank_skip_reads_zero_literally(monkeypatch):
    _cfg(monkeypatch, prov, {"rerank_skip_after_ms": 0})
    assert prov._rerank_skip_after_ms() == 0.0
    _cfg(monkeypatch, mem, {"rerank_skip_after_ms": 0})
    assert mem._rerank_skip_after_ms() == 0.0


def test_explicit_values_are_honoured(monkeypatch):
    _cfg(monkeypatch, prov, {"prefetch_budget_ms": 6500, "rerank_skip_after_ms": 4500})
    assert prov._prefetch_budget_ms() == 6500.0
    assert prov._rerank_skip_after_ms() == 4500.0


def test_unset_falls_back_to_defaults(monkeypatch):
    _cfg(monkeypatch, prov, {})
    assert prov._prefetch_budget_ms() == 6500.0
    assert prov._rerank_skip_after_ms() == 4500.0


# ---- (2) historical_rumination goes through stage_lesson -----------------
def _load_hist():
    p = Path.home() / ".hermes" / "scripts" / "historical_rumination.py"
    spec = importlib.util.spec_from_file_location("hr_under_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wrapper_routes_through_stage_lesson(monkeypatch):
    hr = _load_hist()
    from mpmf_plus import cognition

    seen = {}

    def fake_stage(text, **kw):
        seen["text"] = text
        seen.update(kw)
        return {"action": "staged", "name": "cog_stage_abc123"}

    monkeypatch.setattr(cognition, "stage_lesson", fake_stage)
    got = hr.store_reanalysis_lesson(
        {"situation": "S", "principle": "P", "rule": "R"}, "sess1")
    assert got == "cog_stage_abc123"
    assert seen["origin"] == "historical_rumination"
    assert seen["session"] == "sess1"
    assert "S" in seen["text"] and "R" in seen["text"]


def test_wrapper_returns_empty_when_skipped(monkeypatch):
    hr = _load_hist()
    from mpmf_plus import cognition
    monkeypatch.setattr(cognition, "stage_lesson",
                        lambda text, **kw: {"action": "skipped", "reason": "empty_text"})
    assert hr.store_reanalysis_lesson({"principle": "P"}, "s") == ""


# ---- (3) "staged" metric renamed ----------------------------------------
def test_intake_gate_summary_has_staged_in_and_left():
    src = (ROOT / "src/mpmf_plus/cognition.py").read_text()
    assert '"staged_in"' in src
    assert '"staged_left"' in src
    assert 'summary["staged"] =' not in src
    assert "staged_left=%(staged_left)d" in src
