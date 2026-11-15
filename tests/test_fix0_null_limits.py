"""fix0: every reader of the key endpoint's limit / limit_remaining / usage_daily handles null.

The OpenRouter key endpoint returns these as JSON null when the key has no spending limit;
dict.get(k, 0) does not apply its default for a present-but-null key. The readers live outside the
package (ops scripts), so the modules are imported by path; the tests skip when a script is not on
this box. Synthetic responses only, no network, no model call.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HOME = Path.home()
HR_PATH = HOME / ".hermes" / "scripts" / "historical_rumination.py"
RUNNER_DIR = HOME / "ops-mpmf" / "runner"
CHECKS_DIR = HOME / "ops-mpmf" / "checks"


def _load(name, path, extra_sys_path=None):
    if extra_sys_path and str(extra_sys_path) not in sys.path:
        sys.path.insert(0, str(extra_sys_path))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def hr():
    if not HR_PATH.is_file():
        pytest.skip("historical_rumination.py not on this box")
    return _load("hr_fix0", HR_PATH)


@pytest.fixture(scope="module")
def r3():
    if not (RUNNER_DIR / "runner3.py").is_file() or not (RUNNER_DIR / "budget_baseline.json").is_file():
        pytest.skip("ops runner not on this box")
    return _load("runner3_fix0", RUNNER_DIR / "runner3.py", RUNNER_DIR)


# --------------------------------------------------------------------------- historical_rumination
NULL = {"limit": None, "limit_remaining": None, "usage_daily": 1.5}


def test_hr_parse_null_stays_none(hr):
    b = hr.parse_balance(NULL)
    assert b == {"limit_remaining": None, "limit": None, "usage_daily": 1.5}


def test_hr_parse_absent_is_zero_as_before(hr):
    assert hr.parse_balance({}) == {"limit_remaining": 0.0, "limit": 0.0, "usage_daily": 0.0}
    assert hr.parse_balance(None) == {"limit_remaining": 0.0, "limit": 0.0, "usage_daily": 0.0}


def test_hr_parse_numeric_and_string_numeric(hr):
    b = hr.parse_balance({"limit_remaining": 12.5, "limit": "20", "usage_daily": "1.25"})
    assert b == {"limit_remaining": 12.5, "limit": 20.0, "usage_daily": 1.25}


@pytest.mark.parametrize("bad", ["abc", "", [], {}, True, float("nan"), float("inf")])
def test_hr_parse_garbage_is_zero_not_a_crash(hr, bad):
    assert hr.parse_balance({"limit_remaining": bad})["limit_remaining"] == 0.0


def test_hr_plan_budget_null_uses_default_max(hr):
    budget, lines = hr.plan_budget(hr.parse_balance(NULL), 0.10)
    assert budget == hr.DEFAULT_BUDGET_MAX
    text = "\n".join(lines)
    assert "no limit set" in text
    assert "inf" not in text.lower().replace("information", "")
    assert "$1.50" in text


def test_hr_plan_budget_numeric_unchanged(hr):
    budget, lines = hr.plan_budget({"limit_remaining": 20.0, "limit": 50.0, "usage_daily": 1.0}, 0.10)
    assert budget == pytest.approx(2.0)
    assert "$20.00 remaining" in lines[0] and "$50.00 daily limit" in lines[0]
    # large balance is capped at the default max (value of the cap unchanged: 5.00)
    budget, _ = hr.plan_budget({"limit_remaining": 1000.0, "limit": 1000.0, "usage_daily": 0.0}, 0.10)
    assert budget == hr.DEFAULT_BUDGET_MAX == 5.00


def test_hr_plan_budget_string_numeric_through_parse(hr):
    budget, _ = hr.plan_budget(hr.parse_balance({"limit_remaining": "30", "limit": None, "usage_daily": None}), 0.10)
    assert budget == pytest.approx(3.0)


def test_hr_plan_budget_null_usage_and_limit_do_not_crash(hr):
    _, lines = hr.plan_budget({"limit_remaining": 10.0, "limit": None, "usage_daily": None}, 0.5)
    assert "unknown" in lines[0]


def test_hr_check_balance_parses_http_body(hr, monkeypatch):
    import types

    class Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"data": NULL}

    fake = types.ModuleType("requests")
    fake.get = lambda *a, **k: Resp()
    monkeypatch.setitem(sys.modules, "requests", fake)
    assert hr.check_openrouter_balance("k")["limit_remaining"] is None


def test_hr_fake_balance_hook_inert_unless_set_and_dry_only(hr, monkeypatch):
    monkeypatch.delenv("MPMF_FAKE_BALANCE_JSON", raising=False)
    assert hr._fake_balance() is None
    monkeypatch.setenv("MPMF_FAKE_BALANCE_JSON", json.dumps(NULL))
    assert hr._fake_balance()["limit_remaining"] is None

    class Args:
        dry_run = False
        budget_ratio = 0.1

    # not a dry run: the hook is never consulted, the real balance call is made (stubbed to fail loudly)
    def boom(_k):
        raise RuntimeError("real balance path taken")

    monkeypatch.setattr(hr, "check_openrouter_balance", boom)
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    with pytest.raises(RuntimeError, match="real balance path taken"):
        hr._run_locked(Args())


# --------------------------------------------------------------------------- runner3.usage_now / fmt_rem
@pytest.mark.parametrize("resp,usage,rem", [
    ({"usage": 10.0, "limit": None, "limit_remaining": None}, 10.0, float("inf")),   # null
    ({"usage": 10.0}, 10.0, float("inf")),                                            # absent
    ({"usage": 10.0, "limit_remaining": 7.5}, 10.0, 7.5),                             # numeric
    ({"usage": "10.5", "limit_remaining": "7.5"}, 10.5, 7.5),                         # string-numeric
])
def test_runner3_usage_now(r3, monkeypatch, resp, usage, rem):
    monkeypatch.setattr(r3.R, "budget", lambda: resp)
    assert r3.usage_now() == (usage, rem)


def test_runner3_usage_now_missing_usage_raises(r3, monkeypatch):
    monkeypatch.setattr(r3.R, "budget", lambda: {"limit_remaining": None})
    with pytest.raises(ValueError):
        r3.usage_now()


@pytest.mark.parametrize("rem,expect", [
    (float("inf"), "no limit set"), (None, "no limit set"), (7.5, "$7.50"), ("7.5", "$7.50"),
])
def test_runner3_fmt_rem(r3, rem, expect):
    out = r3.fmt_rem(rem)
    assert expect in out
    assert "inf" not in out


def test_runner3_money_stop_is_lifetime_cap_when_null(r3, monkeypatch):
    """no balance floor when limit_remaining is null: inf is never below the floor."""
    monkeypatch.setattr(r3.R, "budget", lambda: {"usage": 3.0, "limit_remaining": None})
    _, rem = r3.usage_now()
    assert not (rem < r3.FLOOR)
    assert r3.FLOOR == 5.0  # the value of the floor is unchanged


def test_old_runner_budget_readers_survive_null(monkeypatch):
    """runner.py reads usage_daily with float(x.get(k) or 0) (null-safe)."""
    if not (RUNNER_DIR / "runner.py").is_file():
        pytest.skip("ops runner not on this box")
    src = (RUNNER_DIR / "runner.py").read_text()
    assert 'float(before.get("usage_daily", 0))' not in src
    assert 'float(after.get("usage_daily", 0))' not in src
    assert 'float(budget().get("usage_daily", 0))' not in src
    assert float({"usage_daily": None}.get("usage_daily") or 0) == 0.0


# --------------------------------------------------------------------------- no unguarded format left
def test_no_unguarded_limit_remaining_format_in_runner_and_checks():
    files = []
    if RUNNER_DIR.is_dir():
        files += sorted(RUNNER_DIR.glob("runner*.py"))
    if CHECKS_DIR.is_dir():
        files += [CHECKS_DIR / "queue_preflight.py", CHECKS_DIR / "launch_preflight.py"]
    bad = []
    for f in files:
        if not f.is_file():
            continue
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if "${rem:.2f}" in line or "['limit_remaining']:.2f" in line or "['usage_daily']:.2f" in line or "['limit']:.2f" in line:
                bad.append(f"{f.name}:{n}")
    assert bad == [], bad
    # historical_rumination reads the raw balance dict only inside plan_budget (guarded, tested above)
    if HR_PATH.is_file():
        assert sum('balance["limit' in ln for ln in HR_PATH.read_text().splitlines()) == 1


# --------------------------------------------------------------------------- _run_locked, dry-run path
def test_run_locked_dry_run_with_null_balance_substituted(hr, monkeypatch, capsys):
    """The balance block runs in --dry-run when the hook is set: no network, no model call, no crash."""
    monkeypatch.setenv("MPMF_FAKE_BALANCE_JSON", json.dumps(NULL))
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setattr(hr, "check_openrouter_balance", lambda k: pytest.fail("network balance call in dry-run"))
    monkeypatch.setattr(hr, "discover_sessions_from_db", lambda n: [])
    monkeypatch.setattr(hr, "discover_sessions", lambda d, n: [])

    class Args:
        dry_run = True
        budget_ratio = 0.10
        max_sessions = 3

    assert hr._run_locked(Args()) == 0
    out = capsys.readouterr().out
    assert "Balance: no limit set" in out
    assert "$5.00" in out
    assert "inf" not in out.lower().replace("information", "")


def test_run_locked_dry_run_without_hook_skips_balance(hr, monkeypatch, capsys):
    monkeypatch.delenv("MPMF_FAKE_BALANCE_JSON", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setattr(hr, "check_openrouter_balance", lambda k: pytest.fail("network balance call in dry-run"))
    monkeypatch.setattr(hr, "discover_sessions_from_db", lambda n: [])
    monkeypatch.setattr(hr, "discover_sessions", lambda d, n: [])

    class Args:
        dry_run = True
        budget_ratio = 0.10
        max_sessions = 3

    assert hr._run_locked(Args()) == 0
    out = capsys.readouterr().out
    assert "Balance:" not in out and "DRY RUN" in out
