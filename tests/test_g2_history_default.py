"""g2: history_mode is ON by default; one resolver for every reader; the nightly revert writes false.

Synthetic fixtures only. Throwaway MPMF_HOME; no network, no live paths, no model call.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mpmf_plus import paths                      # noqa: E402
from mpmf_plus import provider as P              # noqa: E402
from mpmf_plus import block as B                 # noqa: E402
from mpmf_plus import history as H               # noqa: E402


@pytest.fixture
def home(tmp_path, monkeypatch):
    cfg = tmp_path / "recall.config.json"
    monkeypatch.setenv("MPMF_HOME", str(tmp_path))
    monkeypatch.setenv("MPMF_RECALL_CONFIG", str(cfg))
    monkeypatch.setenv("MPMF_HEALTH_DIR", str(tmp_path))
    monkeypatch.delenv("MPMF_HISTORY_MODE", raising=False)
    return cfg


def _readers():
    return {"paths.load_config": paths.load_config()["history_mode"],
            "paths.history_mode": paths.history_mode(),
            "provider._history_mode": P._history_mode(),
            "history.enabled": H.enabled()}


def test_defaults_and_example_say_on():
    assert paths.DEFAULTS["history_mode"] is True
    ex = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config.example.json")))
    assert ex["history_mode"] is True


def test_fresh_install_is_on_everywhere(home):
    assert not home.exists()                       # empty MPMF_HOME, no config file at all
    assert paths.load_config()["history_mode"] is True
    assert P._history_mode() is True and H.enabled() is True
    assert set(_readers().values()) == {True}


def test_config_without_the_key_is_on(home):
    home.write_text(json.dumps({"memory_render": "full"}))
    assert set(_readers().values()) == {True}


@pytest.mark.parametrize("val", ["0", "false", "off", "no", "", "FALSE"])
def test_env_turns_it_off_in_every_reader(home, monkeypatch, val):
    monkeypatch.setenv("MPMF_HISTORY_MODE", val)
    assert set(_readers().values()) == {False}


@pytest.mark.parametrize("val", ["1", "true", "on", "yes"])
def test_env_on_beats_explicit_false_in_file(home, monkeypatch, val):
    home.write_text(json.dumps({"history_mode": False}))
    monkeypatch.setenv("MPMF_HISTORY_MODE", val)
    assert set(_readers().values()) == {True}


def test_explicit_false_in_file_beats_the_default(home):
    home.write_text(json.dumps({"history_mode": False}))
    assert set(_readers().values()) == {False}


def test_unreadable_config_falls_back_to_default(home):
    home.write_text("{not json")
    assert set(_readers().values()) == {True}


def test_all_readers_follow_one_file(home, tmp_path):
    """Provider, block and history resolve the SAME file paths.config_path() names.

    r103: the assertion is on the resolver FUNCTIONS (called now), not on import-time
    module attributes — that is the whole point of the fix: the value is read per call.
    """
    assert (str(P._recall_config_path()) == str(B.recall_cfg_path())
            == str(paths.config_path()) == str(home))


def test_nightly_revert_default_on_then_every_reader_off(home):
    """Default on (key absent) -> simulated HOLDOUT drop -> revert -> every reader says off."""
    from mpmf_plus.eval import nightly as N
    base_dir = str(home.parent)
    N.BASE = base_dir
    N.AR_LOG = os.path.join(base_dir, "autorevert_last.txt")
    home.write_text(json.dumps({"rerank_skip_after_ms": 0}))        # no history_mode key: default ON
    assert set(_readers().values()) == {True}
    base = {str(i): ("correct" if i < 30 else "declined") for i in range(1, 65)}
    for f in ("rerank_baseline.json", "history_baseline.json"):
        json.dump({"acc": 30 / 64, "n": 64, "rows": base}, open(os.path.join(base_dir, f), "w"))
    cur = [{"id": i, "split": "holdout", "verdict": "wrong" if i <= 7 else base[str(i)]}
           for i in range(1, 65)]
    assert N.autorevert(cur) == 0
    c = json.load(open(home))
    assert c["history_mode"] is False                               # explicit false, not a removed key
    assert c["rerank_skip_after_ms"] == 4500
    assert set(_readers().values()) == {False}
    paused = open(os.path.join(base_dir, "PAUSED.md")).read()
    assert "history_mode auto-revert" in paused and "history_mode: false" in paused


def test_nightly_revert_with_key_true_writes_false(home):
    from mpmf_plus.eval import nightly as N
    home.write_text(json.dumps({"history_mode": True}))
    out = N._revert_history(0.3, 0.45, 0.15, 0.01, 0, 8)
    assert out.startswith("HISTORY AUTO-REVERTED")
    assert json.load(open(home))["history_mode"] is False
    assert set(_readers().values()) == {False}


def test_nightly_revert_is_noop_when_already_off(home):
    from mpmf_plus.eval import nightly as N
    home.write_text(json.dumps({"history_mode": False}))
    assert N._revert_history(0.3, 0.45, 0.15, 0.01, 0, 8) == ""
    assert not os.path.exists(os.path.join(str(home.parent), "PAUSED.md"))


def test_fresh_process_empty_home_is_on(tmp_path):
    """A brand-new process with an empty MPMF_HOME (no config file) sees history_mode on."""
    import subprocess
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    code = ("from mpmf_plus import paths, provider, history; "
            "print(paths.load_config()['history_mode'], provider._history_mode(), history.enabled())")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("MPMF_")}
    env.update(MPMF_HOME=str(tmp_path), PYTHONPATH=src)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.split() == ["True", "True", "True"], out.stdout
    env["MPMF_HISTORY_MODE"] = "0"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         timeout=120)
    assert out.stdout.split() == ["False", "False", "False"], out.stdout
