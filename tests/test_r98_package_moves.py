"""r98: host scripts moved INTO the package.

`mpmf_daily_checks.py` and `vet40.py` used to live on the host (~/.hermes/scripts,
~/.mpmf/gate_eval) and be imported by path, so nothing the deploy gate ran could prove they
worked. Their implementation now ships in the package; the host paths are thin shims.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
HOME = Path(os.environ.get("HOME", "~")).expanduser()
SHIM = HOME / ".hermes" / "scripts" / "mpmf_daily_checks.py"
GATE_VET = Path(os.environ.get("MPMF_HOME") or (HOME / ".mpmf")) / "gate_eval" / "vet40.py"


def _dry_home(tmp_path):
    (tmp_path / "health").mkdir()
    (tmp_path / "recall.config.json").write_text(
        json.dumps({"store_path": str(tmp_path / "dry.mpmf"), "store_mode": "attach"}))
    (tmp_path / "dry.mpmf").write_bytes(b"")
    return tmp_path


def _env(home):
    e = dict(os.environ, MPMF_HOME=str(home), MPMF_RECALL_CONFIG=str(home / "recall.config.json"),
             MPMF_HEALTH_DIR=str(home / "health"), MPMF_EVAL_DIR=str(home / "eval"),
             PYTHONPATH=str(SRC), MPMF_LIB_PATH=str(SRC), PYTHONDONTWRITEBYTECODE="1")
    return e


def test_daily_checks_run_from_the_package(tmp_path):
    home = _dry_home(tmp_path)
    r = subprocess.run([sys.executable, "-m", "mpmf_plus.daily", "--dry-run"],
                       capture_output=True, text=True, env=_env(home), timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    assert "DRY-RUN daily OK" in r.stdout
    assert "capture_skips(info)" in r.stdout


def test_host_daily_script_is_a_thin_shim(tmp_path):
    if not SHIM.exists():
        return                                    # host without the shim: nothing to prove here
    text = SHIM.read_text()
    assert "mpmf_plus.daily" in text
    code = [l for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    assert len(code) <= 15, "the host script still carries logic: %d code lines" % len(code)
    assert "run_daily" not in text and "check_disk" not in text
    home = _dry_home(tmp_path)
    r = subprocess.run([sys.executable, str(SHIM), "--dry-run"],
                       capture_output=True, text=True, env=_env(home), timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    assert "DRY-RUN daily OK" in r.stdout


def test_shim_forwards_the_alert_exit_code(monkeypatch, tmp_path):
    """A real alert must still exit 1 through the shim (cron relies on it)."""
    from mpmf_plus import daily, health
    monkeypatch.setattr(health, "run_daily", lambda: ["something is wrong"])
    monkeypatch.setattr(health, "daily_info", lambda: ["capture: captured=0"])
    monkeypatch.setattr(health, "check_disk", lambda *a, **k: None)
    assert daily.main([]) == 1
    monkeypatch.setattr(health, "run_daily", lambda: [])
    assert daily.main([]) == 0


def test_vet40_judge_ships_in_the_package():
    from mpmf_plus.eval import vet40
    assert len(vet40.SYS) > 500 and "form=principle" in vet40.SYS
    assert set(vet40.MODELS) and set(vet40.CAP) == set(vet40.MODELS)
    src = (SRC / "mpmf_plus" / "eval" / "vet40.py").read_text()
    assert str(HOME) not in src, "vet40 must not hard-code a home path"
    assert "MPMF_GATE_EVAL" in src or "gate_eval_dir" in src


def test_cognition_gets_the_shipped_judge_prompt(monkeypatch):
    from mpmf_plus import cognition
    from mpmf_plus.eval import vet40
    monkeypatch.setattr(sys, "path", [p for p in sys.path if "gate_eval" not in p])
    assert cognition._panel_prompt() == vet40.SYS


def test_gate_eval_vet40_is_only_a_shim():
    if not GATE_VET.exists():
        return
    text = GATE_VET.read_text()
    assert "mpmf_plus.eval.vet40" in text
    code = [l for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    assert len(code) <= 20, "gate_eval/vet40.py still carries logic: %d code lines" % len(code)
    assert "def judge" not in text and "OPENROUTER" not in text.replace("sys.path", "")
