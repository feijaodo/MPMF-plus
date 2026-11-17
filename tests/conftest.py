"""Make `pytest` test THIS working tree, not an installed `mpmf` elsewhere.

Without this, `pytest` picked up whatever `mpmf` was on sys.path (e.g. an
editable install of another checkout) and the suite silently tested the wrong
code. Round 77 hit exactly that. Subprocess-based tests inherit PYTHONPATH.

r103 — hermeticity:
  * MPMF_HOME / MPMF_RECALL_CONFIG (and the health/eval dirs) point at a throwaway
    directory created HERE, before any test module imports the package. Default
    store paths therefore resolve inside that directory: no test reads or writes
    the live store just because it forgot to pass a path.
  * an audit hook refuses any write-mode open() under the REAL ~/.mpmf, so a test
    that "legitimately" needs live data fails loudly instead of mutating it.
    A test that needs real data copies the file into tmp_path first.
  * the mtime of the live store is NOT used as a check: the Hermes gateway writes
    to it concurrently on this host, which would make the gate flaky.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_pp = os.environ.get("PYTHONPATH", "")
if _SRC not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = _SRC + (os.pathsep + _pp if _pp else "")

# --- hermetic MPMF_HOME (r103) ---------------------------------------------------------------
REAL_MPMF_HOME = (Path.home() / ".mpmf").resolve()
_TEST_HOME = Path(tempfile.mkdtemp(prefix="mpmf-pytest-"))
(_TEST_HOME / "health").mkdir(parents=True, exist_ok=True)
(_TEST_HOME / "eval").mkdir(parents=True, exist_ok=True)
_TEST_CFG = _TEST_HOME / "recall.config.json"
if not _TEST_CFG.exists():
    _TEST_CFG.write_text(json.dumps({"memory_render": "full"}))

# Only set what the caller has not already set: `deploy.sh` and the cron gate pass
# their own throwaway home, and a test that sets one itself must win.
os.environ.setdefault("MPMF_HOME", str(_TEST_HOME))
os.environ.setdefault("MPMF_RECALL_CONFIG", str(_TEST_CFG))
os.environ.setdefault("MPMF_HEALTH_DIR", str(_TEST_HOME / "health"))
os.environ.setdefault("MPMF_EVAL_DIR", str(_TEST_HOME / "eval"))

_WRITE_MODES = set("wax+")


def _install_host_stubs() -> None:
    """Keep HOST glue out of the test process (r103).

    ``llm_call_guard.py`` in the DEPLOYED MPMF home is glue, not package code: it
    hard-codes the live health directory, so importing it makes every test that trips
    an LLM path write the LIVE health dir. The package imports it defensively
    (a missing guard degrades to a no-op), so the suite registers a neutral no-op
    instead and stays hermetic. The real guard's own contract is exercised in
    ``tests/test_host_contract.py``.
    """
    import contextlib as _ctx
    import types

    mod = types.ModuleType("llm_call_guard")

    @_ctx.contextmanager
    def guarded(component, *a, **k):
        yield

    mod.guarded = guarded
    mod.record = lambda *a, **k: None
    mod.__doc__ = "test stub installed by tests/conftest.py (r103 hermeticity)"
    sys.modules["llm_call_guard"] = mod


_install_host_stubs()


def _install_write_guard(real_home: Path) -> None:
    """Refuse write-mode opens under the real MPMF_HOME for the whole test session."""

    def hook(event, args):
        if event != "open":
            return
        path, mode = args[0], args[1]
        if not path or not mode or not (_WRITE_MODES & set(str(mode))):
            return
        try:
            p = Path(os.path.abspath(str(path))).resolve()
        except Exception:
            return
        if p == real_home or real_home in p.parents:
            raise RuntimeError(
                "r103 hermeticity: a test tried to WRITE %s inside the real MPMF_HOME (%s). "
                "Copy the file into tmp_path and point MPMF_HOME / MPMF_STORE_PATH / "
                "MPMF_RECALL_CONFIG at the copy." % (p, real_home))

    sys.addaudithook(hook)


if REAL_MPMF_HOME.is_dir() and str(REAL_MPMF_HOME) != str(Path(os.environ["MPMF_HOME"]).resolve()):
    _install_write_guard(REAL_MPMF_HOME)


# Script-style test files (module-level code, sys.exit at the end) cannot be
# imported by pytest; tests/test_scripts.py runs each of them in a subprocess.
collect_ignore = [
    "test_autorevert_pairing.py", "test_dry_run_no_key.py", "test_reenrich_accounting.py",
    "test_real_turns_only.py", "test_e2e_store_growth.py", "test_graph_value.py",
    "test_host_contract.py", "test_integration.py", "test_r69_guards.py",
]
