"""Run the script-style test files (module-level code, exit code = result) under pytest.

pytest cannot import them (they sys.exit or execute at import), so each runs in a
subprocess with this checkout's src on PYTHONPATH and must exit 0.
"""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
SCRIPTS = [
    "test_autorevert_pairing.py", "test_dry_run_no_key.py", "test_reenrich_accounting.py",
    "test_real_turns_only.py", "test_e2e_store_growth.py", "test_graph_value.py",
    "test_host_contract.py", "test_r69_guards.py",
    pytest.param("test_integration.py", marks=pytest.mark.skip(
        reason="obsolete: v0.1.0 script calls MPMFMemory.remember()/create_node outside write_txn(), which the core now refuses (r69 guard); the lifecycle is covered by test_cas.py, test_core.py and test_e2e_store_growth.py")),
]


@pytest.mark.parametrize("name", SCRIPTS)
def test_script(name):
    if name == "test_host_contract.py" and not os.path.exists(os.path.expanduser("~/.hermes/plugins/mpmf_plus_user/__init__.py")):
        pytest.skip("host plugin not installed on this box; deploy.sh runs the host contract on the live host")
    env = dict(os.environ, PYTHONPATH=SRC, MPMF_LIB_PATH=SRC)
    r = subprocess.run([sys.executable, os.path.join(HERE, name)], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, (r.stdout + r.stderr)[-1500:]
