"""The package must not depend on any absolute filesystem path.

Two guarantees:
  1. MPMF_CORE_PATH defaults to the core bundled inside this package, so an
     installed wheel works with no /root/... tree present.
  2. MPMF_CORE_PATH can still be overridden through the environment.
"""
import os
import subprocess
import sys

import mpmf
import mpmf_plus.memory as memory


def test_default_core_path_is_the_bundled_core():
    assert memory.MPMF_CORE_PATH == os.path.dirname(os.path.abspath(mpmf.__file__))
    assert os.path.basename(memory.MPMF_CORE_PATH) == "mpmf"
    assert os.path.isdir(memory.MPMF_CORE_PATH)


def test_default_core_path_is_importable_without_env(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "MPMF_CORE_PATH"}
    env["MPMF_HOME"] = str(tmp_path)
    out = subprocess.run(
        [sys.executable, "-c",
         "import mpmf_plus.memory as m; print(m.MPMF_CORE_PATH)"],
        capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == os.path.dirname(os.path.abspath(mpmf.__file__))


def test_core_path_env_override(tmp_path):
    env = dict(os.environ)
    env["MPMF_HOME"] = str(tmp_path)
    env["MPMF_CORE_PATH"] = str(tmp_path / "elsewhere")
    out = subprocess.run(
        [sys.executable, "-c",
         "import mpmf_plus.memory as m; print(m.MPMF_CORE_PATH)"],
        capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == str(tmp_path / "elsewhere")
