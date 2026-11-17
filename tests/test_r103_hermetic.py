"""r103: the suite is hermetic — no test reads or writes the live MPMF home by accident.

conftest.py points MPMF_HOME / MPMF_RECALL_CONFIG at a throwaway dir before any test
imports the package, and installs an audit hook that refuses write-mode opens under the
real home. These tests prove both halves exist and actually work.
"""
import os
from pathlib import Path

import pytest

from mpmf_plus import paths, provider, block, history


def _real_home() -> Path:
    return (Path.home() / ".mpmf").resolve()


def test_default_home_is_a_throwaway_dir():
    """conftest creates the suite's home; it must never be the live one."""
    import conftest

    home = Path(conftest._TEST_HOME).resolve()
    assert "mpmf-pytest-" in home.name, home
    assert home != _real_home()
    assert home.is_dir()


def test_every_reader_resolves_the_same_path():
    """Order-independent: whatever MPMF_HOME is right now, all readers agree on the file
    and that file lives in that home."""
    one = {paths.config_path().resolve(), provider._recall_config_path().resolve(),
           block.recall_cfg_path().resolve()}
    assert len(one) == 1, one
    assert paths.config_path().resolve().parent == paths.home().resolve()


def test_write_guard_refuses_the_real_home(tmp_path):
    real = _real_home()
    if not real.is_dir():
        pytest.skip("no live MPMF home on this host")
    probe = real / "__r103_pytest_probe__.tmp"
    with pytest.raises(RuntimeError) as e:
        open(probe, "w").close()
    assert "hermeticity" in str(e.value)
    assert not probe.exists()


def test_reads_from_the_real_home_are_still_allowed():
    """The guard is about WRITES. A test that inspects live data read-only is fine."""
    real = _real_home()
    if not (real / "recall.config.json").is_file():
        pytest.skip("no live recall config on this host")
    with open(real / "recall.config.json"):      # must not raise
        pass


def test_a_test_that_needs_a_store_gets_a_copy(tmp_path):
    """The sanctioned pattern: copy, then point the env at the copy."""
    src = _real_home() / "recall.config.json"
    if not src.is_file():
        pytest.skip("no live recall config on this host")
    dst = tmp_path / "recall.config.json"
    dst.write_bytes(src.read_bytes())
    assert dst.read_text() == src.read_text()
