"""Single source of truth for CODE_REV.

CODE_REV = "<version> <git describe>", e.g. "0.3.0 v0.2.0-50-gad1304b".

  * version  -- ``[project] version`` of the pyproject.toml that sits above ``src/``
                (falls back to installed package metadata, then "unknown").
  * describe -- ``git describe --tags --match "v[0-9]*" --always`` of that same checkout
                (falls back to "nogit" when there is no checkout / no git).

Deliberately stdlib-only and import-free of the rest of the package, so
``mpmf_plus.provider``, ``mpmf_plus.memory`` and out-of-tree tools (e.g.
/root/.mpmf/verify_restart.py, which loads THIS FILE by path) all read the same
value.  Nothing here is hand-bumped.
"""
import re
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _version() -> str:
    try:
        m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', (_ROOT / "pyproject.toml").read_text())
        if m:
            return m.group(1)
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version("mpmf-plus")
    except Exception:
        return "unknown"


def _describe() -> str:
    try:
        if not (_ROOT / ".git").exists():
            return "nogit"
        out = subprocess.run(
            ["git", "-C", str(_ROOT), "describe", "--tags", "--match", "v[0-9]*", "--always"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return out or "nogit"
    except Exception:
        return "nogit"


def compute_code_rev() -> str:
    return "%s %s" % (_version(), _describe())


CODE_REV = compute_code_rev()
