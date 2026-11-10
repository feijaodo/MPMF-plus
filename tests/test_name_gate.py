"""f2 name gate: the owner's name / D-as-a-person phrasing must not appear in tracked files,
in commit messages of origin/main..HEAD, and no absolute home path may appear in src/ or tests/.

The patterns live in an UNTRACKED file (one regex per line, case-insensitive; a line may use
``(?-i:...)`` for a case-sensitive part): ``$MPMF_PRIVATE_TERMS``, else
``$MPMF_HOME/private_terms.txt``, else ``~/.mpmf/private_terms.txt``. The gate FAILS when the
file is missing -- it must never pass silently. Hits are reported as file:line + pattern number
only, never the matched text.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Commits in origin/main..HEAD whose MESSAGE predates the gate and may not be rewritten by an
# unattended run (history rewrite is a manual step, see steps/f2.notes.md). Delete the entry once
# the message has been reworded; after that the gate is strict for every commit.
# MPMF_GATE_STRICT_MSGS=1 ignores the list (use it to confirm a rewritten history).
KNOWN_LEGACY_MESSAGES = {"f4eefee"}

# built from pieces so this file does not contain the literal itself
ABS_HOME = "/" + "ro" + "ot"


def _terms_path() -> Path:
    """First existing of: $MPMF_PRIVATE_TERMS, $MPMF_HOME/private_terms.txt, ~/.mpmf/...

    Other tests point MPMF_HOME at a throwaway dir inside the same pytest process, so an
    MPMF_HOME without the file must not hide the real one."""
    env = os.environ.get("MPMF_PRIVATE_TERMS")
    if env:
        return Path(env)
    cands = []
    if os.environ.get("MPMF_HOME"):
        cands.append(Path(os.environ["MPMF_HOME"]) / "private_terms.txt")
    cands.append(Path.home() / ".mpmf" / "private_terms.txt")
    for c in cands:
        if c.is_file():
            return c
    return cands[0]


def _load_patterns():
    p = _terms_path()
    if not p.is_file():
        pytest.fail(
            "name gate: private terms file missing: %s\n"
            "create it (untracked, one regex per line, case-insensitive): the owner's first name "
            "with word boundaries plus the D-as-a-person phrases; see steps/f2.notes.md" % p)
    pats = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    if not pats:
        pytest.fail("name gate: %s has no patterns" % p)
    return [re.compile(x, re.I) for x in pats]


def _git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True, errors="replace")


def _tracked():
    return [f for f in _git("ls-files").split("\n") if f]


def _lines(rel):
    try:
        data = (REPO / rel).read_bytes()
    except OSError:
        return []
    if b"\0" in data[:4096]:
        return []
    return data.decode("utf-8", "replace").splitlines()


def test_no_private_terms_in_tracked_files():
    rx = _load_patterns()
    hits = []
    for rel in _tracked():
        for i, ln in enumerate(_lines(rel), 1):
            for n, r in enumerate(rx):
                if r.search(ln):
                    hits.append("%s:%d:p%d" % (rel, i, n))
    assert not hits, "private-term hits (file:line:pattern):\n" + "\n".join(hits)


def test_no_private_terms_in_commit_messages():
    rx = _load_patterns()
    try:
        revs = _git("log", "--format=%H", "origin/main..HEAD").split()
    except subprocess.CalledProcessError:
        pytest.fail("name gate: cannot resolve origin/main..HEAD (no origin/main ref?)")
    hits = []
    for h in revs:
        if not os.environ.get("MPMF_GATE_STRICT_MSGS") and any(h.startswith(k) for k in KNOWN_LEGACY_MESSAGES):
            continue
        msg = _git("log", "-1", "--format=%B", h)
        for ln_no, ln in enumerate(msg.splitlines(), 1):
            for n, r in enumerate(rx):
                if r.search(ln):
                    hits.append("%s:msgline%d:p%d" % (h[:7], ln_no, n))
    assert not hits, "private-term hits in commit messages:\n" + "\n".join(hits)


def test_no_absolute_home_path_in_src_and_tests():
    hits = []
    for rel in _tracked():
        if not (rel.startswith("src/") or rel.startswith("tests/")):
            continue
        for i, ln in enumerate(_lines(rel), 1):
            if ABS_HOME in ln:
                hits.append("%s:%d" % (rel, i))
    assert not hits, "absolute home path in src/ or tests/:\n" + "\n".join(hits)


def test_missing_terms_file_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("MPMF_PRIVATE_TERMS", str(tmp_path / "nope.txt"))
    with pytest.raises(pytest.fail.Exception) as e:
        _load_patterns()
    assert "private terms file missing" in str(e.value)
