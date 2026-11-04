"""r83f: every embed call site must truncate identically.

Runtime equality for the three importable functions (core, thinking writer,
legacy wrapper); static call-site assertions for the ones that embed inside a
CLI loop or a batch (ingest, rebuild tool, scan, daily check).
"""
import importlib.util
import os
import sys

import pytest

sys.path.insert(0, "/root/mpmf-release/src")
from mpmf.embed_text import EMBED_CHARS, embed_text  # noqa: E402

TEXTS = [
    "",
    "short fact",
    "x" * (EMBED_CHARS - 1),
    "y" * EMBED_CHARS,
    "z" * (EMBED_CHARS + 1),
    ("Round 83e: long text about vectors. " * 200),
    "unicode — em dash, ünïcode, 中文, and a trailing tail that must be cut",
]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_core_rule_is_head_512():
    assert EMBED_CHARS == 512
    assert embed_text(None) == ""
    for t in TEXTS:
        assert embed_text(t) == (t or "")[:512]
        assert len(embed_text(t)) <= 512


def test_thinking_writer_agrees():
    th = _load("/root/.mpmf/athena_thinking.py", "athena_thinking_r83f")
    for t in TEXTS:
        assert th._embed_text(t) == embed_text(t), "thinking writer drifted"
    assert th.EMBED_CHARS == EMBED_CHARS


def test_legacy_wrapper_agrees():
    w = _load("/root/.mpmf/athena_memory.py", "athena_memory_r83f")
    for t in TEXTS:
        assert w._truncate_for_embed(t) == embed_text(t), "wrapper document path drifted"


@pytest.mark.parametrize("path,needles", [
    ("/root/mpmf-release/src/mpmf_plus/ingest.py",
     ["embed(embed_text(sentence))", "embed(embed_text(t))"]),
    ("/root/mpmf-release/src/mpmf_plus/rebuild/cli.py", ["_canon_et(text)"]),
    ("/root/r83e/scan_athena.py", ["_canon(t)"]),
    ("/root/.hermes/scripts/mpmf_daily_checks.py", ["_canon(m.get_text(i))"]),
])
def test_call_sites_use_the_core_rule(path, needles):
    assert os.path.exists(path), path
    src = open(path).read()
    for n in needles:
        assert n in src, "%s does not call the canonical rule (%r missing)" % (path, n)
