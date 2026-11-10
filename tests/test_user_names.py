"""f2: the user's first name comes from local config (recall.config key "user_names"), never code.

Tests use a fake name ("Zed"). Set -> history stop-words include it and the extraction-prompt
example uses it; unset ([] / missing / malformed) -> neither, and the prompt says "the user".
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mpmf_plus import paths                      # noqa: E402
from mpmf_plus import provider as P              # noqa: E402
from mpmf_plus import memory as M                # noqa: E402


def _cfg(tmp_path, monkeypatch, data):
    p = tmp_path / "recall.config.json"
    if data is not None:
        p.write_text(json.dumps(data))
    monkeypatch.setattr(P, "_RECALL_CONFIG_PATH", p)
    return p


def test_default_is_empty_list():
    assert paths.DEFAULTS["user_names"] == []
    assert paths.user_names({}) == []


@pytest.mark.parametrize("bad", [None, 5, {"a": 1}, [1, None, ""], "  "])
def test_malformed_values_are_ignored(bad):
    assert paths.user_names({"user_names": bad}) == []


def test_history_stop_set(tmp_path, monkeypatch):
    _cfg(tmp_path, monkeypatch, {"history_mode": True, "user_names": ["Zed"]})
    assert "zed" in P._history_stop()
    assert "what" in P._history_stop()          # fixed list kept


def test_history_stop_unset(tmp_path, monkeypatch):
    for data in ({"history_mode": True, "user_names": []}, {"history_mode": True}, None):
        _cfg(tmp_path, monkeypatch, data)
        assert "zed" not in P._history_stop()


class _MM:
    """Two entities: 'zed' (the user) and 'jobs'; both with 2+ dated facts."""
    facts = {1: ("zed", "2025-01-01", "Zed started at Acme"), 2: ("zed", "2025-06-01", "Zed left Acme"),
             3: ("jobs", "2025-02-01", "First job was at Acme"), 4: ("jobs", "2025-07-01", "Second job at Beta")}

    def all_node_ids(self):
        return list(self.facts)

    def get_wiki(self, nid):
        e, d, _ = self.facts[nid]
        return {"entity": e, "_fact_date": d}

    def get_text(self, nid):
        return self.facts[nid][2]


class _Store:
    mem = _MM()


def test_history_block_entity_choice_follows_config(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "_open", lambda: _Store())
    q = "when did Zed change jobs"
    _cfg(tmp_path, monkeypatch, {"history_mode": True, "user_names": ["Zed"]})
    assert P.history_block(q).startswith("## History of jobs")      # name is a stop-word
    _cfg(tmp_path, monkeypatch, {"history_mode": True, "user_names": []})
    assert P.history_block(q).startswith("## History of zed")       # old behaviour without it


@pytest.mark.parametrize("prompt", [P.EXTRACTION_SYSTEM, M.EXTRACTION_SYSTEM])
def test_prompt_example_uses_name_when_set(prompt):
    out = paths.personalize_extraction_prompt(prompt, {"user_names": ["Zed", "Zedd"]})
    assert "Zed is direct" in out and '"name": "Zed"' in out
    assert "The user is direct" not in out


@pytest.mark.parametrize("prompt", [P.EXTRACTION_SYSTEM, M.EXTRACTION_SYSTEM])
def test_prompt_example_says_the_user_when_unset(prompt):
    for cfg in ({"user_names": []}, {}):
        out = paths.personalize_extraction_prompt(prompt, cfg)
        assert out == prompt
        assert "The user is direct" in out and '"name": "user"' in out
