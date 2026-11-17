"""r102: history_grounded must not match an entity on a stopword ("and")."""
from mpmf_plus import history as H


class _MM:
    def get_wiki(self, nid):
        return {"entity": "ralph.md and ralph_loop.py"}


def test_stopword_does_not_ground_entity():
    q = "Read-only, no changes. Paste raw, unedited: the same query rendered with render_format=tagged and history_additive on."
    assert H.named_entities(q, ["ralph.md and ralph_loop.py"], [6139], _MM()) == []


def test_named_entity_still_grounds():
    q = "What did the user decide about ralph.md?"
    assert H.named_entities(q, ["ralph.md and ralph_loop.py"], [6139], _MM()) == ["ralph.md and ralph_loop.py"]


def test_alias_still_grounds():
    class MM:
        def get_wiki(self, nid):
            return {"entity": "ralph.md and ralph_loop.py", "aliases": ["ralph loop"]}

    q = "Did the ralph loop ever finish?"
    assert H.named_entities(q, ["ralph.md and ralph_loop.py"], [1], MM()) == ["ralph.md and ralph_loop.py"]
