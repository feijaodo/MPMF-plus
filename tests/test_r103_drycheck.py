"""r103/r105: a source-changing cron must FAIL in dry-run when no embedder is wired.

`mpmf_plus.drycheck` is what the cron dry-run calls. These tests pin both directions:
armed -> exit 0; a writer with no embed-source->vector hook -> exit 1 with a reason a
human can act on. They use scratch stores only.
"""
from mpmf.query import MPMFMemory
from mpmf_plus import drycheck


def test_guard_is_armed_on_this_tree():
    assert drycheck.guard_errors() == []
    assert drycheck.main([]) == 0


def test_drycheck_fails_when_the_writer_has_no_hook(monkeypatch):
    """Simulates the mistake: MPMFMemory opened without installing the embed hook."""
    orig = MPMFMemory.__init__

    def no_hook(self, *a, **k):
        orig(self, *a, **k)
        self.mem.embedder = None

    monkeypatch.setattr(MPMFMemory, "__init__", no_hook)
    errs = drycheck.guard_errors()
    assert errs, "a writer with no embedder must be reported in dry-run, not at 04:00"
    assert any("source->vector hook" in e for e in errs), errs
    assert drycheck.main([]) == 1


def test_drycheck_fails_when_the_guard_itself_is_gone(monkeypatch):
    """If set_node_text stops recording the drift, the dry-run must say so."""
    from mpmf.memory import MemoryManager

    def loose_set_text(self, node_id, text):
        self._require_write_txn("set_node_text")
        c = self._content.get(node_id)
        if c is not None:
            c.text = text                      # no dirty flag, no embed

    monkeypatch.setattr(MemoryManager, "set_node_text", loose_set_text)
    errs = drycheck.guard_errors()
    assert errs
    assert any("ACCEPTED" in e for e in errs), errs
