"""r65: every memory carries three dates — said, ingested, fact.

The failure this prevents: a 2024 conversation imported today looked newer than
a 2026 fact, so "what car do I drive now" answered with the old one.
"""
import pytest

from mpmf_plus import block, memory
from mpmf_plus.dates import date_fields, extract_fact_date, ingest_now, normalise_ts

DIM = memory.VECTOR_DIM


@pytest.fixture(autouse=True)
def _stub(monkeypatch, tmp_path):
    monkeypatch.setenv("MPMF_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setattr(memory, "_embed_floats_local",
                        lambda t, dim=DIM, cap=None: [0.01] * DIM)


# ── normalise_ts ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,expected", [
    (1775164844179, "2026-04-02T21:20:44.179000Z"),          # epoch ms
    (1775164844.179, "2026-04-02T21:20:44.179000Z"),         # epoch s
    ("1775164844179", "2026-04-02T21:20:44.179000Z"),        # numeric string
    ("2026-04-02T21:20:44.139292Z", "2026-04-02T21:20:44.139292Z"),  # ISO passthrough
    ({"$date": {"$numberLong": "1775164844179"}}, "2026-04-02T21:20:44.179000Z"),
])
def test_normalise_accepts_every_source_format(raw, expected):
    assert normalise_ts(raw) == expected


@pytest.mark.parametrize("bad", [None, "", "not a date", 0, -5])
def test_normalise_returns_none_not_a_guess(bad):
    assert normalise_ts(bad) is None


# ── extract_fact_date ──────────────────────────────────────────────────────
@pytest.mark.parametrize("text,expected", [
    ("the lease started on 2026-02-28", "2026-02-28"),
    ("bought it on March 4, 2026 at the dealer", "2026-03-04"),
    ("ordered 22 December 2025", "2025-12-22"),
    ("purchased in April 2024 in Springfield", "2024-04-01"),
    ("the user owns a Volvo", None),
    ("in 2024 he moved", None),          # bare year is not a date
])
def test_extract_fact_date(text, expected):
    assert extract_fact_date(text) == expected


# ── date_fields ────────────────────────────────────────────────────────────
def test_explicit_said_at_wins_and_is_labelled():
    f = date_fields("the car arrived on 2026-02-28", said_at="2025-12-22T10:00:00Z")
    assert f["_said_at"] == "2025-12-22T10:00:00Z"
    assert f["_said_at_source"] == "explicit"
    assert f["_fact_date"] == "2026-02-28"
    assert f["_ingested_at"].endswith("Z")


def test_missing_said_at_falls_back_to_ingest_but_says_so():
    f = date_fields("the user owns a Volvo", said_at=None)
    assert f["_said_at"] == f["_ingested_at"]
    assert f["_said_at_source"] == "ingest"
    assert f["_fact_date"] == ""


def test_conv_id_recorded_when_given():
    f = date_fields("a fact", said_at=1775164844179, conv_id="abc-123")
    assert f["_conv_id"] == "abc-123"
    assert "_conv_id" not in date_fields("a fact")


# ── store() writes them ────────────────────────────────────────────────────
def _store(tmp_path, name, text, **kw):
    p = str(tmp_path / "d.mpmf")
    from mpmf_plus import store as st
    if not (tmp_path / "d.mpmf").exists():
        st.create_empty(st.StoreSettings(path=p, mode="create", origins={}), embed_model=memory.EMBED_MODEL)
    memory.configure_store(path=p, mode="attach")
    memory.store(text, name=name, auto_extract=False, auto_link=False,
                 namespace="memory", **kw)
    return p


def test_store_writes_all_three_dates(tmp_path):
    p = _store(tmp_path, "d1", "the lease was signed on 2026-02-28",
               said_at="2026-02-28T12:00:00Z")
    w = memory.get("d1").get("wiki") if isinstance(memory.get("d1"), dict) else None
    node = memory._open().mem
    nid = node.get_node_id_by_name("d1")
    w = node.get_wiki(nid)
    assert w["_said_at"] == "2026-02-28T12:00:00Z"
    assert w["_said_at_source"] == "explicit"
    assert w["_fact_date"] == "2026-02-28"
    assert w["_ingested_at"]


def test_store_default_said_at_is_ingest_and_marked(tmp_path):
    _store(tmp_path, "d2", "the user likes tea")
    node = memory._open().mem
    w = node.get_wiki(node.get_node_id_by_name("d2"))
    assert w["_said_at_source"] == "ingest"
    assert w["_said_at"] == w["_ingested_at"]
    assert w["_fact_date"] == ""


def test_store_records_conv_id(tmp_path):
    p = _store(tmp_path, "d3", "a fact about the harbour", conv_id="conv-9")
    node = memory._open().mem
    assert node.get_wiki(node.get_node_id_by_name("d3"))["_conv_id"] == "conv-9"


# ── block renders the date ────────────────────────────────────────────────
class _Res:
    def __init__(self, nid, text):
        self.top_nodes = [nid]
        self.raw_texts = {nid: text}
        self.summaries = {nid: text}


def test_block_shows_said_date(tmp_path):
    p = _store(tmp_path, "d4", "the lease was signed on 2026-02-28",
               said_at="2026-02-28T12:00:00Z")
    node = memory._open().mem
    nid = node.get_node_id_by_name("d4")
    body, items = block.render_memory_block("car", _Res(nid, "the lease was signed"), node, 5)
    assert "said 2026-02-28" in body
    assert "not when it was said" not in body


def test_block_flags_ingest_date_as_not_said(tmp_path):
    p = _store(tmp_path, "d5", "the user likes tea")
    node = memory._open().mem
    nid = node.get_node_id_by_name("d5")
    body, _ = block.render_memory_block("tea", _Res(nid, "the user likes tea"), node, 5)
    assert "said " in body
    assert "ingest date, not when it was said" in body
