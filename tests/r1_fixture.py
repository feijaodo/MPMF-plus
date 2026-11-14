"""Synthetic fixture helpers for r1 (no real data). Used by tests/test_r1_injection.py."""
import re
from types import SimpleNamespace


class FakeMem:
    def __init__(self, nodes):
        self.nodes = nodes     # nid -> dict(name,text,wiki)
        self.header = SimpleNamespace(node_count=max(nodes))

    def get_name(self, nid): return self.nodes[nid]["name"]
    def get_text(self, nid): return self.nodes[nid]["text"]
    def get_wiki(self, nid): return self.nodes[nid]["wiki"]
    def get_summary(self, nid): return self.nodes[nid]["text"][:120]


def mem():
    W = lambda ent, said, fd, kind, mode, **k: dict(entity=ent, _said_at=said, _fact_date=fd,
                                                   _life_kind=kind, _life_mode=mode, **k)
    return FakeMem({
        1: dict(name="f_a_1", text="The user adopted a cat named Miso in March.",
                wiki=W("cat", "2024-03-02", "2024-03-01", "have", "actual")),
        2: dict(name="f_a_2", text="The user's cat Miso had a vet visit before the move.",
                wiki=W("cat", "2024-06-10", "2024-06-09", "have", "actual")),
        3: dict(name="f_a_3", text="The user later said Miso now sleeps on the sofa & <purrs>.",
                wiki=W("cat", "2024-09-30", "2024-09-30", "state", "actual", _said_at_source="ingest")),
        4: dict(name="f_b_1", text="The user plans a trip to Lisbon in May.",
                wiki=W("trip", "2024-04-01", "2024-05-01", "intent", "hypothetical")),
        5: dict(name="f_b_2", text="The user booked the Lisbon flight after the trip plan changed.",
                wiki=dict(entity="trip", _said_at="2024-04-20", kind="change")),   # no fact date / life fields
        6: dict(name="f_c_1", text="Short.", wiki=dict(entity="x")),
        8: dict(name="f_e_1", text="The user ordered a \"special\" <b>blend</b> and waited, A&B.",
                wiki=dict(entity='odd "quote" <x> and y', _said_at="2024-02-02", _fact_date="2024-02-02",
                          _life_kind="have", _life_mode="actual")),
        9: dict(name="f_e_2", text="The user later changed the \"special\" order.",
                wiki=dict(entity='odd "quote" <x> and y', _said_at="2024-03-03", _fact_date="2024-03-03")),
        7: dict(name="f_d_1", text="The user likes green tea " + "very much " * 80,
                wiki=dict(entity="tea", _said_at="2024-01-05", _fact_date="2024-01-05")),
    })


class Res:
    def __init__(self, ids, scores=None):
        self.top_nodes = list(ids)
        m = mem()
        self.raw_texts = {i: m.nodes[i]["text"] for i in ids}
        self.summaries = {i: m.nodes[i]["text"][:120] for i in ids}
        if scores is not None:
            self.scores = scores


def entity_attr_values(s):
    return re.findall(r'<history entity="([^"]*)"', s)


CASES = {"temporal_cat": ("When did the user's cat Miso change, before or after the move?", [1, 2, 3]),
         "plain_cat": ("Tell me about the cat Miso", [1, 2, 3]),
         "temporal_trip": ("What happened first with the Lisbon trip, the plan or the flight?", [4, 5]),
         "temporal_odd": ("When did the special order change?", [8, 9]),
         "long_item": ("what tea does the user like", [7, 6, 1]),
         "only_short": ("anything", [6])}

LESSON = ("<prior-reasoning>\nThe following lessons were distilled from prior sessions. Apply them where relevant, "
          "but do not treat them as inviolable rules.\n1. [hypothesis] When A happens. | Because B. | Do C.\n"
          "2. [unknown] When D. | E. | F " + "x" * 600 + "\n</prior-reasoning>")
LESSONS3 = ("<prior-reasoning>\nThe following lessons were distilled from prior sessions. Apply them where relevant, "
            "but do not treat them as inviolable rules.\n1. [hypothesis] When A happens. | Because B. | Do C.\n"
            "2. When D <x> happens. | E & F. | Do G.\n3. [unknown] When H. | I. | " + "j" * 150 + "\n</prior-reasoning>")
THINK = [{"context": "c" * 200, "conclusion": "d" * 300, "outcome": "ok"},
         {"context": "second ctx", "conclusion": "", "outcome": "unknown"}]
RAW_HITS = [{"text": "I bought a green Subaru last spring", "date": "2024-04-01", "cos": .8},
            {"text": "word " * 200, "date": "", "cos": .7}, {"text": "third", "date": "2024-05-05", "cos": .6}]


def build_legacy_outputs():
    """Every string the legacy renderers make on the synthetic fixture (old API only, so the
    same function runs on the pre-r1 tree to create the committed golden)."""
    from mpmf_plus import block as B, history as H, rendering as R, raw_layer as RL
    H.reset()
    m = mem()
    out = {}
    for k, (q, ids) in CASES.items():
        body, items = B.render_memory_block(q, Res(ids), m, 8, elapsed=0.0)
        out[k] = body
        out[k + ".items"] = items
    out["cap_small"] = B.render_memory_block("Tell me about the cat Miso", Res([1, 2, 3, 7]), m, 8,
                                             max_chars=200)[0]
    out["thinking"] = R.thinking_block(THINK, 800)
    out["thinking_small"] = R.thinking_block([{"context": "c" * 100, "conclusion": "d" * 100,
                                               "outcome": "ok"}] * 4, 300)
    out["cognition"] = R.cognition_block(LESSON, 700)
    out["cognition_full"] = R.cognition_block(LESSON, 5000)
    out["raw"] = RL.render(RAW_HITS, 500, 1500)
    out["raw_small"] = RL.render(RAW_HITS, 100, 300)
    out["compose"] = R.compose(out["plain_cat"], out["thinking"], out["cognition"])
    return out
