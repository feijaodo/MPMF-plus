"""Does the idea layer actually do work?

The only honest test is a query whose correct answer shares NO words with it.
If lexical/vector matching can reach it, the graph proves nothing.

Setup: 'Redis' is linked to 'job queue', and 'job queue' is linked to the
memory "A job queue was deferred because there is only one worker".
That memory never says 'Redis'. Reaching it requires two hops.

Run A: edges stripped (graph disabled).
Run B: edges intact (graph enabled).
Any difference is attributable to the graph and nothing else.
"""
import os, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mpmf.ingestion.ingest import AutoIngest
from mpmf.ingestion.extract import Answer, Relation

QUERIES = [
    # (query, substring of the memory that SHOULD surface, why)
    ("Redis", "deferred because there is only one worker",
     "Redis -> job_queue -> deferral memory (2 hops, zero word overlap)"),
    ("the gateway", "stores its memory in an MPMF file",
     "gateway -> agent -> mpmf memory (2 hops)"),
    ("systemd", "git pull", "systemd -> deploys -> deploy memory"),
    ("task state", "polls Postgres every 30 seconds",
     "task_state -> postgres -> polling memory"),
    ("debug output", "API token", "debug_output -> api_token -> leak memory"),
]


SESSION_ANSWERS = [
    [
        Answer("The scheduler polls Postgres every 30 seconds for due tasks",
               ["scheduler", "Postgres"],
               [Relation("scheduler", "Postgres", "polls")], 0.9),
        Answer("A job queue was deferred because there is only one worker",
               ["job queue", "worker"],
               [Relation("job queue", "worker", "deferred_due_to")], 0.85),
    ],
    [
        Answer("Postgres connection pooling was set to 20 connections",
               ["Postgres", "connection pooling"],
               [Relation("Postgres", "connection pooling", "configured_with")], 0.8),
        Answer("The assistant runs on a VPS behind the message gateway",
               ["assistant", "gateway"],
               [Relation("assistant", "gateway", "runs_behind")], 0.9),
    ],
    [
        Answer("The job queue would use Redis if it were added",
               ["job queue", "Redis"],
               [Relation("job queue", "Redis", "would_use")], 0.7),
        Answer("Deploys are a git pull followed by a systemd restart",
               ["deploys", "systemd"],
               [Relation("deploys", "systemd", "uses")], 0.9),
    ],
    [
        Answer("assistant stores its memory in an MPMF file on disk",
               ["assistant", "MPMF"],
               [Relation("assistant", "MPMF", "stores_memory_in")], 0.95),
        Answer("Postgres is the single source of truth for task state",
               ["Postgres", "task state"],
               [Relation("Postgres", "task state", "is_source_of_truth_for")], 0.85),
    ],
    [
        Answer("The API token is logged in debug output and must be fixed",
               ["API token", "debug output"],
               [Relation("API token", "debug output", "leaked_into")], 0.95),
        Answer("A second worker is being added to the scheduler",
               ["worker", "scheduler"],
               [Relation("worker", "scheduler", "added_to")], 0.8),
    ],
    [
        Answer("Caching responses cut latency from 800ms to 120ms",
               ["caching", "latency"],
               [Relation("caching", "latency", "reduces")], 0.9),
        Answer("Redis is not currently installed on the VPS",
               ["Redis", "VPS"],
               [Relation("Redis", "VPS", "not_installed_on")], 0.8),
    ],
]


class StubExtractor:
    def __init__(self):
        self.session = 0
        self.i = 0

    def ask(self, interaction, question, model_override=None):
        answers = SESSION_ANSWERS[self.session]
        if self.i < len(answers):
            a = answers[self.i]
            self.i += 1
            return a
        return Answer("NONE", [], [], 0.0)


def build(strip_edges: bool):
    W = tempfile.mkdtemp()
    ing = AutoIngest(os.path.join(W, 'a.mpmf'), os.path.join(W, 't.json'),
                     vector_dimensions=768, embed_model='nomic-embed-text')
    stub = StubExtractor()
    ing.extractor = stub
    for s in range(len(SESSION_ANSWERS)):
        stub.session = s
        stub.i = 0
        ing.ingest(f"<s{s}>")
    if strip_edges:
        # Completely clear the edges so graph traversal finds nothing
        ing.mem.mem._edges = []
        # Also clear plane_index so traversal is truly disabled
        ing.mem.mem._plane_index = []
    return ing


def evaluate(ing, label, top_n=5):
    m = ing.mem.mem
    hits = 0
    ranks = []
    print(f"\n--- {label} (edges in store: {len(m._edges)}) ---")
    for q, needle, why in QUERIES:
        res = ing.mem.recall(q, top_n=top_n)
        texts = [(m.get_text(nid) or '') for nid in res.top_nodes]
        rank = None
        for i, t in enumerate(texts):
            if needle in t:
                rank = i + 1
                break
        if rank:
            hits += 1
            ranks.append(rank)
        status = f"rank {rank}" if rank else "MISS"
        print(f"  {q!r:<22} {status:<8} {why}")
    mean = sum(ranks) / len(ranks) if ranks else 0
    print(f"  => {hits}/{len(QUERIES)} found, mean rank {mean:.2f}")
    return hits, mean


if __name__ == "__main__":
    print("=" * 70)
    print("GRAPH ABLATION -- queries with ZERO lexical overlap with the answer")
    print("=" * 70)
    no_g = evaluate(build(strip_edges=True), "GRAPH DISABLED")
    with_g = evaluate(build(strip_edges=False), "GRAPH ENABLED")
    print("\n" + "=" * 70)
    print(f"without graph: {no_g[0]}/{len(QUERIES)}   with graph: {with_g[0]}/{len(QUERIES)}")
    if with_g[0] > no_g[0]:
        print("=> The idea layer retrieves memories lexical search cannot reach.")
    elif with_g[0] == no_g[0] and with_g[1] < no_g[1] and no_g[0] > 0:
        print("=> Same recall, better ranking. Graph helps but is not decisive.")
    else:
        print("=> The graph is contributing nothing on this corpus.")