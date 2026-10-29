"""Offline end-to-end example: write facts to a temp store, recall, render a block.

No model and no network: it builds a throwaway store with the bundled core,
does a toy lexical recall over the stored facts, and renders the prompt block
with the package's own block builder. It never touches a real store.
"""
import json
import os
import tempfile

d = tempfile.mkdtemp(prefix="mpmf-example-")
os.environ["MPMF_HOME"] = d
os.environ["MPMF_STORE_PATH"] = os.path.join(d, "demo.mpmf")
with open(os.path.join(d, "recall.config.json"), "w") as f:
    json.dump({"memory_render": "full", "memory_item_chars": 400,
               "memory_block_chars": 2000}, f)

from mpmf.memory import MemoryManager          # noqa: E402
from mpmf_plus import block                    # noqa: E402

FACTS = [
    ("backup_time", "The nightly backup runs at 02:00 UTC."),
    ("gateway_port", "The API gateway listens on port 8443."),
    ("key_rotation", "Rotate the signing key every 90 days."),
]

store = MemoryManager(os.environ["MPMF_STORE_PATH"], vector_dimensions=64)
store.load()
for name, text in FACTS:
    store.create_node(name=name, node_type=0, plane_flags=1, text=text,
                      summary=text, wiki={"_ns": "memory"}, importance=0.5)
store.periodic_save()
print("nodes written:", len(store.all_node_ids()))

# reopen from disk: proves the image round-trips
store = MemoryManager(os.environ["MPMF_STORE_PATH"], vector_dimensions=64)
store.load()
ids = {store.get_name(i): i for i in store.all_node_ids()}
names = set(ids)
print("nodes after reopen:", len(names))

# toy lexical recall (the real path is mpmf_plus.memory.recall, which needs the
# live mxbai embedder; this keeps the example free and offline)
query = "when does the backup run?"
qw = set(query.lower().split())
scored = [f for f in FACTS if qw & set(f[1].lower().split())]
hits = scored[:2]

shim = block.RecallShim()
shim.top_nodes = [ids[name] for name, _ in hits]
shim.raw_texts = {ids[name]: text for name, text in hits}
shim.summaries = dict(shim.raw_texts)
body, items = block.render_memory_block(query, shim, store, top_k=2)
print("--- rendered block ---")
print(body)
assert items, "example failed: no block rendered"
