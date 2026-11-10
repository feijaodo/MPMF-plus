#!/usr/bin/env python3
"""P8: write the raw corpus (items + vectors from p8_corpus.py) into a COPY of the store.

usage: p8_backfill.py <corpus_dir> <store_copy.mpmf>
Refuses the live store (raw_layer.backfill does too). Idempotent: nodes are named by
text hash, so re-running only adds what is missing.
"""
import glob, json, os, sys, time
import numpy as np

corpus, store = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.environ.get("MPMF_SRC", "/root/mpmf-release/src"))
sys.path.insert(0, "/root/.mpmf")
from mpmf_plus import raw_layer as R   # noqa: E402

assert os.path.realpath(store) != os.path.realpath("/root/.mpmf/athena.mpmf"), "refusing the live store"
items = [json.loads(l) for l in open(os.path.join(corpus, "raw_items.jsonl"))]
vecs = [np.load(f) for f in sorted(glob.glob(os.path.join(corpus, "vec_*.npy")))]
V = np.vstack(vecs) if vecs else np.zeros((0, 1024), np.float32)
n = V.shape[0]
print("items %d, embedded %d" % (len(items), n), flush=True)
t = time.time()
r = R.backfill(items[:n], V, store)
print(json.dumps(r), "%.1fs" % (time.time() - t), flush=True)
os._exit(0)
