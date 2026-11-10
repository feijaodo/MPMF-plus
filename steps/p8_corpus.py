#!/usr/bin/env python3
"""P8: build the raw-message corpus and embed it (resumable).

Reads every real user message (ChatGPT export, Grok export, Hermes telegram/cli
sessions; the same loaders as p6_qgen), drops <40 chars, secret-looking text and exact
duplicates, then embeds the first 512 chars (mpmf.embed_text rule) with the local
mxbai-embed-large. Output (outside the repo, private chat content):
  <out>/raw_items.jsonl      one {"i","text","date","conv","src","idx"} per line
  <out>/vec_<k>.npy          float32 (n,1024) chunks of 1000 items
No .mpmf store is opened or written here.

usage: p8_corpus.py <outdir>      env: MPMF_SRC
"""
import hashlib, json, os, sys, time
import numpy as np
import requests

OUT = sys.argv[1]
sys.argv = [sys.argv[0], OUT]            # p6_qgen reads argv[1] as its own outdir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPMF_SRC", "/root/mpmf-release/src")
import p6_qgen as q                      # noqa: E402  (loaders + secret filter only)
sys.path.insert(0, os.environ["MPMF_SRC"])
from mpmf.embed_text import embed_text   # noqa: E402

URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL, CHUNK, BATCH = "mxbai-embed-large", 1000, 16
os.makedirs(OUT, exist_ok=True)


def corpus():
    items, seen = [], set()
    for fn in (q.chatgpt_convs, q.grok_convs, q.session_convs):
        for src, cid, title, msgs in fn():
            for k, (t, txt) in enumerate(msgs):
                if len(txt) < 40 or not q.clean_ok(txt):
                    continue
                h = hashlib.sha1(txt.encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                items.append({"text": txt[:4000], "date": q.day(t), "conv": cid, "src": src, "idx": k})
    for i, it in enumerate(items):
        it["i"] = i
    return items


def main():
    p = os.path.join(OUT, "raw_items.jsonl")
    if os.path.exists(p):
        items = [json.loads(l) for l in open(p)]
    else:
        items = corpus()
        with open(p, "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
    print("items", len(items), flush=True)
    t0 = time.time()
    for c in range(0, len(items), CHUNK):
        fp = os.path.join(OUT, "vec_%04d.npy" % (c // CHUNK))
        if os.path.exists(fp):
            continue
        vs = []
        part = items[c:c + CHUNK]
        for b in range(0, len(part), BATCH):
            texts = [embed_text(x["text"]) for x in part[b:b + BATCH]]
            for attempt in range(4):
                try:
                    r = requests.post(URL + "/api/embed", json={"model": MODEL, "input": texts}, timeout=600)
                    r.raise_for_status()
                    vs.extend(r.json()["embeddings"])
                    break
                except Exception as e:
                    if attempt == 3:
                        raise
                    time.sleep(5 * (attempt + 1))
        np.save(fp + ".tmp.npy", np.asarray(vs, dtype=np.float32))
        os.replace(fp + ".tmp.npy", fp)
        print("chunk %d done  %d/%d  %.0fs" % (c // CHUNK, c + len(part), len(items), time.time() - t0), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
