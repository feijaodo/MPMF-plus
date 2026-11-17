"""b0 ingest + recall plumbing: the REAL per-turn pipeline into a fresh bench store, and the three context makers.

* MPMF arms ("off" / "on") share ONE ingest: history_mode is a recall-time switch (env MPMF_HISTORY_MODE=0|1,
  resolved by mpmf_plus.paths.history_mode), the store bytes are identical for both arms.
* The ingest call mirrors hermes_adapter.sync_turn -> provider._ingest_turn: digest text
  ``[session=<id> t=<epoch>]\\nUSER: ...\\nASSISTANT: ...``, context = the last two earlier digests of the SAME session,
  msg_index counting up inside a session, extraction through the packaged ``mpmf_plus.ingest.ingest`` with the
  frozen 67e prompt and the extraction model of production (deepseek/deepseek-v4.1-flash, temperature 0, reasoning off).
* Module-level caches in provider/memory/history are per store: ``reset_caches`` clears them when the harness
  switches stores inside one process.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from .. import paths as _paths
from . import llm as L
from . import textmetrics as T
from .datasets import Corpus, units

CTX_CHARS = 1500          # production: _INGEST_CTX keeps text[:1500]
BM25_K = 8                # = recall top_k default (paths.DEFAULTS["top_k"])
BM25_BUDGET = 3500        # = recall max_chars default
BM25_ITEM = 1200          # = item cap of render mode "full"

RECALL_CONFIG = {"inject_thinking": False, "inject_cognition": False, "prewarm_store": False,
                 "prewarm_warm_query": False, "prefetch_budget_ms": 600000}


def bench_env(home: str) -> Dict[str, str]:
    """Environment of a bench process: a throwaway MPMF_HOME so no live config or store is ever read."""
    return {"MPMF_HOME": home, "MPMF_RECALL_CONFIG": os.path.join(home, "recall.config.json"),
            "MPMF_HEALTH_DIR": os.path.join(home, "health"),
            "MPMF_PROVIDER_CONFIG": os.path.join(home, "provider.config.json"),
            "MPMF_BENCH_CHILD": "1"}


def write_recall_config(home: str, rerank: bool = False) -> Dict[str, Any]:
    cfg = dict(RECALL_CONFIG)
    cfg["rerank_skip_after_ms"] = 600000 if rerank else -1      # -1 = never rerank (deterministic, faster)
    _paths.dump_config(cfg, os.path.join(home, "recall.config.json"))
    return cfg


def fake_embed(text: str, dim: int = 1024, cap: Optional[int] = None) -> List[float]:
    """Hash bag-of-words embedder for --fake-model / tests (never used for a real number)."""
    v = [0.0] * dim
    for w in re.findall(r"[a-z0-9]+", (text or "").lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def fake_extract(body: str) -> str:
    """Stand-in extractor for --fake-model: one fact per last message."""
    last = body.strip().splitlines()[-1]
    m = re.match(r"\[(\d+)\]", last)
    idx = int(m.group(1)) if m else 0
    txt = last.split(": ", 1)[1] if ": " in last else last
    ent = (re.findall(r"[A-Za-z]{5,}", txt) or ["thing"])[0].lower()
    return json.dumps({"facts": [{"msg_index": idx, "fact_sentence": "User said " + txt[:200], "entity": ent,
                                  "evidence_sentence": txt[:200], "fact_date": None}]})


# ------------------------------------------------------------------------------------------------ caches
def reset_caches() -> None:
    """Forget everything provider/memory/history cached about the previous store."""
    from .. import provider as P, memory as M, history as H
    with P._OPEN_CACHE_LOCK:
        P._OPEN_CACHE.clear()
    for d in (getattr(P, "_D20_MATRIX", None), getattr(M, "_D20_MATRIX", None)):
        if isinstance(d, dict):
            d.update({"key": None, "count": -1, "ids": None, "M": None, "norms": None, "keep": None})
    if isinstance(getattr(P, "_SIT_CACHE", None), dict):
        P._SIT_CACHE.update({"key": None, "ids": [], "mat": None, "meta": {}})
    for name, val in (("_COG_VEC_CACHE", {"count": None, "vecs": {}}), ("_COG_STATE_CACHE", {"key": None, "val": None}),
                      ("_VET_CACHE", {"key": None, "val": None}), ("_VET_MODE_CACHE", {"key": None, "val": None}),
                      ("_VET_MASK_CACHE", {"key": None, "mask": None, "stats": None}),
                      ("_SIT_POOL_CACHE", {"count": None, "ids": set()})):
        d = getattr(P, name, None)
        if isinstance(d, dict):
            d.clear()
            d.update(val)
    H.reset()


def use_store(path: str, create: bool) -> None:
    from .. import provider as P, memory as M
    M.configure_store(path=path, mode="create" if create else "attach")
    P.configure(PATH=path)
    reset_caches()


# ------------------------------------------------------------------------------------------------ ingest
def ingest_calls(corpus: Corpus) -> List[Dict[str, Any]]:
    """Every ingest() call of this corpus, in order. Independent of any extraction result, so the
    extractor can be run ahead of the writes (see prefetch_extractions)."""
    out: List[Dict[str, Any]] = []
    ctx: List[Dict[str, str]] = []
    cur = None
    for u in units(corpus):
        sid = "s%03d_%s" % (u["session"], corpus.cid)
        if sid != cur:
            cur, ctx = sid, []
        said = u["date"]
        t = int(datetime.strptime(said, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
        digest = "[session=%s t=%d]\nUSER: %s" % (sid, t, u["user"][:600])
        if u["assistant"]:
            digest += "\nASSISTANT: %s" % u["assistant"][:600]
        out.append({"turn_text": digest, "said_at": said, "conv_id": sid, "msg_index": u["idx"],
                    "context": list(ctx[-2:])})
        ctx.append({"role": "user", "text": digest[:CTX_CHARS]})
    return out


def extraction_body(call: Dict[str, Any]) -> str:
    """The exact body ingest() will send (mirrors ingest.ingest: PROMPT + window of context + this turn)."""
    from ..rebuild import extraction as _x
    messages = list(call["context"]) + [{"role": "user", "text": call["turn_text"]}]
    conv = {"id": call["conv_id"], "start": call["said_at"], "messages": messages}
    return _x.PROMPT + "\n\nMESSAGES:\n" + _x._window(conv)


def _strip_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


class Extractor:
    """call_model for ingest(): disk-cached by md5(body), metered, with a thread-pool prefetch."""

    def __init__(self, chat: Callable[..., Dict[str, Any]], model: str, meter: L.Meter, cache_path: Optional[str],
                 fake: bool = False):
        self.chat, self.model, self.meter, self.fake = chat, model, meter, fake
        self.cache: Dict[str, str] = {}
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self.misses = 0
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                        self.cache[r["k"]] = r["v"]
                    except Exception:  # noqa: BLE001  (truncated last line after a kill)
                        pass

    @staticmethod
    def key(body: str) -> str:
        return hashlib.md5(body.encode()).hexdigest()

    def __call__(self, body: str) -> str:
        k = self.key(body)
        with self._lock:
            if k in self.cache:
                return self.cache[k]
        if self.fake:
            v = fake_extract(body)
        else:
            r = self.chat(body, self.model, max_tokens=2000, meter=self.meter)
            v = _strip_fence(r["text"])
        with self._lock:
            self.cache[k] = v
            self.misses += 1
            if self.cache_path:
                with open(self.cache_path, "a") as fh:
                    fh.write(json.dumps({"k": k, "v": v}) + "\n")
        return v

    def prefetch(self, bodies: List[str], workers: int, stop: Callable[[], bool]) -> int:
        todo = [b for b in dict.fromkeys(bodies) if self.key(b) not in self.cache]

        def one(b: str) -> None:
            if stop():
                return
            try:
                self(b)
            except Exception:  # noqa: BLE001  (the real call in ingest() will retry and surface it)
                pass

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            list(ex.map(one, todo))
        return len(todo)


def ingest_corpus(corpus: Corpus, store_path: str, ext: Extractor, *, home: str, workers: int,
                  stop: Callable[[], bool], fake: bool = False, progress: Optional[Callable[[int, int], None]] = None
                  ) -> Dict[str, Any]:
    """Fresh store at ``store_path`` (must not exist), every turn through mpmf_plus.ingest.ingest."""
    from .. import provider as P, ingest as I
    if os.path.exists(store_path):
        raise FileExistsError("bench store exists: " + store_path)
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    use_store(store_path, create=True)
    mem = P._open().mem
    calls = ingest_calls(corpus)
    ext.prefetch([extraction_body(c) for c in calls], workers, stop)
    st = {"turns": len(calls), "written": 0, "skipped": 0, "failed": 0, "errors": [], "stopped": False}
    log = os.path.join(home, "health", "store.log")
    dead = os.path.join(home, "health", "ingest_dead_letter.jsonl")
    for i, c in enumerate(calls):
        if stop():
            st["stopped"] = True
            break
        try:
            r = I.ingest(c["turn_text"], said_at=c["said_at"], conv_id=c["conv_id"], msg_index=c["msg_index"],
                         context=c["context"], model=L.READER_MODEL, namespace="memory", store=mem,
                         call_model=ext, embed_fn=(fake_embed if fake else None), log_path=log,
                         dead_letter_path=dead)
            st["written"] += len(r.get("written") or [])
            st["skipped"] += len(r.get("skipped") or [])
        except Exception as e:  # noqa: BLE001  (counted, listed, never silently zero)
            st["failed"] += 1
            if len(st["errors"]) < 5:
                st["errors"].append("%s: %s" % (type(e).__name__, str(e)[:160]))
        if progress and (i + 1) % 50 == 0:
            progress(i + 1, len(calls))
    st["nodes"] = int(getattr(P._open().mem.header, "node_count", -1))
    return st


# ------------------------------------------------------------------------------------------------ contexts
def bm25_index(corpus: Corpus) -> Tuple[T.BM25, List[str]]:
    """BM25 over RAW turns (every user and assistant message is one document, date-prefixed)."""
    docs: List[str] = []
    for t in corpus.turns:
        docs.append("[%s] %s: %s" % (t.date[:10], t.speaker, t.text))
    return T.BM25(docs), docs


def bm25_context(index: Tuple[T.BM25, List[str]], query: str, k: int = BM25_K, budget: int = BM25_BUDGET,
                 item: int = BM25_ITEM) -> str:
    bm, docs = index
    out: List[str] = []
    used = 0
    for i, _s in bm.top(query, k):
        d = docs[i][:item]
        if used + len(d) + 1 > budget:
            break
        out.append(d)
        used += len(d) + 1
    return "\n".join(out)


class Recaller:
    """The MPMF recall: MPMFPlusProvider.prefetch(query) with history_mode forced by env, per arm."""

    def __init__(self) -> None:
        from .. import hermes_adapter
        self.prov = hermes_adapter.MPMFPlusProvider({})

    def context(self, query: str, history: bool) -> str:
        prev = os.environ.get("MPMF_HISTORY_MODE")
        os.environ["MPMF_HISTORY_MODE"] = "1" if history else "0"
        try:
            return self.prov.prefetch(query) or ""
        finally:
            if prev is None:
                os.environ.pop("MPMF_HISTORY_MODE", None)
            else:
                os.environ["MPMF_HISTORY_MODE"] = prev
