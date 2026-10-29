"""
mpmf/retrieval.py — Four-Stage Retrieval Pipeline (MPMF spec §"Four-Stage Retrieval")

Operates entirely on the in-memory working set exposed by MemoryManager
(get_node, get_edges_from, get_text, get_summary, get_wiki, get_vector).

Stage 1  Graph Traversal   — BFS over ASSOCIATIVE (+ optional CAUSAL/PRIORITY)
                             planes from seed nodes, up to depth D.
Stage 2  Wiki Expansion    — pull key/value facts from neighborhood nodes,
                             use them as extra query terms to pull in matches.
Stage 3  Vector Search     — composite (mean) query vector over the working set,
                             cosine-score all candidate nodes, keep top-K.
Stage 4  Summary-Guided    — score node summaries against the original query,
         Raw Text            load full raw text for only the top-ranked nodes.

The pipeline can iterate: Stage-4 top nodes seed Stage-1 of the next pass.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# numpy powers the vector stage (contiguous matrix + single matmul). It is
# optional: when absent we fall back to the original per-node Python cosine so
# the module keeps working anywhere it is imported.
try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy is present in the live env
    _np = None

from .format import Plane, NodeRecord, NodeType
from .embedding import EmbeddingProvider


_WORD_RE = re.compile(r"[a-z0-9]+")


def _terms(text: str) -> Set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class RetrievalResult:
    query: str
    nodes: List[int] = field(default_factory=list)        # full working set (ids)
    top_nodes: List[int] = field(default_factory=list)    # top-ranked ids
    raw_texts: Dict[int, str] = field(default_factory=dict)
    summaries: Dict[int, str] = field(default_factory=dict)
    iterations: int = 0
    # Per-phase wall times (seconds) for observability. Keys: embed, traverse,
    # wiki_expand, vector, lexical, total. Populated by query().
    timings: Dict[str, float] = field(default_factory=dict)


class RetrievalPipeline:
    def __init__(
        self,
        memory_manager,
        embedder: Optional[EmbeddingProvider] = None,
        depth: int = 2,
        top_k: int = 20,
        max_iterations: int = 2,
        planes: Optional[List[Plane]] = None,
        graph_bonus_depth1: float = 0.15,
        graph_bonus_depth2: float = 0.08,
        hub_propagation: float = 0.60,
        return_hubs: bool = False,
        hub_max_words: int = 3,
        wiki_expand: bool = True,
        wiki_min_hits: int = 2,
        wiki_max_add: int = 300,
        wiki_df_max: float = 0.10,
        wiki_seed_cap: int = 10,
    ):
        self.mem = memory_manager
        self.embedder = embedder or EmbeddingProvider(memory_manager.vector_dimensions)
        self.depth = depth
        self.top_k = top_k
        self.max_iterations = max_iterations
        self.planes = planes or [Plane.ASSOCIATIVE, Plane.CAUSAL, Plane.PRIORITY]
        # Graph-weighted ranker bonuses — nodes close to seeds get a lift.
        # Depth 1 (direct neighbor of seed) gets a boost, depth 2 gets a smaller one.
        # These are configurable so the user can tune or disable (set to 0).
        self.graph_bonus_depth1 = graph_bonus_depth1
        self.graph_bonus_depth2 = graph_bonus_depth2
        # Hub routing. hub_propagation=0.0 disables it entirely;
        # return_hubs=True restores the old behaviour of returning bare
        # idea nodes as results.
        self.hub_propagation = hub_propagation
        self.return_hubs = return_hubs
        self.hub_max_words = hub_max_words
        self.wiki_expand = wiki_expand
        self.wiki_min_hits = wiki_min_hits
        self.wiki_max_add = wiki_max_add
        self.wiki_df_max = wiki_df_max
        self.wiki_seed_cap = wiki_seed_cap
        self._hub_cache: Dict[int, bool] = {}

    # ---- Per-instance caches ------------------------------------------
    # The pipeline is constructed once per recall; these caches let the
    # 2-iteration loop (and the bootstrap seed search) reuse work computed on
    # iteration 1 instead of re-scoring the whole store on every pass.
    def _query_vec(self, query: str):
        """Embed the query once per pipeline (Ollama HTTP), cached by text."""
        cache = getattr(self, "_qvec_cache", None)
        if cache is None:
            cache = self._qvec_cache = {}
        if query not in cache:
            cache[query] = self.embedder.embed(query, kind="query")
        return cache[query]

    def _node_vector_matrix(self):
        """Return (ids, M, norms) for every node carrying a vector.

        M is a contiguous (N, dim) float32 numpy matrix. Cached ON the memory
        manager object so the (expensive) matrix build happens once per loaded
        store and is reused by every subsequent pipeline/recall, not rebuilt on
        each query. If numpy is unavailable, M is a list of row lists and
        norms is None (pure-Python fallback).
        """
        cached = getattr(self.mem, "_mpmf_vec_matrix", None)
        if cached is not None:
            return cached
        ids: List[int] = []
        rows: List[list] = []
        for nid in self.mem.all_node_ids():
            v = self.mem.get_vector(nid)
            if v is not None:
                ids.append(nid)
                rows.append(v)
        if _np is not None and rows:
            M = _np.asarray(rows, dtype=_np.float32)
            norms = _np.linalg.norm(M, axis=1)
            out = (ids, M, norms)
        else:
            out = (ids, rows, None)
        try:
            self.mem._mpmf_vec_matrix = out
        except Exception:
            pass
        return out

    def _doc_terms(self):
        """Name+summary+text+wiki term set for every node, cached on the mem
        object (built once per store, reused across recalls/iterations)."""
        cached = getattr(self.mem, "_mpmf_doc_terms", None)
        if cached is not None:
            return cached
        docs: Dict[int, Set[str]] = {}
        for nid in self.mem.all_node_ids():
            blob = " ".join([
                self.mem.get_name(nid) or "",
                self.mem.get_summary(nid) or "",
                self.mem.get_text(nid) or "",
                " ".join(f"{k} {v}" for k, v in self.mem.get_wiki(nid).items()),
            ])
            docs[nid] = _terms(blob)
        try:
            self.mem._mpmf_doc_terms = docs
        except Exception:
            pass
        return docs

    # ---- Stage 1 -------------------------------------------------------
    def _traverse(self, seeds: Set[int], depth: int) -> Tuple[Set[int], Dict[int, int]]:
        """BFS traversal over edge planes. Returns (visited_nodes, depth_map)."""
        visited: Set[int] = set()
        depth_map: Dict[int, int] = {}  # node_id → shortest BFS depth from any seed
        frontier: Set[int] = set(seeds)
        # Seeds are depth 0
        for s in seeds:
            depth_map[s] = 0
        plane_mask = 0
        for p in self.planes:
            plane_mask |= int(p.value)
        for d in range(1, depth + 1):
            nxt: Set[int] = set()
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                for edge in self.mem.get_edges_from(nid):
                    if int(edge.plane_id) & plane_mask:
                        tgt = edge.target_node_id
                        if tgt not in visited and tgt not in depth_map:
                            depth_map[tgt] = d
                            nxt.add(tgt)
            frontier = nxt
            if not frontier:
                break
        visited |= set(seeds)
        return visited, depth_map

    # ---- Stage 2 -------------------------------------------------------
    # The wiki block holds TWO kinds of key:
    #   * reserved bookkeeping (leading underscore) used by ingestion/decay —
    #     _qid, _base_weight, _reinforced, _ns, _extraction_model. These are
    #     identical or near-identical across the whole store and carry no
    #     retrieval signal.
    #   * semantic tags written by the extractor — subject, topic, agent,
    #     relates_to, trigger, component, project ...  THESE are the point of
    #     this stage: they catch nodes whose prose doesn't embed near the query
    #     but whose tags say they concern the same subject.
    # Matching on the first kind pulls in the entire store on every query.
    _WIKI_RESERVED = {"planes", "priority", "weight", "plane"}

    def _wiki_terms(self, nid: int) -> Set[str]:
        cache = getattr(self, "_wiki_term_cache", None)
        if cache is None:
            cache = self._wiki_term_cache = {}
        if nid in cache:
            return cache[nid]
        ts: Set[str] = set()
        for k, v in self.mem.get_wiki(nid).items():
            if k.startswith("_") or k.lower() in self._WIKI_RESERVED:
                continue
            # Value terms only: key names (`subject`, `topic`) are schema,
            # not content, and are shared by thousands of nodes.
            ts |= _terms(str(v))
        cache[nid] = ts
        return ts

    def _wiki_df(self) -> Dict[str, int]:
        df = getattr(self, "_wiki_df_cache", None)
        if df is not None:
            return df
        from collections import Counter
        c: Counter = Counter()
        for nid in self.mem.all_node_ids():
            for t in self._wiki_terms(nid):
                c[t] += 1
        self._wiki_df_cache = c
        return c

    def _wiki_expand(self, working: Set[int], query: str) -> Set[int]:
        if not self.wiki_expand:
            return set()
        n_total = max(1, len(list(self.mem.all_node_ids())))
        df = self._wiki_df()
        cutoff = self.wiki_df_max * n_total

        def informative(ts: Set[str]) -> Set[str]:
            return {t for t in ts if df.get(t, 0) <= cutoff and len(t) > 2}

        query_terms = informative(_terms(query))
        # Enrich from the SEEDS only. Harvesting the whole working set makes
        # the term set the union of the store after a single iteration.
        for nid in sorted(working)[: self.wiki_seed_cap]:
            query_terms |= informative(self._wiki_terms(nid))
        if not query_terms:
            return set()

        scored: List[Tuple[float, int]] = []
        for nid in self.mem.all_node_ids():
            if nid in working:
                continue
            wt = informative(self._wiki_terms(nid))
            if not wt:
                continue
            overlap = query_terms & wt
            if len(overlap) < self.wiki_min_hits:
                continue
            # Rarer shared tags are stronger evidence of the same subject.
            score = sum(1.0 / (1.0 + df.get(t, 1)) for t in overlap)
            scored.append((score, nid))
        scored.sort(reverse=True)
        return {nid for _, nid in scored[: self.wiki_max_add]}

    # ---- Stage 3 -------------------------------------------------------
    def _vector_scores(self, query: str, context_ids: Set[int] = None) -> Dict[int, float]:
        """Cosine score every node against the query embedding, min-max
        normalized to [0,1] across the store.

        FAST PATH: one numpy matrix-multiply over a contiguous (N, dim) matrix
        with precomputed row norms — O(N·dim) in C instead of a per-node Python
        loop (the previous ~670 ms/pass cost). The query embedding is embedded
        once and the full score dict is cached per query, so the bootstrap seed
        search and BOTH loop iterations reuse a single computation.
        """
        cache = getattr(self, "_vscores_cache", None)
        if cache is None:
            cache = self._vscores_cache = {}
        if query in cache:
            return cache[query]

        qvec = self._query_vec(query)
        ids, M, norms = self._node_vector_matrix()
        raw: Dict[int, float] = {}
        if not ids:
            cache[query] = {}
            return {}
        if _np is not None and norms is not None:
            q = _np.asarray(qvec, dtype=_np.float32)
            qn = float(_np.linalg.norm(q)) or 1.0
            sims = (M @ q) / (norms * qn + 1e-9)
            for nid, s in zip(ids, sims):
                raw[int(nid)] = float(s)
        else:  # pragma: no cover - numpy fallback
            for nid, row in zip(ids, M):
                raw[nid] = _cosine(row, qvec)

        lo = min(raw.values())
        hi = max(raw.values())
        span = (hi - lo) or 1.0
        out = {nid: (s - lo) / span for nid, s in raw.items()}
        cache[query] = out
        return out

    def _vector_search(self, candidates: Set[int], query: str) -> List[Tuple[int, float]]:
        scores = self._vector_scores(query, candidates)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[: self.top_k]

    def _composite(self, ids: Set[int]) -> Optional[List[float]]:
        dim = self.mem.vector_dimensions
        acc = [0.0] * dim
        n = 0
        for nid in ids:
            v = self.mem.get_vector(nid)
            if v is not None:
                for i in range(dim):
                    acc[i] += v[i]
                n += 1
        if n == 0:
            return None
        return [a / n for a in acc]

    # ---- Stage 4 -------------------------------------------------------
    def _lexical_scores(self, working: Set[int], query: str) -> Dict[int, float]:
        """IDF-weighted lexical overlap of the query against each node's full
        searchable text (name + summary + text + wiki keys/values). Rare query
        terms (e.g. 'openclaw', 'docker') carry far more weight than common
        ones, which is exactly what a small embedding model fails to capture.

        The per-node term sets are cached on the mem object and the raw
        IDF-weighted sums are cached per query on this pipeline, so the second
        loop iteration is a cheap dict re-normalization, not a full re-scan.
        Normalization stays over the caller's `working` set (identical results
        to the previous implementation).
        """
        qterms = _terms(query)
        if not qterms:
            return {nid: 0.0 for nid in working}

        raw_cache = getattr(self, "_lexraw_cache", None)
        if raw_cache is None:
            raw_cache = self._lexraw_cache = {}
        raw_full = raw_cache.get(query)
        if raw_full is None:
            docs = self._doc_terms()
            all_ids = self.mem.all_node_ids()
            N = max(1, len(all_ids))
            # document frequency of each query term across ALL nodes
            df = {t: sum(1 for s in docs.values() if t in s) for t in qterms}
            idf = {t: max(math.log((N + 1) / (df.get(t, 0) + 0.5)), 0.0)
                   for t in qterms}
            raw_full = {}
            for nid in all_ids:
                terms = docs.get(nid, set())
                score = 0.0
                for t in qterms & terms:
                    score += idf[t]
                raw_full[nid] = score
            raw_cache[query] = raw_full

        raw = {nid: raw_full.get(nid, 0.0) for nid in working}
        hi = max(raw.values()) if raw else 0.0
        if hi <= 0:
            return {nid: 0.0 for nid in working}
        return {nid: s / hi for nid, s in raw.items()}

    def _final_rank(self, working: Set[int], query: str,
                    vec_scores: Dict[int, float], k: int,
                    bfs_depth: Dict[int, int] = None) -> List[int]:
        """Hybrid ranking: semantic similarity + IDF lexical + importance +
        graph distance bonus (nodes close to seeds get a contextual boost)."""
        lex_scores = self._lexical_scores(working, query)

        base: Dict[int, float] = {}
        for nid in working:
            vscore = vec_scores.get(nid, 0.0)
            lscore = lex_scores.get(nid, 0.0)
            node = self.mem.get_node(nid)
            imp = node.importance if node else 0.0
            base[nid] = (0.55 * vscore) + (0.55 * lscore) + (0.10 * imp)

        # An idea hub ("Postgres", "job queue") is an index entry, not a
        # memory. Returning it wastes a result slot and, with a graph bonus
        # applied, it outranks the memories it points at. So hubs ROUTE:
        # a hub's score flows to the memories attached to it, and the hub
        # itself is withheld from the results.
        final: Dict[int, float] = dict(base)
        if self.hub_propagation > 0.0:
            for nid in working:
                if not self._is_hub(nid):
                    continue
                s = base.get(nid, 0.0)
                if s <= 0.0:
                    continue
                for e in self.mem.get_edges_from(nid):
                    tgt = e.target_node_id
                    if tgt in final and not self._is_hub(tgt):
                        final[tgt] += self.hub_propagation * s

        scored: List[Tuple[int, float]] = []
        for nid in working:
            if self._is_hub(nid) and not self.return_hubs:
                continue
            sc = final.get(nid, 0.0)
            # Graph-weighted bonus: nodes at BFS depth 1 or 2 from the
            # original seeds get a boost so closely connected facts surface.
            if bfs_depth is not None:
                d = bfs_depth.get(nid, 999)
                if d == 1:
                    sc += self.graph_bonus_depth1
                elif d == 2:
                    sc += self.graph_bonus_depth2
            scored.append((nid, sc))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in scored[:k]]

    def _is_hub(self, nid: int) -> bool:
        """A hub is a bare idea/entity node: an ENTITY whose text is just the
        idea's name rather than a statement about it."""
        if nid in self._hub_cache:
            return self._hub_cache[nid]
        node = self.mem.get_node(nid)
        txt = self.mem.get_text(nid) or ""
        is_hub = bool(
            node is not None
            and node.node_type == NodeType.ENTITY
            and len(txt.split()) <= self.hub_max_words
        )
        self._hub_cache[nid] = is_hub
        return is_hub

    # ---- Orchestration -------------------------------------------------
    def query(
        self,
        seed_text: str,
        seed_node_ids: Optional[Set[int]] = None,
        top_n: int = 10,
    ) -> RetrievalResult:
        _pc = time.perf_counter
        result = RetrievalResult(query=seed_text)
        t = result.timings
        for k in ("embed", "traverse", "wiki_expand", "vector", "lexical"):
            t[k] = 0.0
        t_all = _pc()

        # Query embedding (cached per pipeline — one Ollama round-trip).
        _s = _pc()
        self._query_vec(seed_text)
        t["embed"] += _pc() - _s

        # Bootstrap seeds: explicit ids, or vector similarity, or name match.
        _s = _pc()
        if seed_node_ids:
            seeds = set(seed_node_ids)
        else:
            seeds = set(nid for nid, _ in self._vector_search(set(), seed_text)[:5])
            if not seeds:
                qterms = _terms(seed_text)
                for nid in self.mem.all_node_ids():
                    if _terms(self.mem.get_name(nid)) & qterms:
                        seeds.add(nid)
        t["vector"] += _pc() - _s

        working: Set[int] = set(seeds)
        top: List[int] = []
        bfs_depth: Dict[int, int] = {}

        # Graph distance is measured from the ORIGINAL seeds, once.
        # Recomputing it each iteration is wrong: by iteration 2 `working` has
        # absorbed most of the store, so every node becomes a seed at depth 0,
        # no node is left at depth 1 or 2, and the graph bonus computed on
        # iteration 1 is silently erased before the final ranking.
        _s = _pc()
        _, seed_depths = self._traverse(set(seeds), self.depth)
        t["traverse"] += _pc() - _s
        bfs_depth.update(seed_depths)

        for it in range(self.max_iterations):
            _s = _pc()
            neighborhood, _ = self._traverse(working, self.depth)
            working |= neighborhood
            t["traverse"] += _pc() - _s

            _s = _pc()
            working |= self._wiki_expand(working, seed_text)
            t["wiki_expand"] += _pc() - _s

            # Stage 3: score candidates by query-vector similarity, pull in top-K.
            # (_vector_scores is cached per query, so only the first call pays.)
            _s = _pc()
            vec_scores = self._vector_scores(seed_text, working)
            for nid, _ in sorted(vec_scores.items(), key=lambda x: x[1], reverse=True)[: self.top_k]:
                working.add(nid)
            t["vector"] += _pc() - _s

            # Stage 4: final rank using vector + lexical + importance + graph bonus.
            _s = _pc()
            top = self._final_rank(working, seed_text, vec_scores, top_n, bfs_depth)
            t["lexical"] += _pc() - _s
            result.iterations = it + 1
            if not top:
                break
            working |= set(top)

        t["total"] = _pc() - t_all
        result.nodes = sorted(working)
        result.top_nodes = top
        for nid in top:
            result.raw_texts[nid] = self.mem.get_text(nid)
            result.summaries[nid] = self.mem.get_summary(nid)
        return result
