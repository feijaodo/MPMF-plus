"""
mpmf/ingestion/ingest.py — Auto-ingestion orchestrator.

Implements the spec's ingestion process:
  Trigger -> Question Selection -> Extraction -> Node/Edge Building
  -> Dedup/Consolidation -> Contradiction/Decay -> Save.

Plus retrieval-attribution feedback (record_retrieval) and template evolution.

Designed to run as a BACKGROUND task between interactions, not in the live
response path (extraction on a local 7B is ~5-15 tok/s).
"""

import math
import re
import time
from typing import Dict, List, Optional

from ..format import Plane, NodeType
from ..query import MPMFMemory
from ..embedding import EmbeddingProvider
from .template import QuestionTemplate
from .extract import Extractor, Answer
from . import decay

_PLANE_MAP = {
    "ASSOCIATIVE": Plane.ASSOCIATIVE, "CAUSAL": Plane.CAUSAL,
    "TEMPORAL": Plane.TEMPORAL, "FUNCTIONAL": Plane.FUNCTIONAL,
    "PRIORITY": Plane.PRIORITY, "META": Plane.ASSOCIATIVE,
}
_NTYPE_MAP = {
    "CONCEPT": NodeType.CONCEPT, "ENTITY": NodeType.ENTITY, "EVENT": NodeType.EVENT,
    "ACTION": NodeType.ACTION, "GOAL": NodeType.GOAL, "FACT": NodeType.FACT,
    "CLAIM": NodeType.CLAIM, "CONTEXT": NodeType.CONTEXT, "AGENT": NodeType.AGENT,
    "TOOL": NodeType.TOOL,
}

DEDUP_THRESHOLD = 0.92  # cosine similarity above which we consolidate (spec)


def _slug(text: str, maxlen: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (s[:maxlen] or "node").rstrip("_")


_STOPWORDS = {
    "the", "a", "an", "was", "were", "is", "are", "be", "been", "being",
    "to", "of", "for", "in", "on", "at", "by", "with", "from", "as",
    "that", "this", "it", "its", "and", "or", "but", "if", "then",
    "made", "being", "there", "their", "has", "have", "had", "will",
    "would", "should", "could", "can", "may", "might", "must", "do",
    "does", "did", "not", "no", "so", "up", "out", "about", "into",
    "use", "used", "using", "need", "needs", "problem", "solved",
}


def _concise_name(answer: str, entities: List[str], maxlen: int = 40) -> str:
    """Build a short, human-readable node name.

    Prefers a salient entity over the full answer sentence, so nodes are named
    `cost_conscious_animations` rather than
    `the_problem_being_solved_was_the_need_for_cost`. Falls back to the
    significant tokens of the answer with stopwords dropped.
    """
    # 1. Prefer the shortest entity that carries at least one non-stopword.
    usable = []
    for ent in entities or []:
        toks = [t for t in re.split(r"[^a-z0-9]+", (ent or "").lower()) if t]
        if any(t not in _STOPWORDS for t in toks):
            usable.append(ent)
    if usable:
        best = min(usable, key=len)
        s = _slug(best, maxlen)
        if s:
            return s

    # 2. Otherwise take significant tokens from the answer itself.
    toks = [t for t in re.split(r"[^a-z0-9]+", (answer or "").lower()) if t]
    sig = [t for t in toks if t not in _STOPWORDS]
    if not sig:
        sig = toks
    out = "_".join(sig[:4])[:maxlen].strip("_")
    return out or "node"


def _cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class AutoIngest:
    def __init__(self, store_path: str, template_path: str,
                 vector_dimensions: int = 1024,
                 embed_model: str = "mxbai-embed-large",
                 extraction_model: str = "mistral:7b-instruct-q4_K_M",
                 dedup_threshold: float = DEDUP_THRESHOLD):
        self.mem = MPMFMemory(store_path, vector_dimensions=vector_dimensions,
                              embed_model=embed_model)
        self.template = QuestionTemplate(template_path)
        self.extractor = Extractor(model=extraction_model)
        self.embedder = self.mem.embedder
        self.dedup_threshold = dedup_threshold

    # ------------------------------------------------------------------
    def _find_similar(self, vec: List[float]):
        """Return (node_id, score) of the most similar existing node, or None."""
        best_id, best = None, 0.0
        for nid in self.mem.mem.all_node_ids():
            v = self.mem.mem.get_vector(nid)
            s = _cosine(vec, v) if v else 0.0
            if s > best:
                best_id, best = nid, s
        return (best_id, best) if best_id else None

    def _upsert(self, name: str, text: str, ntype, plane_flags: int,
                qid: str, confidence: float, now: float) -> int:
        """Create a node, or consolidate into an existing similar one.
        Returns the node_id."""
        vec = self.embedder.embed(text, kind="document")
        sim = self._find_similar(vec)
        meta = {decay.KEY_QID: str(qid), decay.KEY_BASE: f"{confidence:.3f}",
                decay.KEY_REINFORCED: str(int(now))}

        if sim and sim[1] >= self.dedup_threshold:
            nid = sim[0]
            # consolidate: refresh text/vector, reinforce, keep latest qid
            self.mem.mem.set_node_text(nid, text)
            self.mem.mem.set_node_summary(nid, text[:120])
            self.mem.mem.set_node_vector(nid, vec)
            self.mem.mem.update_node_wiki(nid, meta)
            return nid

        node = self.mem.remember(
            text, name=self._unique_name(name), node_type=ntype,
            planes=plane_flags, summary=text[:120],
            wiki=dict(meta), importance=confidence,
        )
        return node.node_id

    def _unique_name(self, base: str) -> str:
        name = base
        i = 1
        while self.mem.mem.get_node_id_by_name(name) is not None:
            i += 1
            name = f"{base}_{i}"
        return name

    # ------------------------------------------------------------------
    def ingest(self, interaction: str, model_override: Optional[str] = None,
               verbose: bool = False) -> Dict:
        """Run one full ingestion pass over an interaction transcript."""
        now = time.time()
        questions = self.template.select()
        created_nodes = 0
        consolidated = 0
        created_edges = 0
        none_count = 0
        failed = 0

        for q in questions:
            ans = self.extractor.ask(interaction, q.text, model_override=model_override)
            if ans is None:
                failed += 1
                self.template.record_asked(q.question_id, 0)
                continue
            if ans.is_none:
                none_count += 1
                self.template.record_asked(q.question_id, 0)
                continue

            plane = _PLANE_MAP.get(q.plane, Plane.ASSOCIATIVE)
            ntype = _NTYPE_MAP.get(q.node_type_hint, NodeType.CONCEPT)
            pre_n = len(self.mem.mem.all_node_ids())

            # main answer node
            # Name the memory from its own significant tokens, NOT from an
            # entity -- naming it after an entity collides with that entity's
            # hub node and produces postgres_2, postgres_3, ...
            main_name = _concise_name(ans.answer, [])
            main_id = self._upsert(main_name, ans.answer, ntype,
                                   plane.value, q.question_id, ans.confidence, now)
            name_to_id = {ans.answer.lower(): main_id}

            # entity nodes
            entity_ids = []
            for ent in ans.entities:
                eid = self._upsert(_slug(ent), ent, NodeType.ENTITY,
                                   Plane.ASSOCIATIVE.value, q.question_id,
                                   ans.confidence, now)
                name_to_id[ent.lower()] = eid
                entity_ids.append(eid)

            # Attach the memory to each idea hub it mentions.
            # Without this, extracted relations only ever connect entity to
            # entity, and the answer node -- the actual memory -- ends up with
            # zero edges. Traversing from an idea then reaches other ideas but
            # never the content, which defeats idea-level retrieval.
            for eid in entity_ids:
                if eid != main_id:
                    self.mem.mem.create_edge(
                        eid, main_id, Plane.ASSOCIATIVE,
                        "evidenced_by", weight=ans.confidence)
                    self.mem.mem.create_edge(
                        main_id, eid, Plane.ASSOCIATIVE,
                        "mentions", weight=ans.confidence)
                    created_edges += 2

            # relations as edges on this question's plane
            for rel in ans.relations:
                sid = name_to_id.get(rel.src.lower())
                tid = name_to_id.get(rel.tgt.lower())
                if sid is None:
                    sid = self._upsert(_slug(rel.src), rel.src, NodeType.CONCEPT,
                                       plane.value, q.question_id, ans.confidence, now)
                    name_to_id[rel.src.lower()] = sid
                if tid is None:
                    tid = self._upsert(_slug(rel.tgt), rel.tgt, NodeType.CONCEPT,
                                       plane.value, q.question_id, ans.confidence, now)
                    name_to_id[rel.tgt.lower()] = tid
                if sid != tid:
                    self.mem.mem.create_edge(
                        sid, tid, plane,
                        rel.rtype or q.relation_hint, weight=ans.confidence)
                    created_edges += 1

            post_n = len(self.mem.mem.all_node_ids())
            new_here = post_n - pre_n
            created_nodes += new_here
            # nodes touched but not newly created => consolidations
            consolidated += max(0, (1 + len(ans.entities)) - new_here)
            self.template.record_asked(q.question_id, max(1, new_here))
            if verbose:
                print(f"  [{q.plane}] {q.text[:40]}... -> +{new_here} nodes")

        self.mem.save()
        self.template.evolve()
        self.template.save()
        return {
            "questions_asked": len(questions),
            "nodes_created": created_nodes,
            "consolidated": consolidated,
            "edges_created": created_edges,
            "none_answers": none_count,
            "extraction_failures": failed,
            "store": self.mem.stats(),
        }

    # ------------------------------------------------------------------
    def recall(self, query: str, top_n: int = 5, reinforce_used: bool = True):
        """Recall with retrieval attribution: reinforces returned nodes and
        feeds usefulness signals back to the template."""
        res = self.mem.recall(query, top_n=top_n)
        now = time.time()
        out = []
        touched = False
        for rank, nid in enumerate(res.top_nodes):
            wiki = self.mem.mem.get_wiki(nid)
            qid = wiki.get(decay.KEY_QID)
            used = reinforce_used and rank == 0  # treat top hit as "used"
            if qid:
                self.template.record_retrieval(qid, used=used)
            if used:
                self.mem.mem.update_node_wiki(nid, decay.reinforce(wiki, now))
                touched = True
            out.append({
                "name": self.mem.mem.get_name(nid),
                "text": self.mem.mem.get_text(nid),
                "effective_weight": round(decay.effective_weight(wiki, now), 3),
                "qid": qid,
            })
        if touched:
            self.mem.save()
            self.template.save()
        return out