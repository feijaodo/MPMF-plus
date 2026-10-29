"""
mpmf/ingestion/template.py — Self-evolving Question Template.

The template is a persistent JSON file of questions the extraction model
answers about each interaction. Each question feeds one MPMF plane. Over time
the template evolves: useful questions are reinforced, dead ones go to
PROBATION then RETIRED, and high-usefulness questions can spawn variants.

Per the Auto-Ingestion spec §"The Question Template".
"""

import json
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

# Question status lifecycle
ACTIVE = "ACTIVE"
PROBATION = "PROBATION"
RETIRED = "RETIRED"

# Evolution thresholds (exposed as config on QuestionTemplate)
DEFAULT_T_MIN = 20            # asks before a 0-retrieval question is judged
DEFAULT_PROBATION_RATE = 5    # probation questions asked 1-in-N ingestions


@dataclass
class Question:
    question_id: str
    plane: str                      # ASSOCIATIVE | CAUSAL | TEMPORAL | FUNCTIONAL | PRIORITY | META
    text: str
    node_type_hint: str = "CONCEPT"
    relation_hint: str = "RELATED_TO"
    depends_on: Optional[str] = None
    activation_condition: Optional[str] = None
    created_at: int = field(default_factory=lambda: int(time.time()))
    times_asked: int = 0
    memories_produced: int = 0
    retrieval_count: int = 0
    usefulness_score: float = 0.0
    status: str = ACTIVE


# Starting question set (spec §"Starting Question Set")
_BASE_QUESTIONS = [
    # ASSOCIATIVE
    ("ASSOCIATIVE", "What new concepts or topics appeared in this interaction?", "CONCEPT", "RELATED_TO"),
    ("ASSOCIATIVE", "What concepts are related to each other, and how?", "CONCEPT", "RELATED_TO"),
    ("ASSOCIATIVE", "What entities (people, tools, systems, places) were mentioned?", "ENTITY", "RELATED_TO"),
    ("ASSOCIATIVE", "What does this interaction connect to from earlier context?", "CONCEPT", "RELATED_TO"),
    # CAUSAL
    ("CAUSAL", "What caused what in this interaction?", "EVENT", "CAUSES"),
    ("CAUSAL", "What problem was being solved, and what was the proposed cause?", "CONCEPT", "CAUSES"),
    ("CAUSAL", "What decisions were made, and what reasoning drove them?", "EVENT", "CAUSES"),
    ("CAUSAL", "What was identified as a dependency or prerequisite?", "CONCEPT", "REQUIRES"),
    # TEMPORAL
    ("TEMPORAL", "What was the sequence of events or steps discussed?", "EVENT", "BEFORE"),
    ("TEMPORAL", "What happened before what?", "EVENT", "BEFORE"),
    ("TEMPORAL", "What is planned to happen next, and when?", "EVENT", "AFTER"),
    ("TEMPORAL", "What deadlines, durations, or time references were mentioned?", "EVENT", "DURING"),
    # FUNCTIONAL
    ("FUNCTIONAL", "What actions were taken or proposed?", "ACTION", "USES"),
    ("FUNCTIONAL", "What tools, methods, or procedures were used or recommended?", "TOOL", "USES"),
    ("FUNCTIONAL", "How was a particular task accomplished?", "ACTION", "ACHIEVES"),
    ("FUNCTIONAL", "What worked and what didn't?", "FACT", "IMPLEMENTS"),
    # PRIORITY
    ("PRIORITY", "What was identified as most important or urgent?", "GOAL", "MORE_IMPORTANT"),
    ("PRIORITY", "What goals were stated or implied?", "GOAL", "SUBGOAL_OF"),
    ("PRIORITY", "What are the subgoals of those goals?", "GOAL", "SUBGOAL_OF"),
    ("PRIORITY", "What trade-offs or conflicts of priority appeared?", "CONCEPT", "CONFLICTS"),
    # META (cross-plane)
    ("META", "Does anything in this interaction contradict what I already believe? If so, what?", "CLAIM", "RELATED_TO"),
    ("META", "What was surprising, novel, or worth remembering specifically?", "FACT", "RELATED_TO"),
]


class QuestionTemplate:
    def __init__(self, path: str,
                 t_min: int = DEFAULT_T_MIN,
                 probation_rate: int = DEFAULT_PROBATION_RATE):
        self.path = Path(path).expanduser()
        self.t_min = t_min
        self.probation_rate = probation_rate
        self.questions: Dict[str, Question] = {}
        self.ingestion_count = 0
        self.load()

    # ---- persistence --------------------------------------------------
    def load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.ingestion_count = data.get("ingestion_count", 0)
            for q in data.get("questions", []):
                self.questions[q["question_id"]] = Question(**q)
        else:
            self._seed()
            self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "0.1",
            "ingestion_count": self.ingestion_count,
            "questions": [asdict(q) for q in self.questions.values()],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)

    def _seed(self):
        for plane, text, ntype, rel in _BASE_QUESTIONS:
            qid = f"q_{uuid.uuid4().hex[:8]}"
            self.questions[qid] = Question(
                question_id=qid, plane=plane, text=text,
                node_type_hint=ntype, relation_hint=rel,
            )

    # ---- selection (spec Step 2) --------------------------------------
    def select(self) -> List[Question]:
        """Return questions to ask this ingestion, ordered by plane then
        usefulness desc. PROBATION questions asked only 1-in-N."""
        self.ingestion_count += 1
        chosen: List[Question] = []
        for q in self.questions.values():
            if q.status == RETIRED:
                continue
            if q.status == PROBATION and (self.ingestion_count % self.probation_rate != 0):
                continue
            chosen.append(q)
        chosen.sort(key=lambda q: (q.plane, -q.usefulness_score))
        return chosen

    # ---- feedback / evolution (spec §"Evolution Rules") ---------------
    def record_asked(self, qid: str, produced: int):
        q = self.questions.get(qid)
        if q:
            q.times_asked += 1
            q.memories_produced += produced

    def record_retrieval(self, qid: str, used: bool):
        """Called when a node produced by qid is retrieved (and maybe used)."""
        q = self.questions.get(qid)
        if not q:
            return
        q.retrieval_count += 1
        # running average of usefulness (1.0 if used, 0.3 if retrieved-not-used)
        signal = 1.0 if used else 0.3
        n = q.retrieval_count
        q.usefulness_score = ((q.usefulness_score * (n - 1)) + signal) / n
        if q.status == PROBATION:
            q.status = ACTIVE  # a probation question proved useful

    def evolve(self) -> Dict[str, int]:
        """Apply reinforce/probation/retire rules. Returns counts of changes."""
        changes = {"probation": 0, "retired": 0}
        for q in self.questions.values():
            if q.status == ACTIVE and q.times_asked > self.t_min and q.retrieval_count == 0:
                q.status = PROBATION
                changes["probation"] += 1
            elif q.status == PROBATION and q.times_asked > 2 * self.t_min and q.retrieval_count == 0:
                q.status = RETIRED
                changes["retired"] += 1
        return changes

    def add_question(self, plane: str, text: str,
                     node_type_hint: str = "CONCEPT",
                     relation_hint: str = "RELATED_TO") -> str:
        qid = f"q_{uuid.uuid4().hex[:8]}"
        self.questions[qid] = Question(
            question_id=qid, plane=plane, text=text,
            node_type_hint=node_type_hint, relation_hint=relation_hint,
        )
        return qid

    def stats(self) -> Dict:
        by_status: Dict[str, int] = {}
        for q in self.questions.values():
            by_status[q.status] = by_status.get(q.status, 0) + 1
        return {
            "total": len(self.questions),
            "by_status": by_status,
            "ingestion_count": self.ingestion_count,
        }
