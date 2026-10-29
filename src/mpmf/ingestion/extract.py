"""
mpmf/ingestion/extract.py — Extraction model caller.

Poses a single template question against an interaction transcript and parses
a structured answer. The system prompt enforces the spec's critical guardrail:
"answer ONLY from explicit content, respond NONE if absent" — this is the main
defense against hallucinated memories.

Uses Ollama's /api/generate with format=json for reliable structured output.
Supports a per-call model override that resets automatically (no persistent
state change).
"""

import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import List, Dict, Optional

OLLAMA_GENERATE = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral:7b-instruct-q4_K_M"

_SYSTEM = (
    "You are a memory extraction component. Answer the question about the "
    "conversation below using ONLY information explicitly present. If the "
    "conversation contains no relevant information, set answer to \"NONE\". "
    "Do not infer or invent. Be concise and structured."
)

_PROMPT_TMPL = """{system}

CONVERSATION:
{interaction}

QUESTION: {question}

Respond in JSON with exactly these keys:
{{
  "answer": "...",
  "entities": ["..."],
  "relations": [{{"from": "...", "to": "...", "type": "..."}}],
  "confidence": 0.0
}}
If there is no relevant information, set "answer" to "NONE"."""


@dataclass
class Relation:
    src: str
    tgt: str
    rtype: str


@dataclass
class Answer:
    answer: str
    entities: List[str] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    confidence: float = 0.5

    @property
    def is_none(self) -> bool:
        a = (self.answer or "").strip().upper()
        return a == "" or a == "NONE"


class Extractor:
    def __init__(self, model: str = DEFAULT_MODEL, timeout: int = 120):
        self.model = model
        self.timeout = timeout

    def _call(self, prompt: str, model: str) -> str:
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0},
        }).encode()
        req = urllib.request.Request(
            OLLAMA_GENERATE, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        return data.get("response", "")

    @staticmethod
    def _parse(raw: str) -> Optional[Answer]:
        raw = raw.strip()
        # tolerate code fences / stray prose around the JSON
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        rels = []
        for r in obj.get("relations", []) or []:
            if isinstance(r, dict):
                rels.append(Relation(
                    src=str(r.get("from", "")).strip(),
                    tgt=str(r.get("to", "")).strip(),
                    rtype=str(r.get("type", "")).strip(),
                ))
        try:
            conf = float(obj.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        ents = [str(e).strip() for e in (obj.get("entities") or []) if str(e).strip()]
        return Answer(
            answer=str(obj.get("answer", "")).strip(),
            entities=ents,
            relations=[r for r in rels if r.src and r.tgt],
            confidence=max(0.0, min(1.0, conf)),
        )

    def ask(self, interaction: str, question: str,
            model_override: Optional[str] = None) -> Optional[Answer]:
        """Pose one question. Returns Answer (possibly is_none) or None on
        a hard failure (model unreachable / unparseable). The override applies
        only to this call."""
        model = model_override or self.model
        prompt = _PROMPT_TMPL.format(
            system=_SYSTEM, interaction=interaction[:8000], question=question,
        )
        try:
            raw = self._call(prompt, model)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            return None
        return self._parse(raw)
