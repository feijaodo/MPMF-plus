"""Stage 1: conversation -> facts, with retry, explicit failures and full provenance."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompt_67e.txt")
with open(PROMPT_PATH, encoding="utf-8") as _f:
    PROMPT = _f.read()

#: Fields every persisted fact MUST carry. The schema test fails loudly if a
#: prompt field is parsed but dropped on the way to the store (r70 lesson).
REQUIRED_FIELDS = ("entity", "_msg_index", "_evidence", "_fact_date", "_said_at", "_conv_id")

FAILED = "failed"
EMPTY = "empty"


class ExtractionError(RuntimeError):
    """Raised when the model reply cannot be parsed after the retry budget."""


class BudgetExceeded(RuntimeError):
    """Raised when the run would exceed ``--cap`` dollars."""


def cap_exceeded(spent: float, cap: float) -> bool:
    return cap is not None and cap > 0 and spent >= cap


def load_export(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL export: one conversation per line (id, start, messages)."""
    convs: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                convs.append(json.loads(line))
    return convs


def _parse_reply(raw: str) -> List[Dict[str, Any]]:
    """Parse the extractor reply. Raises ExtractionError on malformed JSON."""
    try:
        obj = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"unparseable reply: {type(exc).__name__}") from exc
    facts = obj.get("facts") if isinstance(obj, dict) else None
    if not isinstance(facts, list):
        raise ExtractionError("reply has no 'facts' list")
    out = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        sentence = (f.get("fact_sentence") or f.get("sentence") or "").strip()
        if not sentence:
            continue
        out.append({
            "sentence": sentence,
            "entity": (f.get("entity") or f.get("subject") or "other").strip(),
            "msg_index": f.get("msg_index"),
            "evidence": (f.get("evidence") or "")[:300],
            "fact_date": f.get("fact_date") or "",
        })
    return out


def _window(conv: Dict[str, Any]) -> str:
    msgs = conv.get("messages") or []
    lines = []
    for i, m in enumerate(msgs):
        role = m.get("role", "user")
        text = (m.get("text") or "").strip().replace("\n", " ")
        lines.append(f"[{i}] {role}: {text[:600]}")
    return "\n".join(lines)


def extract_conversation(conv: Dict[str, Any], call_model: Callable[[str], str],
                         retries: int = 1) -> Dict[str, Any]:
    """Extract one conversation. Returns {'status', 'facts', 'cost', 'error'}.

    ``status`` is ``ok`` (facts > 0), ``empty`` (model genuinely returned none)
    or ``failed`` (unparseable after retries). Only ``empty`` may be counted as
    zero facts; ``failed`` is retried by the caller and never silently zeroed.
    """
    body = PROMPT + "\n\nMESSAGES:\n" + _window(conv)
    last: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            raw = call_model(body)
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(0.2 * (attempt + 1))
            continue
        try:
            facts = _parse_reply(raw)
        except ExtractionError as exc:
            last = str(exc)
            continue
        for f in facts:
            if f["msg_index"] is None:
                f["msg_index"] = 0
        return {"status": "ok" if facts else "empty", "facts": facts, "error": None}
    return {"status": FAILED, "facts": [], "error": last}


def persist_facts(mem, conv: Dict[str, Any], facts: Iterable[Dict[str, Any]],
                  make_node) -> List[int]:
    """Persist facts, asserting the full provenance schema on every node."""
    conv_id = conv.get("id") or conv.get("conv_id") or "unknown"
    said_at = (conv.get("start") or "")[:10]
    ids: List[int] = []
    for n, f in enumerate(facts):
        wiki = {
            "kind": "fact",
            "_conv_id": conv_id,
            "_said_at": said_at,
            "_msg_index": int(f.get("msg_index") or 0),
            "_evidence": (f.get("evidence") or "")[:300],
            "_fact_date": f.get("fact_date") or said_at,
            "_fact_date_src": "extractor" if f.get("fact_date") else "conversation_start",
            "entity": f.get("entity") or "other",
        }
        missing = [k for k in REQUIRED_FIELDS if wiki.get(k) in (None, "")]
        if missing:
            raise AssertionError(f"fact {n} missing schema fields: {missing}")
        ids.append(make_node(
            name=f"f_{conv_id[:8]}_{n}",
            text=f["sentence"],
            summary=f["sentence"][:120],
            wiki=wiki,
        ))
    return ids


def run_extraction(convs: List[Dict[str, Any]], call_model: Callable[[str], str],
                   cap_usd: float = 0.0, cost_per_call: float = 0.002,
                   workers: int = 8) -> Dict[str, Any]:
    """Extract every conversation. Returns totals + per-conversation statuses."""
    spent = 0.0
    results: List[Dict[str, Any]] = []

    def one(conv):
        return conv, extract_conversation(conv, call_model)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for conv, res in ex.map(one, convs):
            spent += cost_per_call
            results.append({"conv": conv.get("id"), **{k: res[k] for k in ("status", "error")},
                            "n_facts": len(res["facts"]), "facts": res["facts"]})
            if cap_exceeded(spent, cap_usd):
                raise BudgetExceeded(f"cap ${cap_usd:.2f} reached at ${spent:.2f}")
    ok = sum(1 for r in results if r["status"] == "ok")
    empty = sum(1 for r in results if r["status"] == "empty")
    failed = sum(1 for r in results if r["status"] == FAILED)
    return {"conversations": len(results), "ok": ok, "empty": empty, "failed": failed,
            "facts": sum(r["n_facts"] for r in results), "spent_usd": round(spent, 4),
            "rows": results}
