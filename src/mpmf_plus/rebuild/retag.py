"""Stage 3: context-aware mode re-tag.

A fact's mode cannot be judged from the fact sentence alone: "User would name a
cat Mittens" and "User named a cat Mittens" differ by one word in the *source
message*. This stage re-reads each fact's source message plus the two before it
and re-marks the mode as ``actual`` / ``hypothetical`` / ``unclear``.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Tuple

RETAG_PROMPT = """Decide the MODE of a memory fact, using the user's messages.

Messages (oldest first; the fact comes from the last one):
{context}

FACT: {fact}

- actual: the messages state a real arrangement, possession, action or event.
- hypothetical: conditional, imagined, role-play, fiction, research or a question
  ("if I", "would", "suppose", "what if", or a character/world that is not real).
- unclear: the messages do not settle it.

Return JSON: {{"mode": "actual|hypothetical|unclear", "why": "<=8 words"}}"""


def window_text(messages: List[str], msg_index: int, before: int = 2) -> str:
    lo = max(0, msg_index - before)
    hi = min(len(messages), msg_index + 1)
    return "\n".join(f"[{i}] {messages[i][:500]}" for i in range(lo, hi))


def retag_modes(facts: Iterable[Dict[str, Any]], messages_for: Callable[[str], List[str]],
                call_model: Callable[[str], str], workers: int = 8,
                before: int = 2) -> Dict[str, Any]:
    """Re-mark modes. Returns flips + the new mode per fact name."""
    rows = list(facts)

    def one(f):
        conv = f.get("_conv_id") or ""
        idx = int(f.get("_msg_index") or 0)
        ctx = window_text(messages_for(conv), idx, before)
        prompt = RETAG_PROMPT.format(context=ctx, fact=(f.get("sentence") or f.get("text") or "")[:400])
        try:
            raw = call_model(prompt)
        except Exception:  # noqa: BLE001
            return f, None
        import json
        try:
            mode = str(json.loads(raw).get("mode") or "").lower()
        except Exception:  # noqa: BLE001
            return f, None
        return f, (mode if mode in ("actual", "hypothetical", "unclear") else None)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, rows))
    flips: Dict[str, int] = {}
    modes: Dict[str, str] = {}
    for f, mode in results:
        if mode is None:
            continue
        name = f.get("name") or f.get("sentence", "")[:40]
        modes[name] = mode
        old = f.get("mode") or "actual"
        if mode != old:
            flips[f"{old}->{mode}"] = flips.get(f"{old}->{mode}", 0) + 1
    return {"facts": len(rows), "retagged": len(modes), "flips": flips, "modes": modes}
