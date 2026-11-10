"""mpmf_plus.completion_hook — p11 (RESEARCH): lessons at the moment of action.

A hook that fires right BEFORE an agent's completion claim (its final message), instead of
only at the start of the turn.  It looks at the turn that is about to end and checks two things:

  no_tool    the draft final message claims something is done/created/run, but there was no
             tool call in this turn at all;
  numbered   the user's prompt has numbered items (``1. ...``) and at least one item has no
             trace in the turn's tool calls (no path / backticked command / quoted token of
             that item appears in the arguments of a call that succeeded).

If a check trips, the matching lesson text is returned; the host shows it to the model at
that moment (one extra step) and the model may then finish or keep working.  The hook never
blocks a claim by itself and never edits one.

Flag ``lessons_at_action`` (recall.config.json / ``MPMF_LESSONS_AT_ACTION``), default OFF.
Off: ``maybe_lesson`` returns None before looking at anything.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

LESSONS = {
    "no_tool": (
        "[reminder before you finish] You are about to say the work is done, but you have not "
        "made a single tool call in this turn. A claim of 'done' is only true if a tool call "
        "made it so. Do the work with the tools now (and read the result back if the state "
        "matters), then report what the tool actually returned."),
    "numbered": (
        "[reminder before you finish] The request had numbered items and these have no trace "
        "in your tool calls: {items}. Every numbered item must be done (or you must say "
        "plainly that it was not) before you give a final message. Go through the list item "
        "by item, finish what is missing, and only then finish."),
}

_CLAIM_RE = re.compile(
    r"\b(done|created|completed?|finished|written|wrote|saved|removed|deleted|renamed|moved|"
    r"appended|updated|installed|deployed|built|all set|success(?:ful(?:ly)?)?|i(?:'ve| have))\b",
    re.I)
_ITEM_RE = re.compile(r"^\s*(\d{1,2})[.)]\s+(.+?)\s*$", re.M)
_TOKEN_RE = re.compile(r"`([^`]+)`|(/[\w./-]+[\w/])|(\./[\w./-]+)")


def enabled() -> bool:
    try:
        from . import paths as _paths
        return bool(_paths.load_config().get("lessons_at_action", False))
    except Exception:
        return False


def numbered_items(prompt: str) -> List[str]:
    """Text of each ``N. item`` line, in order (needs at least two to count as a list)."""
    items = [m.group(2) for m in _ITEM_RE.finditer(prompt or "")]
    return items if len(items) >= 2 else []


def _norm(s: str) -> str:
    return " ".join(str(s).split())


def _call_text(calls: Sequence[Dict[str, Any]]) -> str:
    """Arguments of the calls that SUCCEEDED (a call that errored met nothing). Calls without
    an 'ok' field are taken as successful."""
    return _norm(" ".join(json.dumps(c.get("args", {}), sort_keys=True, ensure_ascii=False)
                          for c in calls if c.get("ok", True)))


def _tokens(item: str) -> List[str]:
    out = []
    for m in _TOKEN_RE.finditer(item):
        t = next(g for g in m.groups() if g)
        out.append(_norm(t).rstrip(".,;:"))
    return out


def untouched_items(prompt: str, calls: Sequence[Dict[str, Any]]) -> List[str]:
    """Numbered items whose paths/commands appear in none of this turn's tool calls.
    Items with no extractable token cannot be judged and are not reported."""
    blob = _call_text(calls)
    miss = []
    for i, it in enumerate(numbered_items(prompt), 1):
        toks = _tokens(it)
        if toks and not any(t in blob or t.replace("\\", "\\\\") in blob for t in toks):
            miss.append(f"{i}. {it[:80]}")
    return miss


def check(prompt: str, calls: Sequence[Dict[str, Any]], draft: str) -> Dict[str, Any]:
    """calls: this turn's tool calls as dicts with 'args'. Returns {'no_tool': bool, 'numbered': [..]}."""
    return {
        "no_tool": bool(not calls and _CLAIM_RE.search(draft or "")),
        "numbered": untouched_items(prompt, calls),
    }


def lesson_for(res: Dict[str, Any]) -> Optional[str]:
    parts = []
    if res.get("no_tool"):
        parts.append(LESSONS["no_tool"])
    if res.get("numbered"):
        parts.append(LESSONS["numbered"].format(items="; ".join(res["numbered"])))
    return "\n\n".join(parts) or None


def maybe_lesson(prompt: str, calls: Sequence[Dict[str, Any]], draft: str,
                 force: Optional[bool] = None) -> Optional[str]:
    """The hook entry point. None = say nothing (flag off, or both checks pass)."""
    if not (enabled() if force is None else force):
        return None
    return lesson_for(check(prompt, calls, draft))
