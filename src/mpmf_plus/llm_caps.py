"""llm_caps — what a model can do, learned from the REPLY, never from its name (r98).

No function here maps a slug, vendor or substring ("deepseek", "claude", "-flash") to a
behaviour. The three capabilities MPMF needs are all observable in a response:

  * does the model return a reasoning trace, and in which field?
      ``reasoning_text(msg)``  — OpenAI-style ``reasoning``, DeepSeek-style
      ``reasoning_content``, OpenRouter's ``reasoning_details`` list, vLLM's
      ``reasoning_text``, Anthropic-ish ``thinking``. Empty string when the reply
      carries none; the CALLER decides what an absent trace means.
  * what did the call cost?
      ``usage_cost(data)``     — the ``usage.cost`` the provider reports. Prefer it over
      any price table keyed by a model name.
  * does the endpoint accept a request option it was sent?
      ``post_with_option_fallback`` — POST, and if the endpoint answers "Unsupported
      parameter: <option>" drop that option and retry once, remembering it per endpoint
      for the rest of the process. A model with no switchable reasoning channel is
      discovered from its error, not assumed from its name.

Tests: tests/test_r98_model_agnostic.py
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

#: fields a provider may use for the reasoning trace, in preference order.
REASONING_KEYS = ("reasoning", "reasoning_content", "reasoning_details",
                  "reasoning_text", "thinking", "reasoning_items")

#: HTTP statuses that mean "I did not like the request body" (not "the call failed").
_OPTION_REJECT = (400, 404, 422, 501)
_REJECT_WORDS = ("unsupported", "not support", "unknown", "unrecogni", "unexpected",
                 "invalid", "extra", "additional", "unexpected argument")

#: (endpoint, option) -> True once that endpoint has rejected that option.
_UNSUPPORTED: Dict[Tuple[str, str], bool] = {}


def _flatten(value: Any) -> str:
    """A reasoning field is a str, a list of part-dicts, or a dict; return its text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for k in ("text", "content", "thinking", "reasoning", "reasoning_content", "summary"):
            if isinstance(value.get(k), str) and value[k].strip():
                return value[k]
        return ""
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            t = _flatten(item)
            if t:
                parts.append(t)
        return "\n".join(parts)
    return ""


def reasoning_text(msg: Optional[dict]) -> str:
    """The reasoning trace in ``msg`` (a chat ``message`` object), or ''.

    Strips nothing else: the caller keeps whatever budget/dedup rules it had. Order
    follows REASONING_KEYS so the same reply always yields the same text.
    """
    if not isinstance(msg, dict):
        return ""
    for k in REASONING_KEYS:
        t = _flatten(msg.get(k))
        if t.strip():
            return t
    return ""


def usage_cost(data: Optional[dict]) -> Optional[float]:
    """USD the provider says this call cost, or None when it does not say."""
    if not isinstance(data, dict):
        return None
    for holder in (data.get("usage") or {}, data):
        for k in ("cost", "total_cost"):
            v = holder.get(k) if isinstance(holder, dict) else None
            if isinstance(v, (int, float)):
                return float(v)
    return None


def option_rejected(status: int, body: str, option: str) -> bool:
    """True when ``status``/``body`` say the request option ``option`` is not supported."""
    if int(status or 0) not in _OPTION_REJECT:
        return False
    b = (body or "").lower()
    if option.lower() not in b:
        return False
    return any(w in b for w in _REJECT_WORDS)


def drop_option(payload: dict, option: str) -> dict:
    """A shallow copy of ``payload`` without ``option``."""
    return {k: v for k, v in payload.items() if k != option}


def post_with_option_fallback(post: Callable[..., Any], url: str, payload: dict,
                              *, option: str = "reasoning", headers: Optional[dict] = None,
                              timeout: Optional[float] = None,
                              endpoint: Optional[str] = None) -> Tuple[Any, dict, bool]:
    """POST ``payload``; drop ``option`` if the endpoint rejects it, then retry once.

    Returns ``(response, payload_actually_sent, dropped)``. The rejection is remembered
    for ``endpoint`` (default: ``url``) so later calls do not pay the extra round-trip.
    ``post`` is anything requests.post-like (tests pass a fake).
    """
    key = (endpoint or url, option)
    send = dict(payload)
    dropped = False
    if _UNSUPPORTED.get(key):
        send = drop_option(send, option)
        dropped = option in payload
    resp = post(url, json=send, headers=headers, timeout=timeout)
    if option in send and option_rejected(getattr(resp, "status_code", 0),
                                          getattr(resp, "text", "") or "", option):
        _UNSUPPORTED[key] = True
        send = drop_option(send, option)
        dropped = True
        resp = post(url, json=send, headers=headers, timeout=timeout)
    return resp, send, dropped


def option_supported(endpoint: str, option: str = "reasoning") -> bool:
    """What this process has learned about ``endpoint`` so far (True until proven otherwise)."""
    return not _UNSUPPORTED.get((endpoint, option), False)


def reset_cache() -> None:
    """Forget everything learned (tests)."""
    _UNSUPPORTED.clear()
