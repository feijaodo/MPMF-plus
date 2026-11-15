"""b0 model client: one OpenRouter chat call with cost accounting, plus a deterministic fake.

Cost comes from the response's ``usage.cost`` (requested with ``usage: {include: true}``); when the
provider omits it the token counts are priced from PRICES (USD per token, OpenRouter listing, 2026-11-14).
A call with neither is booked at 0 and flagged ``cost_known=False``.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Dict, Optional

from .. import llm_caps as _caps   # r98: capability from the reply, not the model name

READER_MODEL = "deepseek/deepseek-v4.1-flash"
BASE_URL = "https://openrouter.ai/api/v1"

#: USD per token (prompt, completion) from the OpenRouter model listing of 2026-11-14.
PRICES: Dict[str, tuple] = {
    "deepseek/deepseek-v4.1-flash": (5.07e-8, 2.028e-7),
    "openai/gpt-4o-mini": (1.5e-7, 6e-7),
    "openai/gpt-4.1-mini": (4e-7, 1.6e-6),
    "openai/gpt-4o-2024-08-06": (2.5e-6, 1e-5),
    "openai/gpt-4o": (2.5e-6, 1e-5),
}


class Meter:
    """Thread-safe running spend (USD) and call count."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.usd = 0.0
        self.calls = 0
        self.unknown = 0

    def add(self, usd: float, known: bool = True) -> None:
        with self._lock:
            self.usd += usd
            self.calls += 1
            if not known:
                self.unknown += 1


def _api_key() -> str:
    from .. import provider as _p      # lazy: provider reads MPMF_ENV_FILE once, never prints the key
    k = _p._api_key()
    if not k:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return k


def chat(prompt: str, model: str = READER_MODEL, *, max_tokens: int = 400, temperature: float = 0.0,
         meter: Optional[Meter] = None, retries: int = 3, timeout: int = 120) -> Dict[str, Any]:
    """One user-message completion. Returns {"text", "cost", "cost_known", "model", "finish"}.

    r98: no slug decides this. The first attempt is sent unchanged; if the reply comes back with
    content empty but a reasoning trace present (the r83b failure), the retry switches the
    reasoning channel off — the model TOLD us it spends its budget there. Empty content is
    retried, then raises.
    """
    import requests
    payload: Dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": prompt}],
                               "temperature": temperature, "max_tokens": max_tokens,
                               "usage": {"include": True}}
    reasoning_off = False          # flipped by the first reply that is pure trace, never by a name
    last: Optional[str] = None
    total, known = 0.0, True
    for attempt in range(retries + 1):
        try:
            r = requests.post(BASE_URL + "/chat/completions", json=payload, timeout=timeout,
                              headers={"Authorization": "Bearer " + _api_key(),
                                       "Content-Type": "application/json"})
            if r.status_code in (429, 500, 502, 503, 504):
                last = "http %d" % r.status_code
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            d = r.json()
            u = d.get("usage") or {}
            c = u.get("cost")
            if c is None:
                pr = PRICES.get(model)
                if pr and u:
                    c = float(u.get("prompt_tokens") or 0) * pr[0] + float(u.get("completion_tokens") or 0) * pr[1]
                else:
                    c, known = 0.0, False
            total += float(c)
            if meter is not None:
                meter.add(float(c), known)
            m = d["choices"][0]["message"]
            txt = m.get("content")
            if not reasoning_off and not (isinstance(txt, str) and txt.strip()) \
                    and _caps.reasoning_text(m).strip():
                payload["reasoning"] = {"enabled": False}
                reasoning_off = True
                last = "content empty, reasoning present -> retry with reasoning off"
            if isinstance(txt, str) and txt.strip():
                return {"text": txt.strip(), "cost": total, "cost_known": known,
                        "model": d.get("model", model), "finish": d["choices"][0].get("finish_reason", "")}
            last = "empty content"
        except Exception as e:  # noqa: BLE001
            last = "%s: %s" % (type(e).__name__, str(e)[:200])
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("chat(%s) failed after %d tries: %s" % (model, retries + 1, last))


def fake_chat(prompt: str, model: str = "fake", **_kw: Any) -> Dict[str, Any]:
    """Deterministic stand-in (tests / --fake-model): judge prompts get a label, reader prompts an echo."""
    p = prompt
    if "CORRECT" in p and "WRONG" in p and "Gold answer:" in p:
        gold = p.split("Gold answer:")[1].split("\n")[0].strip().lower()
        gen = p.split("Generated answer:")[1].split("\n")[0].strip().lower()
        ok = bool(gold) and gold.strip(" .!") in gen
        return {"text": '{"label": "%s"}' % ("CORRECT" if ok else "WRONG"), "cost": 0.0, "cost_known": True,
                "model": model, "finish": "stop"}
    if "Is the model response correct?" in p or "Does the model correctly identify" in p:
        ans = p.split("Correct Answer:")[1].split("\n")[0].strip().lower() if "Correct Answer:" in p else ""
        resp = p.split("Model Response:")[1].split("\n")[0].strip().lower() if "Model Response:" in p else ""
        return {"text": "yes" if ans and ans in resp else "no", "cost": 0.0, "cost_known": True,
                "model": model, "finish": "stop"}
    if "RUBRIC CRITERION" in p:
        item = p.split("(what to check):")[1].split("\n")[0].lower()
        resp = p.split("RESPONSE TO EVALUATE):")[-1] if "RESPONSE TO EVALUATE):" in p else p.split("- RESPONSE TO EVALUATE:")[1].split("\n")[0]
        resp = resp.lower()
        sc = 1.0 if any(w in resp for w in item.split() if len(w) > 4) else 0.0
        return {"text": '{"score": %s, "reason": "fake"}' % sc, "cost": 0.0, "cost_known": True,
                "model": model, "finish": "stop"}
    q = p.rsplit("Question:", 1)[-1]
    ctx = p.split("Question:")[0]
    best = ""
    qw = {w for w in q.lower().split() if len(w) > 3}
    for line in ctx.splitlines():
        if len(line) > 10 and not line.startswith(("Below is", "Based on", "I will give")) and qw & set(line.lower().split()):
            best = line
            break
    return {"text": (best or "No information available")[:120], "cost": 0.0, "cost_known": True,
            "model": model, "finish": "stop"}


def get_chat(fake: bool) -> Callable[..., Dict[str, Any]]:
    return fake_chat if fake else chat
