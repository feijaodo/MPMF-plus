"""Path + config resolution for MPMF+. The only place that knows where things live.

Order: explicit env -> MPMF_HOME -> ~/.mpmf. Never hard-codes a host path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

DEFAULTS: Dict[str, Any] = {
    # f2: the user's own first name(s), from LOCAL config only (never in code). Used as
    # history stop-words and in the extraction-prompt example. Default: none.
    "user_names": [],
    # recall
    "ns_prefilter": True,
    "cognition_rank": "lexical",
    "cognition_floor": 0.0,
    "prefetch_budget_ms": 6500,
    "rerank_skip_after_ms": 0,
    "embed_query_cap_chars": 1200,
    "cognition_min_cos": 0.65,
    "prewarm_store": True,
    "prewarm_query": False,
    "prewarm_warm_query": True,
    "memory_arm": "d20",
    "memory_render": "full",
    "rerank_clamp_chars": 0,
    # p7: let the proposer apply a TUNE+HOLDOUT winner to a recall_config registry key (default OFF)
    "auto_apply": False,
    # p8: raw user messages as their own searchable layer, fused into prefetch (default OFF, RESEARCH)
    "raw_layer": False,
    "raw_k": 3,
    "raw_min_cos": 0.5,
    "raw_rerank": True,
    "raw_item_chars": 500,
    "raw_block_chars": 1500,
    # p9: re-ask the extractor once when a fact drops a name/number/date of its evidence (default OFF)
    "keep_details": False,
    # p11: show a lesson right before a completion claim lacking a tool call / numbered items (default OFF)
    "lessons_at_action": False,
    # p12: 1-hop walk over IS_A / DERIVED_FROM / supersedes edges from the top hits (default OFF, RESEARCH)
    "typed_walk": False,
    "typed_walk_k": 2,
    "typed_walk_seeds": 4,
    "typed_walk_min_cos": 0.3,
    # injection
    "inject_thinking": True,
    "thinking_top_k": 3,
    "thinking_max_chars": 800,
    "inject_cognition": True,
    "cognition_top_k": 3,
    "cognition_max_chars": 700,
    # storage
    "top_k": 8,
    "max_chars": 3500,
    "auto_write": True,
    "turn_namespace": "transcript",
}


def home() -> Path:
    """MPMF_HOME, defaulting to ~/.mpmf (never a literal absolute home)."""
    return Path(os.environ.get("MPMF_HOME") or (Path.home() / ".mpmf")).expanduser()


def config_path() -> Path:
    return Path(os.environ.get("MPMF_RECALL_CONFIG") or (home() / "recall.config.json"))


def store_path() -> Path:
    return Path(os.environ.get("MPMF_STORE_PATH") or (home() / "athena.mpmf"))


def thinking_store() -> Path:
    return Path(os.environ.get("MPMF_THINKING_STORE") or (home() / "athena-thinking.mpmf"))


def health_dir() -> Path:
    return Path(os.environ.get("MPMF_HEALTH_DIR") or (home() / "health"))


def log_path(name: str) -> Path:
    return health_dir() / name


def load_config(path=None) -> Dict[str, Any]:
    """Defaults <- file <- env overrides (MPMF_<KEY>)."""
    cfg = dict(DEFAULTS)
    p = Path(path) if path else config_path()
    try:
        with open(p) as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            cfg.update(data)
    except Exception:
        pass
    for key in list(cfg):
        env = os.environ.get("MPMF_" + key.upper())
        if env is not None:
            try:
                cfg[key] = json.loads(env)
            except Exception:
                cfg[key] = env
    return cfg


def user_names(cfg: Any = None) -> list:
    """f2: first name(s) from the ``user_names`` config key (list of str; default [])."""
    if cfg is None:
        cfg = load_config()
    v = cfg.get("user_names") if isinstance(cfg, dict) else None
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, (list, tuple)):
        return []
    return [s.strip() for s in v if isinstance(s, str) and s.strip()]


def personalize_extraction_prompt(prompt: str, cfg: Any = None) -> str:
    """f2: the planes-prompt example names the first configured user, else 'the user'."""
    names = user_names(cfg)
    if not names:
        return prompt
    n = names[0]
    return (prompt.replace("The user is direct,", n + " is direct,")
                  .replace('"name": "user"', '"name": "%s"' % n))


def ensure_dirs() -> None:
    for d in (home(), health_dir()):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
