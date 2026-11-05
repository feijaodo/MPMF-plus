"""Path + config resolution for MPMF+. The only place that knows where things live.

Order: explicit env -> MPMF_HOME -> ~/.mpmf. Never hard-codes a host path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

DEFAULTS: Dict[str, Any] = {
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


def ensure_dirs() -> None:
    for d in (home(), health_dir()):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
