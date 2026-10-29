"""Explicit failure types — the package never fails silently.

A missing store, an unreadable config or a dead embedder must raise, not fall
back to another file or to an empty result. These three exceptions are the
whole contract.
"""


class MPMFPlusError(RuntimeError):
    """Base class for every package-level failure."""


class StoreNotFoundError(MPMFPlusError):
    """The configured store path does not exist or cannot be stat'ed."""


class ConfigError(MPMFPlusError):
    """The recall config file is missing, unreadable or invalid."""


class EmbedderError(MPMFPlusError):
    """The embedder is unreachable or returned no/garbage vector."""
