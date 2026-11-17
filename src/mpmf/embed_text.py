"""The ONE rule for "what string does a node's vector represent".

Every writer (ingest, the thinking capture, the legacy wrapper, the rebuild
tool) and every reader-side audit (the full scan, the daily health check) must
truncate identically, or a node's vector stops matching its own text and
cos(self) silently degrades. Before this module existed there were four rules:
512 chars in the thinking writer, head+tail 1200 in the wrapper, the full
sentence in ingest, 1000 then 512 in the audit — three of them wrong.

Rule: the first EMBED_CHARS characters of the text, head only. mxbai's window
is 512 TOKENS, so 512 characters never hits the model's own truncation and the
vector is reproducible byte-for-byte from (text, model).
"""

EMBED_CHARS = 512

#: wiki key that overrides what a node's vector represents (see embed_source)
EMBED_FROM_KEY = "embed_from"


def embed_text(text) -> str:
    """Return the canonical string to embed for a node whose text is `text`."""
    if not text:
        return ""
    return str(text)[:EMBED_CHARS]


def embed_source(text="", embed_from="") -> str:
    """The ONE rule: which string a node's vector represents.

    r106: ``embed_from or text``, truncated by embed_text(). The summary is NOT
    part of the rule. Measured on the live store (random 200 of 25,120, 2026-11-17):
    94.0% of vectors match the text and 62.5% match the summary -- the summary only
    "matches" when a node is short enough that summary == text[:120]. The r105
    summary-first rule would have re-pointed ~23,000 vectors to satisfy the 6% of
    nodes whose summary differs from their text.

    Everything that embeds or audits goes through this function, so the writers and
    the audits cannot hold different opinions about what a vector means.
    """
    for candidate in (embed_from, text):
        if candidate:
            return embed_text(str(candidate))
    return ""
