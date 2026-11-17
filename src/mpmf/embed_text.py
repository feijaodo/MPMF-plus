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


def embed_source(text="", summary="", embed_from="") -> str:
    """The ONE rule: which string a node's vector represents.

    r105: `embed_from` wins, then the summary, then the text -- the precedence
    ``memory.remember()`` has always used. The winner is then truncated by
    embed_text(), so a node built by the ingest path, by the cog/enrichment
    writers and by set_node_* all produce the same bytes for the same node.

    Why this exists: the r103 audit compared every vector against
    embed_text(text), so it indicted 1,783 summary-embedded nodes as "drifted"
    when their vectors matched their own summaries exactly (cos 1.0000). A rule
    that lives in one function cannot disagree with itself.
    """
    for candidate in (embed_from, summary, text):
        if candidate:
            return embed_text(str(candidate))
    return ""
