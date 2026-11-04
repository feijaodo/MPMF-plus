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


def embed_text(text) -> str:
    """Return the canonical string to embed for a node whose text is `text`."""
    if not text:
        return ""
    return str(text)[:EMBED_CHARS]
