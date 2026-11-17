"""r102: thinking/lessons sections never cut mid-text — whole items are dropped instead."""
from mpmf_plus import rendering as R, tagged as T


def _lessons(n, width=120):
    lines = ["Prior reasoning lessons:"]
    for i in range(1, n + 1):
        lines.append("%d. [status] lesson %d %s" % (i, i, "x" * width))
    return "\n".join(lines)


def test_tagged_lessons_drop_whole_items_never_cut():
    out = T.lessons(_lessons(6), cap=260, framing="Framing line.")
    assert "…" not in out
    assert out.count("<lesson n=") == out.count("</lesson>")
    # every kept lesson is a complete original line (full x-run)
    for chunk in out.split("<lesson n=")[1:]:
        body = chunk.split(">", 1)[1].split("</lesson>")[0]
        assert body.rstrip().endswith("x" * 20)


def test_tagged_prior_reasoning_full_text_and_whole_items():
    res = [{"context": "c" * 150 + " END", "conclusion": "k" * 250 + " DONE", "outcome": "success", "score": 0.9}] * 3
    out = T.prior_reasoning(res, cap=1200)
    assert "…" not in out
    for chunk in out.split("<trace")[1:]:
        assert "END" in chunk and "DONE" in chunk


def test_legacy_thinking_block_stops_at_the_cap():
    """Legacy block is byte-frozen (pre-r1 golden): it may slice ONE entry, but it must
    then STOP. The r103 regression dropped the `break`, so the truncated entry was
    followed by the full one and the cap was exceeded."""
    ctx = ("long context " + "y" * 400).strip()
    res = [{"context": ctx, "conclusion": "done.", "outcome": "success"} for _ in range(3)]
    entry = "- **%s** → done. *(outcome: success)*" % ctx[:140]

    # 300: entry 1 fits whole, entry 2 is cut once and then the block STOPS
    out = R.thinking_block(res, max_chars=300)
    lines_ = out.split("\n")
    assert lines_[-1] == "</thinking-context>"
    assert lines_[-2] == entry[:300 - len(entry)] + "…"
    assert out.count("…") == 1                       # one cut, never two
    body = [l for l in lines_ if l.startswith("- **")]
    assert len(body) == 2 and body[0] == entry       # whole entries first
    assert body[-1].endswith("…")                    # then at most one cut tail

    # 200: not even a marker's worth of room -> one WHOLE entry, no cut, no overrun
    out = R.thinking_block(res, max_chars=200)
    assert "…" not in out
    assert [l for l in out.splitlines() if l.startswith("- **")] == [entry]


# ---------------------------------------------------------------- r103: cognition block
def _cog(n, width=120, header=True, cont=False):
    parts = ["<prior-reasoning>"]
    if header:
        parts.append("The following lessons were distilled from prior sessions.")
    for i in range(1, n + 1):
        parts.append("%d. [status] lesson %d %s" % (i, i, "x" * width))
        if cont:
            parts.append("   continued tail of lesson %d zzzz" % i)
    parts.append("</prior-reasoning>")
    return "\n".join(parts)


def test_cognition_truncation_drops_whole_items_never_slices():
    """r103: the cap used to be a raw character slice, so a block could end mid-item
    ("## 1. Ov"). Every kept lesson must now be a complete original line."""
    out = R.truncate_cognition_block(_cog(6, 120), 400)
    assert "…" not in out
    assert out.endswith("</prior-reasoning>")
    kept = [ln for ln in out.splitlines() if ln[:1].isdigit()]
    assert kept, "at least one whole item must survive"
    for ln in kept:
        assert ln.endswith("x" * 120), ln[-40:]
    assert "truncated" in out


def test_cognition_truncation_never_ends_mid_item_at_any_cap():
    block = _cog(4, 200)
    for cap in (500, 600, 700, 800, 950, 1100):
        out = R.cognition_block(block, cap)
        if not out:
            continue
        for ln in out.splitlines():
            if ln[:1].isdigit():
                assert ln.endswith("x" * 200), "cut mid-item at cap=%d: %r" % (cap, ln[-40:])


def test_cognition_continuation_lines_drop_with_their_item():
    out = R.truncate_cognition_block(_cog(3, 120, cont=True), 400)
    for ln in out.splitlines():
        if "continued tail" in ln:
            n = ln.split("lesson ")[1].split()[0]
            assert any(l.startswith(n + ". ") for l in out.splitlines()), ln


def test_cognition_is_empty_when_not_even_one_item_fits():
    """A header with no lessons is noise; a cut lesson is misinformation."""
    assert R.truncate_cognition_block(_cog(3, 200), 120) == ""
    assert R.cognition_block(_cog(3, 200), 120) == ""


def test_cognition_short_block_is_untouched():
    block = _cog(1, 10)
    assert R.truncate_cognition_block(block, 5000) == block


def test_small_item_still_fits():
    out = R.thinking_block([{"context": "short ctx", "conclusion": "ok."}], max_chars=800)
    assert "short ctx" in out and "ok." in out
