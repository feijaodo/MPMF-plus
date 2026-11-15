"""r99: the time-question predicate must fire on ORDERING / DATE cues only.

Context: the r98 predicate carried a bare ``time`` alternative, and every DMR question
opens with "Hey, remember that time we talked about ...". Measured on the bench pools:
r98 fired on 130/130 DMR questions (history REPLACED the facts block on all of them),
294/321 LoCoMo temporal and 73/80 BEAM event_ordering+temporal_reasoning. The cues kept
below fire on 21/130, 320/321 and 79/80 respectively.

The corpus-count test reads the bench datasets only when $MPMF_BENCH_DATA points at them
(they are not part of the repo), so the string cases are the portable guard.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from mpmf_plus import block as B            # noqa: E402
from mpmf_plus import history as H         # noqa: E402
from mpmf_plus import paths                # noqa: E402

# --------------------------------------------------------------------------- must NOT fire
NOT_TEMPORAL = [
    # the DMR opener: "time" is a noun here, not a question about when
    "Hey, remember that time we talked about our pets? What is the name of your dog again?",
    "Hey, remember that one time we talked about our jobs? What was it you said you did for a living?",
    "Hey, remember that time we talked about our personal life changes? What was that place you mentioned?",
    # the four phrasings named in the brief
    "Last time we spoke about my surgery, what did you say you would do about it?",
    "This time, remind me what I said about my dogs.",
    "Every time we talk about my farm, what do I mention?",
    "It is time to talk about my favorite food.",
    # state-change words say something changed, not when
    "What do I still do on weekends?",
    "Do I keep in touch with my old roommate anymore?",
    "How did my morning routine progress?",
    "Tell me the history of my project.",
    "What did I use to say about my commute?",
    # "for the first time" / ordinal occurrence, not an ordering question
    "For the first time, list every tool I own.",
    # plain non-temporal asks that contain no cue at all
    "What is the name of my dog?",
    "Which city was I born in?",
]

# --------------------------------------------------------------------------- must fire
TEMPORAL = [
    # question word asking for a date
    "When did Caroline go to the LGBTQ support group?",
    "What time did I leave the party?",
    # calendar value
    "What year did John start surfing?",
    "Which month was John in Italy?",
    "What day did Tim get into his study abroad program?",
    "In 2019, where did I work?",
    "What did I do in March?",
    # duration / gap
    "How long has Melanie been practicing art?",
    "How many weeks passed between Maria adopting Coco and Shadow?",
    "How many days were there between the end of my first sprint and the deadline?",
    # explicit ordering frame
    "Can you list the order in which I brought up different aspects of my budget tracker?",
    "Walk me through the timeline of my app development.",
    "Was the deploy before or after the migration?",
    # rank / recency
    "Which pet did Jolene adopt more recently - Susie or Seraphim?",
    "What did I say earlier about the budget?",
]


@pytest.mark.parametrize("q", NOT_TEMPORAL)
def test_not_temporal(q):
    assert H.is_temporal(q) is False


@pytest.mark.parametrize("q", TEMPORAL)
def test_temporal(q):
    assert H.is_temporal(q) is True


def test_empty_query_is_not_temporal():
    assert H.is_temporal("") is False
    assert H.is_temporal(None) is False


def test_last_and_first_time_do_not_fire_through_last_first():
    """The (?!\\s+time\\b) guard: "last" and "first" are cues unless the noun is "time"."""
    assert H.is_temporal("What did I say last?") is True
    assert H.is_temporal("What was the first thing I said?") is True
    assert H.is_temporal("What did I say last time?") is False
    assert H.is_temporal("What was the first time I visited?") is False


# --------------------------------------------------------------------------- defaults
def test_history_additive_is_the_default():
    assert paths.DEFAULTS["history_additive"] is True
    o = B.injection_opts(dict(paths.DEFAULTS))
    assert o["history_additive"] is True


def test_history_additive_explicit_false_still_holds():
    o = B.injection_opts({"history_additive": False})
    assert o["history_additive"] is False
    o = B.injection_opts({"history_additive": "0"})
    assert o["history_additive"] is False


# --------------------------------------------------------------------------- corpus counts
DATA = os.environ.get("MPMF_BENCH_DATA", "")


@pytest.mark.skipif(not (DATA and os.path.isdir(DATA)), reason="bench datasets not present ($MPMF_BENCH_DATA)")
def test_fire_counts_on_the_bench_pools():
    from mpmf_plus.bench import datasets as D
    _, locomo = D.load("locomo", DATA)
    _, dmr = D.load("dmr", DATA)
    _, beam = D.load("beam", DATA)
    lt = [q for q in locomo if q.category == "temporal"]
    be = [q for q in beam if q.category in ("event_ordering", "temporal_reasoning")]
    n_dmr = sum(1 for q in dmr if H.is_temporal(q.question))
    n_loc = sum(1 for q in lt if H.is_temporal(q.question))
    n_beam = sum(1 for q in be if H.is_temporal(q.question))
    assert n_dmr <= len(dmr) * 0.20, "DMR must fire on few questions, got %d/%d" % (n_dmr, len(dmr))
    assert n_loc >= 0.97 * len(lt), "LoCoMo temporal must fire on most, got %d/%d" % (n_loc, len(lt))
    assert n_beam >= 0.95 * len(be), "BEAM ordering/temporal must fire on most, got %d/%d" % (n_beam, len(be))
