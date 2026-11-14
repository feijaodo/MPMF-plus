"""b0 text metrics: LoCoMo token-F1 (official rules), ROUGE-L (LCS based), small BM25.

No third-party dependency. Differences from the official LoCoMo script (task_eval/evaluation.py):
that script Porter-stems tokens with nltk; nltk is not installed here, so tokens are NOT stemmed.
ROUGE-L here is the LCS-based F1 / recall / precision on whitespace tokens (the rouge package's
``rouge-l`` is the same LCS formulation); the MemGPT paper reports ROUGE-L *recall* for DMR.
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Dict, List, Sequence, Tuple

_TOK = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOK.findall((text or "").lower())


# ---- LoCoMo token F1 (mirrors evaluation.py normalisation; no stemming) ---------------------
def normalize_answer(s: str) -> str:
    s = (s or "").replace(",", "")
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    p = normalize_answer(prediction).split()
    g = normalize_answer(ground_truth).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def f1_multi(prediction: str, ground_truth: str) -> float:
    """Category 1 (multi-hop): comma-split sub-answers, best prediction part per gold part, averaged."""
    preds = [x.strip() for x in prediction.split(",")]
    golds = [x.strip() for x in ground_truth.split(",")]
    return sum(max(f1_score(p, g) for p in preds) for g in golds) / len(golds)


def locomo_f1(category: int, prediction: str, gold: str) -> float:
    """Official per-category token F1. Category 5 uses the official string rule (1/0)."""
    if category == 5:
        o = (prediction or "").lower()
        return 1.0 if ("no information available" in o or "not mentioned" in o) else 0.0
    if category == 3:
        gold = gold.split(";")[0].strip()
    if category == 1:
        return f1_multi(prediction, gold)
    return f1_score(prediction, gold)


# ---- ROUGE-L ---------------------------------------------------------------------------------
def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> Dict[str, float]:
    """LCS ROUGE-L on lower-cased alphanumeric tokens: {'p','r','f'}. 'r' is what the MemGPT paper reports."""
    p, r = tokenize(prediction), tokenize(reference)
    l = _lcs(p, r)
    if l == 0:
        return {"p": 0.0, "r": 0.0, "f": 0.0}
    pp, rr = l / len(p), l / len(r)
    return {"p": pp, "r": rr, "f": 2 * pp * rr / (pp + rr)}


# ---- BM25 (Okapi) ----------------------------------------------------------------------------
class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.n = len(docs)
        self.avg = (sum(self.len) / self.n) if self.n else 0.0
        df: Counter = Counter()
        for t in self.tf:
            df.update(t.keys())
        self.idf = {w: math.log(1.0 + (self.n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: str) -> List[float]:
        q = tokenize(query)
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * (ln / self.avg if self.avg else 0.0))
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out

    def top(self, query: str, k: int) -> List[Tuple[int, float]]:
        sc = self.scores(query)
        order = sorted(range(len(sc)), key=lambda i: (-sc[i], i))
        return [(i, sc[i]) for i in order[:k] if sc[i] > 0]
