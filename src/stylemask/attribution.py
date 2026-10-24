"""Authorship attributors — the "attackers" our rewriter has to beat.

Obfuscation that fools one attacker often fails against another, so we keep
several independent ones and only count a rewrite as successful when none
of them recognises the author:

- "ngram":  character + word n-grams, logistic regression (full text).
- "style":  same, but content words are masked first (text distortion,
            Stamatatos 2013) — sees punctuation, casing, filler words.
- "delta":  Burrows's Delta over the most frequent words — the classic
            method any forensic linguist would try first.
- "morph":  part-of-speech n-grams (pymorphy3) — sentence structure,
            invariant to synonym swaps.
"""
import re
from collections import Counter
from functools import lru_cache

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

_WORD = re.compile(r"\w+")
_TOKEN = re.compile(r"\w+|[^\w\s]")

KINDS = ("ngram", "style", "delta", "morph")


class StyleMasker(BaseEstimator, TransformerMixin):
    """Keep only words that are frequent for *every* author; mask the rest.

    Taking the intersection of per-author top-k lists drops names and
    chat-specific topics, which are frequent for one author only.
    """

    def __init__(self, top_k: int = 300):
        self.top_k = top_k

    def fit(self, X, y):
        per_author = {}
        for text, author in zip(X, y):
            per_author.setdefault(author, Counter()).update(w.lower() for w in _WORD.findall(text))
        tops = [{w for w, _ in c.most_common(self.top_k)} for c in per_author.values()]
        self.keep_ = set.intersection(*tops)
        return self

    def _mask(self, m: re.Match) -> str:
        w = m.group()
        if w.lower() in self.keep_:
            return w
        if w.isdigit():
            return "#" * len(w)
        return "".join("X" if ch.isupper() else "x" for ch in w)

    def transform(self, X):
        return [_WORD.sub(self._mask, text) for text in X]


class BurrowsDelta(BaseEstimator, ClassifierMixin):
    """Nearest author centroid in z-scored most-frequent-word space (Burrows, 2002)."""

    def __init__(self, n_words: int = 300):
        self.n_words = n_words

    def _freqs(self, X):
        counts = self.vec_.transform(X).toarray().astype(float)
        totals = np.array([max(1, len(_WORD.findall(t))) for t in X], dtype=float)[:, None]
        return counts / totals

    def fit(self, X, y):
        self.vec_ = CountVectorizer(token_pattern=r"(?u)\b\w+\b", max_features=self.n_words).fit(X)
        f = self._freqs(X)
        self.mean_, self.std_ = f.mean(0), f.std(0) + 1e-9
        z = (f - self.mean_) / self.std_
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.centroids_ = np.stack([z[y == c].mean(0) for c in self.classes_])
        return self

    def _delta(self, X):
        z = (self._freqs(X) - self.mean_) / self.std_
        return np.abs(z[:, None, :] - self.centroids_[None, :, :]).mean(2)

    def predict(self, X):
        return self.classes_[self._delta(X).argmin(1)]

    def predict_proba(self, X):
        # softmax over negative distances; temperature picked so typical gaps matter
        d = self._delta(X)
        e = np.exp(-(d - d.min(1, keepdims=True)) * 10)
        return e / e.sum(1, keepdims=True)


@lru_cache(maxsize=None)
def _morph():
    import pymorphy3
    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=200_000)
def _pos(token: str) -> str:
    if not token[0].isalnum():
        return token  # punctuation is kept as is
    if token.isdigit():
        return "NUM"
    tag = _morph().parse(token)[0].tag
    return str(tag.POS or ("LATN" if "LATN" in tag else "UNKN"))


def to_pos(text: str) -> str:
    """'короче я пошёл, ну' -> 'ADVB NPRO VERB , PRCL' (line breaks kept as NL)."""
    return " NL ".join(" ".join(_pos(t) for t in _TOKEN.findall(line)) for line in text.split("\n"))


class PosTagger(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [to_pos(t) for t in X]


def build_attributor(kind: str = "ngram"):
    if kind == "delta":
        return BurrowsDelta()
    if kind == "morph":
        return make_pipeline(
            PosTagger(),
            TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 3),
                            sublinear_tf=True, min_df=2, lowercase=False),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
    features = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True, min_df=2, lowercase=False)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
    ])
    steps = [StyleMasker()] if kind == "style" else []
    return make_pipeline(*steps, features, LogisticRegression(max_iter=2000, class_weight="balanced"))
