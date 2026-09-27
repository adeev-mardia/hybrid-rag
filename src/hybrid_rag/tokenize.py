"""Minimal, dependency-free tokenizer for BM25.

Not linguistically sophisticated on purpose -- BM25 cares about term overlap,
not morphology. Lowercases, strips punctuation, splits on whitespace, and
drops a small stopword list so overly common words don't dominate term
frequency scoring.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on for with as by at from is
    are was were be been being this that these those it its into about
    """.split()
)


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens
