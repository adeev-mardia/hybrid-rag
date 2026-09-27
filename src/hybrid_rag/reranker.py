"""Second-stage reranking: a more precise (and more expensive) re-scoring of
a small candidate set, ideally jointly modeling the (query, document) pair
rather than encoding each independently -- typically a meaningful precision
boost over the first-stage retriever's coarse ranking. Not usable for
first-stage retrieval over a whole corpus since it doesn't produce an
indexable vector (or, for the cross-encoder, is O(candidates) per query).

Two implementations, same tradeoff as `embedders.py`:

- `TfidfReranker` (default): fits a fresh, un-reduced TF-IDF vectorizer over
  just the candidate set for each query and scores cosine similarity against
  it. No download, runs anywhere. Sharper than the SVD-reduced retrieval
  score because it isn't compressed to `n_components` dimensions, but it's
  still a lexical (term-overlap) signal, not a learned relevance model.
- `CrossEncoderReranker`: wraps a pretrained cross-encoder (e.g.
  ms-marco-MiniLM-L-6-v2) that was trained on human relevance judgments and
  scores query/document pairs jointly through a transformer -- the
  state-of-the-art approach, but requires network access to download
  weights on first use.
"""

from __future__ import annotations

from typing import Protocol

from hybrid_rag.document import ScoredDocument


class RerankerBackend(Protocol):
    def rerank(self, query: str, candidates: list[ScoredDocument], top_k: int | None = None) -> list[ScoredDocument]: ...


class TfidfReranker:
    def rerank(self, query: str, candidates: list[ScoredDocument], top_k: int | None = None) -> list[ScoredDocument]:
        if not candidates:
            return []
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        texts = [c.text for c in candidates]
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
        # fit on candidates + query jointly so the query's terms are in vocabulary
        matrix = vectorizer.fit_transform(texts + [query])
        doc_vectors, query_vector = matrix[:-1], matrix[-1]
        sims = cosine_similarity(doc_vectors, query_vector).ravel()

        reranked = [
            ScoredDocument(doc_id=c.doc_id, text=c.text, score=float(s), metadata=c.metadata)
            for c, s in zip(candidates, sims)
        ]
        reranked.sort(key=lambda d: d.score, reverse=True)
        return reranked[:top_k] if top_k is not None else reranked


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL, model=None):
        self.model_name = model_name
        self._model = model

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[ScoredDocument], top_k: int | None = None) -> list[ScoredDocument]:
        if not candidates:
            return []
        model = self._ensure_model()
        pairs = [(query, c.text) for c in candidates]
        scores = model.predict(pairs)
        reranked = [
            ScoredDocument(doc_id=c.doc_id, text=c.text, score=float(s), metadata=c.metadata)
            for c, s in zip(candidates, scores)
        ]
        reranked.sort(key=lambda d: d.score, reverse=True)
        return reranked[:top_k] if top_k is not None else reranked


# Backwards/simple-friendly default export
Reranker = TfidfReranker
