"""BM25 sparse retrieval index, built on top of rank_bm25.

Uses BM25+ (Lv & Zhai, 2011) rather than the classic BM25/Okapi variant.
Okapi's IDF term, log((N - n + 0.5) / (n + 0.5)), goes to zero or negative
for a term appearing in roughly half (or more) of the documents -- fine for
web-scale corpora where that's rare, but a real problem for the small,
topically-narrow corpora this package is meant to index (a course's notes,
a team's docs), where a query's key terms often *do* appear in a large
fraction of the relevant documents. BM25+ adds a small positive floor so a
document is never scored down to zero purely for being "too common".
"""

from __future__ import annotations

from rank_bm25 import BM25Plus

from hybrid_rag.document import Document, ScoredDocument
from hybrid_rag.tokenize import tokenize


class BM25Index:
    def __init__(self, documents: list[Document]):
        if not documents:
            raise ValueError("BM25Index requires at least one document")
        self._documents = documents
        self._corpus_tokens = [tokenize(doc.text) for doc in documents]
        self._bm25 = BM25Plus(self._corpus_tokens)

    def search(self, query: str, top_k: int = 10) -> list[ScoredDocument]:
        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in ranked_idx:
            if scores[i] <= 0:
                continue
            doc = self._documents[i]
            results.append(
                ScoredDocument(doc_id=doc.doc_id, text=doc.text, score=float(scores[i]), metadata=doc.metadata)
            )
        return results

    def __len__(self) -> int:
        return len(self._documents)
