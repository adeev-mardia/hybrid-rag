"""Dense vector retrieval: pluggable embedder + brute-force cosine similarity.

Brute-force (rather than FAISS/HNSW) is intentional here: for corpora in the
thousands-to-low-tens-of-thousands range that a single-machine RAG demo
targets, exact cosine similarity over a numpy matrix is fast enough and
keeps the dependency footprint small. Swapping in an ANN index later only
requires changing `DenseIndex.search`.
"""

from __future__ import annotations

import numpy as np

from hybrid_rag.document import Document, ScoredDocument
from hybrid_rag.embedders import Embedder, LsaEmbedder


class DenseIndex:
    def __init__(self, documents: list[Document], embedder: Embedder | None = None):
        if not documents:
            raise ValueError("DenseIndex requires at least one document")
        self._documents = documents
        self.embedder = embedder or LsaEmbedder()
        self.embedder.fit([doc.text for doc in documents])
        self._embeddings: np.ndarray = self.embedder.encode([doc.text for doc in documents])

    def search(self, query: str, top_k: int = 10) -> list[ScoredDocument]:
        query_vec = self.embedder.encode([query])[0]
        # embeddings are L2-normalized, so dot product == cosine similarity
        scores = self._embeddings @ query_vec
        top_k = min(top_k, len(self._documents))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [
            ScoredDocument(
                doc_id=self._documents[i].doc_id,
                text=self._documents[i].text,
                score=float(scores[i]),
                metadata=self._documents[i].metadata,
            )
            for i in top_idx
        ]

    def __len__(self) -> int:
        return len(self._documents)
