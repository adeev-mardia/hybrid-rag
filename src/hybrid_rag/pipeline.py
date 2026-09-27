"""HybridRetriever: the top-level entry point combining BM25 + dense + RRF + optional reranking."""

from __future__ import annotations

from hybrid_rag.dense import DenseIndex
from hybrid_rag.document import Document, ScoredDocument
from hybrid_rag.embedders import Embedder, LsaEmbedder
from hybrid_rag.fusion import reciprocal_rank_fusion
from hybrid_rag.reranker import RerankerBackend, TfidfReranker
from hybrid_rag.sparse import BM25Index


class HybridRetriever:
    """Retrieves documents using BM25 + dense embeddings fused via RRF,
    with an optional second-stage reranking pass over the fused candidates.

    Example:
        >>> docs = [Document(doc_id="1", text="..."), ...]
        >>> retriever = HybridRetriever(docs)
        >>> retriever.search("what is reciprocal rank fusion?", top_k=5)

    By default this runs entirely locally with no network access or model
    download (TF-IDF+SVD for dense retrieval, TF-IDF cosine for reranking).
    Pass `embedder=SentenceTransformerEmbedder()` and/or
    `reranker=CrossEncoderReranker()` from `hybrid_rag.embedders` /
    `hybrid_rag.reranker` for stronger neural variants when you have model
    access.
    """

    def __init__(
        self,
        documents: list[Document],
        embedder: Embedder | None = None,
        reranker: RerankerBackend | None = None,
        use_reranker: bool = True,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ):
        self.documents = documents
        self.bm25_index = BM25Index(documents)
        self.dense_index = DenseIndex(documents, embedder=embedder or LsaEmbedder())
        self.reranker: RerankerBackend | None = (reranker or TfidfReranker()) if use_reranker else None
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        mode: str = "hybrid",
    ) -> list[ScoredDocument]:
        """Search the index.

        Args:
            query: Natural-language query.
            top_k: Number of results to return.
            candidate_k: How many candidates each retriever contributes before
                fusion/reranking (should be >= top_k, larger gives the
                reranker more to work with at some latency cost).
            mode: "hybrid" (BM25 + dense + RRF, default), "bm25" (sparse
                only), or "dense" (vector only) -- the latter two exist so
                the benchmark script can compare against hybrid.
        """
        if mode == "bm25":
            return self.bm25_index.search(query, top_k=top_k)
        if mode == "dense":
            return self.dense_index.search(query, top_k=top_k)
        if mode != "hybrid":
            raise ValueError(f"unknown mode {mode!r}, expected 'hybrid', 'bm25', or 'dense'")

        bm25_results = self.bm25_index.search(query, top_k=candidate_k)
        dense_results = self.dense_index.search(query, top_k=candidate_k)
        fused = reciprocal_rank_fusion(
            [bm25_results, dense_results],
            k=self.rrf_k,
            weights=[self.bm25_weight, self.dense_weight],
        )

        if self.reranker is not None and fused:
            return self.reranker.rerank(query, fused[:candidate_k], top_k=top_k)
        return fused[:top_k]
