import pytest

from hybrid_rag.document import Document
from hybrid_rag.embedders import LsaEmbedder
from hybrid_rag.pipeline import HybridRetriever


@pytest.fixture()
def corpus():
    return [
        Document("1", "The Eiffel Tower is a wrought-iron lattice tower in Paris, France"),
        Document("2", "Reciprocal rank fusion merges ranked lists using reciprocal of rank position"),
        Document("3", "BM25 is a bag-of-words ranking function used by search engines"),
        Document("4", "The Great Wall of China stretches thousands of kilometers"),
        Document("5", "Cross-encoders jointly score a query and document pair for reranking"),
        Document("6", "Paris is the capital city of France and home to many landmarks"),
    ]


@pytest.fixture()
def retriever(corpus):
    return HybridRetriever(corpus, embedder=LsaEmbedder(n_components=4), use_reranker=True)


def test_hybrid_search_returns_results(retriever):
    results = retriever.search("what is BM25 used for", top_k=3)
    assert len(results) > 0
    assert results[0].doc_id == "3"


def test_hybrid_search_finds_paris_docs(retriever):
    results = retriever.search("landmarks in Paris France", top_k=3)
    result_ids = {r.doc_id for r in results}
    assert "1" in result_ids or "6" in result_ids


def test_mode_bm25_only(retriever):
    results = retriever.search("reciprocal rank fusion", top_k=3, mode="bm25")
    assert results[0].doc_id == "2"


def test_mode_dense_only(retriever):
    results = retriever.search("reciprocal rank fusion merging lists", top_k=3, mode="dense")
    assert len(results) > 0


def test_invalid_mode_raises(retriever):
    with pytest.raises(ValueError):
        retriever.search("query", mode="nonsense")


def test_no_reranker_still_returns_fused_results(corpus):
    retriever = HybridRetriever(corpus, embedder=LsaEmbedder(n_components=4), use_reranker=False)
    results = retriever.search("BM25 ranking function", top_k=3)
    assert len(results) > 0
    assert retriever.reranker is None
