from hybrid_rag.document import ScoredDocument
from hybrid_rag.reranker import TfidfReranker


def test_rerank_empty_candidates_returns_empty():
    reranker = TfidfReranker()
    assert reranker.rerank("query", []) == []


def test_rerank_orders_by_relevance():
    candidates = [
        ScoredDocument("1", "The stock market fell sharply today amid inflation fears", 0.5),
        ScoredDocument("2", "Python is a versatile programming language used in data science", 0.4),
        ScoredDocument("3", "Programming in Python is popular for machine learning and data science work", 0.3),
    ]
    reranker = TfidfReranker()
    reranked = reranker.rerank("python programming for data science", candidates)
    assert reranked[0].doc_id in {"2", "3"}
    assert reranked[-1].doc_id == "1"


def test_rerank_respects_top_k():
    candidates = [
        ScoredDocument("1", "apple banana cherry", 0.1),
        ScoredDocument("2", "banana cherry date", 0.2),
        ScoredDocument("3", "cherry date elderberry", 0.3),
    ]
    reranker = TfidfReranker()
    reranked = reranker.rerank("banana", candidates, top_k=1)
    assert len(reranked) == 1
