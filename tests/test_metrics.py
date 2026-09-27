from hybrid_rag.document import ScoredDocument
from hybrid_rag.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def results(*ids):
    return [ScoredDocument(doc_id=i, text="", score=1.0) for i in ids]


def test_recall_at_k_full_hit():
    assert recall_at_k(results("a", "b", "c"), {"a", "b"}, k=3) == 1.0


def test_recall_at_k_partial_hit():
    assert recall_at_k(results("a", "x", "y"), {"a", "b"}, k=3) == 0.5


def test_recall_at_k_no_relevant_docs_returns_zero():
    assert recall_at_k(results("a"), set(), k=3) == 0.0


def test_precision_at_k():
    assert precision_at_k(results("a", "x", "y"), {"a"}, k=3) == 1 / 3


def test_precision_at_k_empty_results():
    assert precision_at_k([], {"a"}, k=3) == 0.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(results("a", "b"), {"a"}) == 1.0


def test_reciprocal_rank_second_position():
    assert reciprocal_rank(results("x", "a"), {"a"}) == 0.5


def test_reciprocal_rank_not_found():
    assert reciprocal_rank(results("x", "y"), {"a"}) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(results("a", "b", "x"), {"a", "b"}, k=3) == 1.0


def test_ndcg_worse_ranking_scores_lower_than_perfect():
    perfect = ndcg_at_k(results("a", "b", "x"), {"a", "b"}, k=3)
    worse = ndcg_at_k(results("x", "a", "b"), {"a", "b"}, k=3)
    assert worse < perfect


def test_ndcg_no_relevant_docs_returns_zero():
    assert ndcg_at_k(results("a"), set(), k=3) == 0.0
