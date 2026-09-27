import pytest

from hybrid_rag.document import Document
from hybrid_rag.sparse import BM25Index


@pytest.fixture()
def corpus():
    return [
        Document("1", "The quick brown fox jumps over the lazy dog"),
        Document("2", "Python is a popular programming language for data science"),
        Document("3", "Reciprocal rank fusion combines multiple ranked lists"),
        Document("4", "The dog chased the fox through the forest"),
    ]


def test_search_returns_relevant_doc_first(corpus):
    index = BM25Index(corpus)
    results = index.search("fox and dog", top_k=4)
    result_ids = [r.doc_id for r in results]
    assert result_ids[0] in {"1", "4"}


def test_search_ranks_keyword_match_over_unrelated(corpus):
    index = BM25Index(corpus)
    results = index.search("programming language", top_k=4)
    assert results[0].doc_id == "2"


def test_search_respects_top_k(corpus):
    index = BM25Index(corpus)
    results = index.search("dog", top_k=1)
    assert len(results) <= 1


def test_empty_corpus_raises():
    with pytest.raises(ValueError):
        BM25Index([])


def test_no_match_returns_empty(corpus):
    index = BM25Index(corpus)
    results = index.search("zzznonexistenttermzzz", top_k=4)
    assert results == []
