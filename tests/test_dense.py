import numpy as np
import pytest

from hybrid_rag.document import Document
from hybrid_rag.dense import DenseIndex
from hybrid_rag.embedders import LsaEmbedder


@pytest.fixture()
def corpus():
    return [
        Document("1", "Cats are small domesticated carnivorous mammals kept as pets"),
        Document("2", "Dogs are loyal companion animals often trained to help humans"),
        Document("3", "Quantum computers use qubits to perform computation via superposition"),
        Document("4", "Kittens and puppies are baby cats and baby dogs respectively"),
    ]


def test_embeddings_are_unit_normalized(corpus):
    index = DenseIndex(corpus, embedder=LsaEmbedder(n_components=3))
    norms = np.linalg.norm(index._embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_pet_query_ranks_pet_docs_above_quantum(corpus):
    index = DenseIndex(corpus, embedder=LsaEmbedder(n_components=3))
    results = index.search("pets and animals", top_k=4)
    result_ids = [r.doc_id for r in results]
    assert result_ids.index("3") == len(result_ids) - 1  # quantum doc ranks last


def test_search_respects_top_k(corpus):
    index = DenseIndex(corpus, embedder=LsaEmbedder(n_components=3))
    results = index.search("cats", top_k=2)
    assert len(results) == 2


def test_empty_corpus_raises():
    with pytest.raises(ValueError):
        DenseIndex([], embedder=LsaEmbedder())


def test_encode_before_fit_raises():
    embedder = LsaEmbedder()
    with pytest.raises(RuntimeError):
        embedder.encode(["hello"])
