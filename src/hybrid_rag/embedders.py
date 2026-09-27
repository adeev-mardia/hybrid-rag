"""Pluggable text-embedding backends.

The dense retrieval layer depends only on the `Embedder` protocol below, not
on a specific model or library. Two implementations are provided:

- `LsaEmbedder` (default): TF-IDF + truncated SVD (i.e. classic Latent
  Semantic Analysis). Trains entirely on the corpus being indexed, offline,
  in milliseconds, with no model download and no GPU. This is what the
  package uses out of the box so `pip install` + run works anywhere.
- `SentenceTransformerEmbedder`: wraps a pretrained bi-encoder (e.g.
  all-MiniLM-L6-v2) from the `sentence-transformers` library for
  meaning-level semantic similarity beyond term co-occurrence. Requires
  network access to download model weights the first time; pass an
  already-loaded model to avoid that.

Both produce L2-normalized float32 vectors so cosine similarity reduces to a
dot product, which is what `DenseIndex` assumes.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Anything that can turn a batch of strings into a matrix of unit vectors."""

    def fit(self, corpus: list[str]) -> None: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...

    @property
    def dim(self) -> int: ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class LsaEmbedder:
    """TF-IDF + truncated SVD embedder. No external downloads, no GPU.

    This is a real, historically important dense-retrieval technique
    (Deerwester et al., 1990) -- not a toy stand-in. It captures term
    co-occurrence structure (so queries and documents sharing no exact
    words but similar topics still score well) but, unlike a transformer
    encoder, has no notion of word order or deeper semantics.
    """

    def __init__(self, n_components: int = 128, random_state: int = 42):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.n_components = n_components
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50_000,
            sublinear_tf=True,
        )
        self._svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        n_components = min(self.n_components, max(1, len(corpus) - 1))
        if n_components != self._svd.n_components:
            from sklearn.decomposition import TruncatedSVD

            self._svd = TruncatedSVD(n_components=n_components, random_state=self._svd.random_state)
        tfidf = self._vectorizer.fit_transform(corpus)
        self._svd.fit(tfidf)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LsaEmbedder.fit(corpus) must be called before encode()")
        tfidf = self._vectorizer.transform(texts)
        reduced = self._svd.transform(tfidf)
        return _l2_normalize(reduced).astype(np.float32)

    @property
    def dim(self) -> int:
        return self._svd.n_components


DEFAULT_ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    """Wraps a pretrained sentence-transformers bi-encoder.

    Requires the `sentence-transformers` package and, on first use, network
    access to download the model weights (or a pre-populated local HF cache
    / `model=` object passed in). See README for offline-mount instructions.
    """

    def __init__(self, model_name: str = DEFAULT_ST_MODEL_NAME, model=None):
        self.model_name = model_name
        self._model = model  # lazily constructed on fit() if not provided
        self._dim = None

    def fit(self, corpus: list[str]) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("SentenceTransformerEmbedder.fit(corpus) must be called before encode()")
        embeddings = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("call fit() first")
        return self._dim
