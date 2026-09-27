# hybrid-rag

A retrieval library for RAG pipelines that combines BM25 (sparse/keyword) search with dense vector search, fused via Reciprocal Rank Fusion (RRF), with an optional second-stage reranking pass — plus a benchmark harness that actually measures whether hybrid retrieval beats either method alone.

## Why hybrid retrieval

Most "RAG in an afternoon" tutorials do vector-only retrieval: embed the corpus, embed the query, take top-k by cosine similarity. That fails in two common, boring ways:

- **Exact terms get lost.** A query containing a specific identifier, acronym, or rare proper noun (an error code, a person's name, a product SKU) often retrieves worse with pure embeddings than with plain keyword search, because embedding models are trained for semantic similarity, not exact lexical match.
- **Paraphrases get missed by keyword search.** BM25 can't tell that "why can't python run threads in parallel" and "the GIL prevents concurrent bytecode execution" are about the same thing if they share few surface words.

Fusing both rankers with RRF gets the benefits of each without needing to normalize or calibrate their very different score scales against each other — it only looks at rank position (Cormack et al., 2009).

## Results

`scripts/benchmark.py` runs all three modes against a hand-built, hard-to-game evaluation set (40 documents across 8 topics, 20 queries with graded relevance judgments — see `src/hybrid_rag/eval_data.py`) and reports:

```
mode       recall@5     nDCG@5       MRR
---------------------------------------------
bm25       0.975        0.981        1.000
dense      1.000        0.948        0.925
hybrid     1.000        0.996        1.000
```

Hybrid retrieval matches or beats both individual methods on every metric — most notably nDCG@5, where it improves ranking quality (not just whether the right doc is retrieved at all, but how high it lands) over both single-method baselines. Reproduce with `python scripts/benchmark.py`.

## Install

```bash
git clone https://github.com/adeev-mardia/hybrid-rag.git
cd hybrid-rag
pip install -e ".[dev]"
```

Requires Python 3.10+. Core install has **no GPU or network dependency** — dense retrieval defaults to a locally-trained TF-IDF+SVD embedder (see below), so `pip install` + run works fully offline.

## Quickstart

```python
from hybrid_rag.document import Document
from hybrid_rag.pipeline import HybridRetriever

docs = [
    Document(doc_id="1", text="Python's GIL prevents multiple native threads from running bytecode simultaneously."),
    Document(doc_id="2", text="Rust's borrow checker enforces memory safety at compile time."),
    # ...
]

retriever = HybridRetriever(docs)
results = retriever.search("why can't python threads run in parallel", top_k=5)
for r in results:
    print(r.score, r.doc_id, r.text[:60])
```

Or run the bundled demo: `python scripts/demo.py`.

## Architecture

```
src/hybrid_rag/
├── document.py     # Document / ScoredDocument data types
├── tokenize.py      # lightweight BM25 tokenizer
├── sparse.py         # BM25Index (rank_bm25's BM25+ variant)
├── embedders.py     # pluggable Embedder protocol: LsaEmbedder (default) / SentenceTransformerEmbedder
├── dense.py           # DenseIndex: brute-force cosine similarity over an Embedder's vectors
├── fusion.py           # reciprocal_rank_fusion()
├── reranker.py           # pluggable reranker: TfidfReranker (default) / CrossEncoderReranker
├── pipeline.py             # HybridRetriever: wires all of the above together
├── metrics.py               # recall@k, precision@k, MRR, nDCG@k
└── eval_data.py               # the hand-built benchmark corpus/queries/qrels
```

### Pluggable dense retrieval — and why the default needs no network access

`DenseIndex` and `Reranker` depend on small protocols (`Embedder`, `RerankerBackend`), not on a specific library:

- **Default (`LsaEmbedder`, `TfidfReranker`):** TF-IDF + truncated SVD (classic Latent Semantic Analysis, Deerwester et al. 1990) trained on the indexed corpus in milliseconds, entirely offline, with `scikit-learn` as the only dependency. This is a real, historically important dense-retrieval technique — not a stand-in — though it captures term co-occurrence structure rather than deep semantics, and has no notion of word order.
- **Optional (`SentenceTransformerEmbedder`, `CrossEncoderReranker`):** wraps pretrained transformer bi-/cross-encoders (e.g. `all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`) via `sentence-transformers` for genuine learned semantic similarity and joint query-document scoring. Install with `pip install -e ".[neural]"` and pass them explicitly:

```python
from hybrid_rag.embedders import SentenceTransformerEmbedder
from hybrid_rag.reranker import CrossEncoderReranker

retriever = HybridRetriever(
    docs,
    embedder=SentenceTransformerEmbedder(),
    reranker=CrossEncoderReranker(),
)
```

This requires network access on first run to download model weights (or a pre-populated local Hugging Face cache).

## How RRF fusion works

Each retriever (BM25, dense) returns its own ranked list. Instead of trying to make a BM25 score and a cosine similarity comparable (they live on completely different scales), RRF scores each document only by *where it ranked*:

```
score(doc) = Σ over retrievers r where doc appears in r's results:  1 / (k + rank_r(doc))
```

`k=60` (the constant from the original paper) dampens the difference between, say, rank 1 and rank 2, so a document that ranks moderately well in *both* lists can outscore a document that ranks first in only one — which is usually what you want from a fused signal. See `fusion.py` and `tests/test_fusion.py` (which specifically tests that cross-retriever agreement can beat a single first-place finish).

## Testing

```bash
pip install -e ".[dev]"
pytest -v
```

25 tests across tokenization, BM25, dense retrieval, fusion, reranking, the end-to-end pipeline, and the evaluation metrics themselves (recall@k, nDCG@k, MRR each have dedicated correctness tests, not just smoke tests).

## Design notes

- **BM25+ over classic Okapi.** Okapi's IDF term can go to zero (or negative, clamped away) for a term appearing in roughly half a corpus — a non-issue at web scale, but common in small, topically-narrow corpora (a course's notes, a team's internal docs) that this library is realistically aimed at. BM25+ (Lv & Zhai, 2011) adds a positive floor so a document is never zeroed out purely for containing a common-but-relevant term. Covered by `tests/test_sparse.py`.
- **Brute-force cosine similarity, not an ANN index.** For corpora in the thousands range, exact cosine similarity over a numpy matrix is fast enough and keeps the dependency surface small. `DenseIndex.search` is the only place that would need to change to plug in FAISS/HNSW for larger corpora.
- **RRF over score-normalization fusion.** Normalizing BM25 and cosine scores onto a shared scale (e.g. min-max) is brittle — it's sensitive to outliers and to how many candidates each retriever returns. RRF sidesteps that by only using rank position.

## License

MIT
