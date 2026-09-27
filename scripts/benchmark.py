#!/usr/bin/env python3
"""Compare BM25-only, dense-only, and hybrid (BM25 + dense + RRF + rerank)
retrieval on the bundled evaluation set, and print a results table.

Usage:
    python scripts/benchmark.py
"""

from __future__ import annotations

from hybrid_rag.embedders import LsaEmbedder
from hybrid_rag.eval_data import load_corpus, load_queries
from hybrid_rag.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from hybrid_rag.pipeline import HybridRetriever

K = 5


def evaluate(retriever: HybridRetriever, queries: list[dict], mode: str) -> dict:
    recalls, ndcgs, rrs = [], [], []
    for q in queries:
        results = retriever.search(q["text"], top_k=K, candidate_k=15, mode=mode)
        recalls.append(recall_at_k(results, q["relevant_doc_ids"], k=K))
        ndcgs.append(ndcg_at_k(results, q["relevant_doc_ids"], k=K))
        rrs.append(reciprocal_rank(results, q["relevant_doc_ids"]))
    n = len(queries)
    return {
        "mode": mode,
        f"recall@{K}": sum(recalls) / n,
        f"nDCG@{K}": sum(ndcgs) / n,
        "MRR": sum(rrs) / n,
    }


def main() -> None:
    corpus = load_corpus()
    queries = load_queries()

    retriever = HybridRetriever(corpus, embedder=LsaEmbedder(n_components=32), use_reranker=True)

    rows = [
        evaluate(retriever, queries, mode="bm25"),
        evaluate(retriever, queries, mode="dense"),
        evaluate(retriever, queries, mode="hybrid"),
    ]

    header = f"{'mode':<10} {'recall@' + str(K):<12} {'nDCG@' + str(K):<12} {'MRR':<8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['mode']:<10} "
            f"{row[f'recall@{K}']:<12.3f} "
            f"{row[f'nDCG@{K}']:<12.3f} "
            f"{row['MRR']:<8.3f}"
        )

    print(f"\n{len(corpus)} documents, {len(queries)} queries, top-{K} cutoff.")


if __name__ == "__main__":
    main()
