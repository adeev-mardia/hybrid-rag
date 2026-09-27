"""Standard IR evaluation metrics for a single query's ranked results."""

from __future__ import annotations

import math

from hybrid_rag.document import ScoredDocument


def recall_at_k(results: list[ScoredDocument], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved = {r.doc_id for r in results[:k]}
    return len(retrieved & relevant_ids) / len(relevant_ids)


def precision_at_k(results: list[ScoredDocument], relevant_ids: set[str], k: int) -> float:
    top_k = results[:k]
    if not top_k:
        return 0.0
    retrieved = {r.doc_id for r in top_k}
    return len(retrieved & relevant_ids) / len(top_k)


def reciprocal_rank(results: list[ScoredDocument], relevant_ids: set[str]) -> float:
    for rank, r in enumerate(results, start=1):
        if r.doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: list[ScoredDocument], relevant_ids: set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    dcg = 0.0
    for rank, r in enumerate(results[:k], start=1):
        rel = 1.0 if r.doc_id in relevant_ids else 0.0
        dcg += rel / math.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
