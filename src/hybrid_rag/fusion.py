"""Reciprocal Rank Fusion (RRF) for combining multiple ranked result lists.

RRF is the standard way to merge heterogeneous rankers (e.g. BM25 term
scores and cosine similarity scores, which live on completely different
scales) without needing to normalize or calibrate their scores against each
other -- it only looks at rank position. See Cormack et al., 2009,
"Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning
Methods".
"""

from __future__ import annotations

from collections import defaultdict

from hybrid_rag.document import ScoredDocument


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredDocument]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[ScoredDocument]:
    """Fuse multiple ranked lists into one, by summed reciprocal rank.

    Args:
        ranked_lists: One ranked list of ScoredDocument per retriever, each
            already sorted best-first.
        k: RRF damping constant (60 is the value used in the original paper
            and in most production hybrid-search systems; higher values
            flatten the influence of top ranks).
        weights: Optional per-list weight, same length as ranked_lists.
            Defaults to equal weighting.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must have the same length as ranked_lists")

    fused_scores: dict[str, float] = defaultdict(float)
    doc_lookup: dict[str, ScoredDocument] = {}

    for weight, ranked_list in zip(weights, ranked_lists):
        for rank, scored_doc in enumerate(ranked_list, start=1):
            fused_scores[scored_doc.doc_id] += weight * (1.0 / (k + rank))
            # first retriever to mention a doc wins the metadata/text snapshot
            doc_lookup.setdefault(scored_doc.doc_id, scored_doc)

    fused = [
        ScoredDocument(
            doc_id=doc_id,
            text=doc_lookup[doc_id].text,
            score=score,
            metadata=doc_lookup[doc_id].metadata,
        )
        for doc_id, score in fused_scores.items()
    ]
    fused.sort(key=lambda d: d.score, reverse=True)
    return fused
