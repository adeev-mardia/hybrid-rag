from hybrid_rag.document import ScoredDocument
from hybrid_rag.fusion import reciprocal_rank_fusion


def sd(doc_id, score, text="text"):
    return ScoredDocument(doc_id=doc_id, text=text, score=score)


def test_single_list_preserves_order():
    ranked = [sd("a", 10), sd("b", 5), sd("c", 1)]
    fused = reciprocal_rank_fusion([ranked])
    assert [d.doc_id for d in fused] == ["a", "b", "c"]


def test_doc_ranked_first_in_both_lists_wins():
    list1 = [sd("a", 10), sd("b", 5)]
    list2 = [sd("a", 3), sd("c", 9)]
    fused = reciprocal_rank_fusion([list1, list2])
    assert fused[0].doc_id == "a"


def test_agreement_beats_single_high_rank():
    # "b" is rank-2 in both lists; "a" is rank-1 in list1 but absent from list2.
    # With RRF, consistent moderate ranking across retrievers can outscore
    # a single first-place finish -- that's the whole point of fusion.
    list1 = [sd("a", 10), sd("b", 5)]
    list2 = [sd("b", 8), sd("c", 1)]
    fused = reciprocal_rank_fusion([list1, list2])
    fused_ids = [d.doc_id for d in fused]
    assert fused_ids[0] == "b"


def test_weights_scale_contribution():
    list1 = [sd("a", 10)]
    list2 = [sd("b", 10)]
    fused_equal = reciprocal_rank_fusion([list1, list2])
    assert fused_equal[0].score == fused_equal[1].score

    fused_weighted = reciprocal_rank_fusion([list1, list2], weights=[2.0, 1.0])
    a_score = next(d.score for d in fused_weighted if d.doc_id == "a")
    b_score = next(d.score for d in fused_weighted if d.doc_id == "b")
    assert a_score > b_score


def test_mismatched_weights_length_raises():
    import pytest

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[sd("a", 1)]], weights=[1.0, 2.0])


def test_empty_lists_returns_empty():
    assert reciprocal_rank_fusion([[], []]) == []
