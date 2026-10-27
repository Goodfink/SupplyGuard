import pytest

from supplyguard.errors import ConfigurationError
from supplyguard.retrieval.hybrid import reciprocal_rank_fusion


def search_result(document_id, score):
    return {
        "document_id": document_id,
        "chunk_id": f"{document_id}-chunk",
        "score": score,
        "record": {"document_id": document_id},
    }


def test_rank_fusion_rewards_documents_found_by_both_retrievers():
    bm25 = [search_result("lexical-only", 40.0), search_result("shared", 30.0)]
    dense = [search_result("semantic-only", 0.9), search_result("shared", 0.8)]

    results = reciprocal_rank_fusion(bm25, dense, top_k=3, rrf_k=60)

    assert [result["document_id"] for result in results] == [
        "shared",
        "lexical-only",
        "semantic-only",
    ]
    assert results[0]["retriever_ranks"] == {"bm25": 2, "dense": 2}
    assert results[0]["retriever_scores"] == {"bm25": 30.0, "dense": 0.8}
    assert results[0]["fusion_method"] == "reciprocal_rank_fusion"
    assert results[0]["rank"] == 1


def test_rank_fusion_deduplicates_documents_within_each_ranked_list():
    duplicate = search_result("shared", 20.0)

    results = reciprocal_rank_fusion(
        [duplicate, duplicate],
        [search_result("shared", 0.7)],
        top_k=5,
        rrf_k=60,
    )

    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(2 / 61)
    assert results[0]["retriever_ranks"] == {"bm25": 1, "dense": 1}


def test_rank_fusion_applies_output_limit_and_deterministic_tie_breaking():
    results = reciprocal_rank_fusion(
        [search_result("b-document", 20.0)],
        [search_result("a-document", 0.8)],
        top_k=1,
        rrf_k=60,
    )

    assert [result["document_id"] for result in results] == ["a-document"]


@pytest.mark.parametrize(("top_k", "rrf_k"), [(0, 60), (5, 0)])
def test_rank_fusion_rejects_invalid_configuration(top_k, rrf_k):
    with pytest.raises(ConfigurationError):
        reciprocal_rank_fusion([], [], top_k=top_k, rrf_k=rrf_k)
