import pytest

from supplyguard.errors import ConfigurationError
from supplyguard.evaluation.retrieval_metrics import (
    aggregate_retrieval_results,
    evaluate_retrieval,
)


def test_evaluate_retrieval_scores_rank_and_complete_multi_hop_recall():
    result = evaluate_retrieval(
        ["GHSA-noise", "GHSA-two", "GHSA-one"],
        ["GHSA-one", "GHSA-two"],
        "multi-hop",
    )

    assert result["evaluable"] is True
    assert result["retrieved_relevant_document_ids"] == ["GHSA-two", "GHSA-one"]
    assert result["recall"] == pytest.approx(1.0)
    assert result["hit"] is True
    assert result["reciprocal_rank"] == pytest.approx(0.5)
    assert result["complete_recall"] is True


def test_partial_multi_hop_recall_is_not_treated_as_complete():
    result = evaluate_retrieval(
        ["GHSA-one", "GHSA-noise"],
        ["GHSA-one", "GHSA-two"],
        "multi-hop",
    )

    assert result["recall"] == pytest.approx(0.5)
    assert result["hit"] is True
    assert result["reciprocal_rank"] == pytest.approx(1.0)
    assert result["complete_recall"] is False


def test_duplicate_documents_do_not_inflate_recall():
    result = evaluate_retrieval(
        ["GHSA-one", "GHSA-one", "GHSA-noise"],
        ["GHSA-one", "GHSA-two"],
        "multi-hop",
    )

    assert result["retrieved_count"] == 2
    assert result["retrieved_relevant_document_ids"] == ["GHSA-one"]
    assert result["recall"] == pytest.approx(0.5)


def test_missed_evidence_has_zero_rank_metrics():
    result = evaluate_retrieval(["GHSA-noise"], ["GHSA-one"], "factoid")

    assert result["recall"] == pytest.approx(0.0)
    assert result["hit"] is False
    assert result["reciprocal_rank"] == pytest.approx(0.0)
    assert result["complete_recall"] is False


def test_unanswerable_query_is_excluded_from_relevance_metrics():
    result = evaluate_retrieval([], [], "unanswerable")

    assert result["evaluable"] is False
    assert result["recall"] is None
    assert result["hit"] is None
    assert result["reciprocal_rank"] is None
    assert result["complete_recall"] is None


def test_aggregate_retrieval_reports_quality_breakdown_and_empty_result_rate():
    results = [
        evaluate_retrieval(["GHSA-one"], ["GHSA-one"], "factoid"),
        evaluate_retrieval(
            ["GHSA-noise", "GHSA-two"],
            ["GHSA-two", "GHSA-three"],
            "multi-hop",
        ),
        evaluate_retrieval([], [], "unanswerable"),
        evaluate_retrieval(["GHSA-noise"], [], "unanswerable"),
    ]

    aggregate = aggregate_retrieval_results(results, top_k=5)

    assert aggregate["top_k"] == 5
    assert aggregate["evaluable_queries"] == 2
    assert aggregate["recall_at_k"] == pytest.approx(0.75)
    assert aggregate["hit_rate_at_k"] == pytest.approx(1.0)
    assert aggregate["mrr_at_k"] == pytest.approx(0.75)
    assert aggregate["complete_recall_rate_at_k"] == pytest.approx(0.5)
    assert aggregate["unanswerable_queries"] == 2
    assert aggregate["unanswerable_empty_result_rate"] == pytest.approx(0.5)
    assert aggregate["breakdown"]["factoid"]["recall_at_k"] == pytest.approx(1.0)
    assert aggregate["breakdown"]["multi_hop"]["recall_at_k"] == pytest.approx(0.5)


def test_aggregate_retrieval_handles_no_evaluable_queries():
    aggregate = aggregate_retrieval_results(
        [evaluate_retrieval([], [], "unanswerable")],
        top_k=5,
    )

    assert aggregate["recall_at_k"] is None
    assert aggregate["hit_rate_at_k"] is None
    assert aggregate["mrr_at_k"] is None
    assert aggregate["complete_recall_rate_at_k"] is None
    assert aggregate["unanswerable_empty_result_rate"] == pytest.approx(1.0)


def test_aggregate_retrieval_rejects_invalid_cutoff():
    with pytest.raises(ConfigurationError, match="top_k must be greater than 0"):
        aggregate_retrieval_results([], top_k=0)
