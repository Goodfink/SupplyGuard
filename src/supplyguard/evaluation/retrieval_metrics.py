"""Document-level metrics for evaluating ranked retrieval results."""

from __future__ import annotations

from typing import Any

from supplyguard.errors import ConfigurationError


def _unique_in_order(document_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(document_ids))


def evaluate_retrieval(
    retrieved_document_ids: list[str],
    relevant_document_ids: list[str],
    question_type: str,
) -> dict[str, Any]:
    """Score one ranked result list against its relevant advisory IDs."""
    retrieved = _unique_in_order(retrieved_document_ids)
    relevant = set(relevant_document_ids)
    retrieved_relevant = [document_id for document_id in retrieved if document_id in relevant]

    result: dict[str, Any] = {
        "question_type": question_type,
        "evaluable": bool(relevant),
        "relevant_document_ids": sorted(relevant),
        "retrieved_relevant_document_ids": retrieved_relevant,
        "retrieved_count": len(retrieved),
        "recall": None,
        "hit": None,
        "reciprocal_rank": None,
        "complete_recall": None,
    }
    if not relevant:
        return result

    recall = len(retrieved_relevant) / len(relevant)
    first_relevant_rank = next(
        (rank for rank, document_id in enumerate(retrieved, 1) if document_id in relevant),
        None,
    )
    result.update(
        {
            "recall": recall,
            "hit": first_relevant_rank is not None,
            "reciprocal_rank": 0.0 if first_relevant_rank is None else 1 / first_relevant_rank,
            "complete_recall": recall == 1.0,
        }
    )
    return result


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate_answerable(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(results),
        "recall_at_k": _average([float(result["recall"]) for result in results]),
        "hit_rate_at_k": _average([float(result["hit"]) for result in results]),
        "mrr_at_k": _average([float(result["reciprocal_rank"]) for result in results]),
        "complete_recall_rate_at_k": _average(
            [float(result["complete_recall"]) for result in results]
        ),
    }


def aggregate_retrieval_results(
    results: list[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    """Aggregate answerable retrieval quality and unanswerable empty-result behavior."""
    if top_k <= 0:
        raise ConfigurationError("top_k must be greater than 0")

    answerable = [result for result in results if result["evaluable"]]
    unanswerable = [result for result in results if not result["evaluable"]]
    factoid = [result for result in answerable if result["question_type"] == "factoid"]
    multi_hop = [result for result in answerable if result["question_type"] == "multi-hop"]
    overall = _aggregate_answerable(answerable)

    return {
        "top_k": top_k,
        "evaluable_queries": len(answerable),
        "recall_at_k": overall["recall_at_k"],
        "hit_rate_at_k": overall["hit_rate_at_k"],
        "mrr_at_k": overall["mrr_at_k"],
        "complete_recall_rate_at_k": overall["complete_recall_rate_at_k"],
        "unanswerable_queries": len(unanswerable),
        "unanswerable_empty_result_rate": _average(
            [float(result["retrieved_count"] == 0) for result in unanswerable]
        ),
        "breakdown": {
            "factoid": _aggregate_answerable(factoid),
            "multi_hop": _aggregate_answerable(multi_hop),
        },
    }
