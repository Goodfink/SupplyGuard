"""Rank fusion for combining lexical and semantic retrieval results."""

from typing import Any

from supplyguard.errors import ConfigurationError

type SearchResult = dict[str, Any]


def reciprocal_rank_fusion(
    bm25_results: list[SearchResult],
    dense_results: list[SearchResult],
    *,
    top_k: int,
    rrf_k: int,
) -> list[SearchResult]:
    """Fuse two ranked lists without comparing their incompatible raw scores."""
    if top_k <= 0:
        raise ConfigurationError("top_k must be greater than 0")
    if rrf_k <= 0:
        raise ConfigurationError("rrf_k must be greater than 0")

    fused: dict[str, SearchResult] = {}
    for retriever_name, results in (("bm25", bm25_results), ("dense", dense_results)):
        seen_document_ids: set[str] = set()
        for rank, result in enumerate(results, start=1):
            document_id = str(result["document_id"])
            if document_id in seen_document_ids:
                continue
            seen_document_ids.add(document_id)

            if document_id not in fused:
                fused[document_id] = {
                    **result,
                    "score": 0.0,
                    "retriever_ranks": {},
                    "retriever_scores": {},
                    "fusion_method": "reciprocal_rank_fusion",
                }

            fused_result = fused[document_id]
            fused_result["score"] += 1 / (rrf_k + rank)
            fused_result["retriever_ranks"][retriever_name] = rank
            fused_result["retriever_scores"][retriever_name] = float(result["score"])

    ranked = sorted(
        fused.values(),
        key=lambda result: (
            -float(result["score"]),
            min(result["retriever_ranks"].values()),
            str(result["document_id"]),
        ),
    )[:top_k]

    return [{**result, "rank": rank} for rank, result in enumerate(ranked, start=1)]
