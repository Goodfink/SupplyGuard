"""Build trusted source records from retrieved OSV advisories."""

import re
from typing import Any, TypedDict
from urllib.parse import quote

from supplyguard.errors import CorpusError

INLINE_CITATION_PATTERN = re.compile(
    r"\s*\[(?:GHSA|GO|CVE)-[^\]]+\]",
    flags=re.IGNORECASE,
)


class AdvisorySource(TypedDict):
    """A cited advisory that was supplied to the language model."""

    advisory_id: str
    chunk_id: str
    url: str
    summary: str
    rank: int
    score: float


def osv_advisory_url(advisory_id: str) -> str:
    """Return the canonical OSV page for an advisory identifier."""
    return f"https://osv.dev/vulnerability/{quote(advisory_id, safe='')}"


def remove_inline_citations(answer: str) -> str:
    """Remove untrusted citation markup emitted by a language model."""
    return INLINE_CITATION_PATTERN.sub("", answer).strip()


def is_abstention(answer: str) -> bool:
    """Return whether an answer explicitly declines to answer."""
    normalized = answer.casefold().replace("’", "'")
    return "i don't know" in normalized or "do not know" in normalized


def build_advisory_sources(retrieved: list[dict[str, Any]]) -> list[AdvisorySource]:
    """Build deduplicated source records from the context used for generation."""
    sources: list[AdvisorySource] = []
    seen_advisory_ids: set[str] = set()

    for result in retrieved:
        record = result["record"]
        advisory_id = str(record.get("advisory_id") or result["document_id"]).strip()
        if not advisory_id:
            raise CorpusError("Retrieved advisory is missing an advisory identifier")
        if advisory_id in seen_advisory_ids:
            continue

        seen_advisory_ids.add(advisory_id)
        sources.append(
            {
                "advisory_id": advisory_id,
                "chunk_id": str(result["chunk_id"]),
                "url": osv_advisory_url(advisory_id),
                "summary": str(result.get("summary") or record.get("summary") or ""),
                "rank": int(result["rank"]),
                "score": float(result["score"]),
            }
        )

    return sources
