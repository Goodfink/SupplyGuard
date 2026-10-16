import pytest

from supplyguard.errors import CorpusError
from supplyguard.sources import (
    build_advisory_sources,
    is_abstention,
    osv_advisory_url,
    remove_inline_citations,
)


def retrieval_result(record, rank=1, score=20.0):
    return {
        "rank": rank,
        "score": score,
        "document_id": record["document_id"],
        "chunk_id": record["chunk_id"],
        "summary": record["summary"],
        "record": record,
    }


def test_build_sources_deduplicates_advisories_and_links_to_osv(advisory_record_factory):
    record = advisory_record_factory(document_id="GHSA-abcd-1234-5678")

    sources = build_advisory_sources(
        [
            retrieval_result(record),
            retrieval_result(record, rank=2, score=15.0),
        ]
    )

    assert sources == [
        {
            "advisory_id": "GHSA-abcd-1234-5678",
            "chunk_id": "GHSA-abcd-1234-5678_chunk_0",
            "url": "https://osv.dev/vulnerability/GHSA-abcd-1234-5678",
            "summary": "Example vulnerability",
            "rank": 1,
            "score": 20.0,
        }
    ]


def test_source_helpers_sanitize_model_output_and_recognize_abstention():
    assert osv_advisory_url("GHSA-abcd-1234-5678") == (
        "https://osv.dev/vulnerability/GHSA-abcd-1234-5678"
    )
    assert (
        remove_inline_citations("Upgrade now [GHSA-abcd-1234-5678] [CVE-2025-1234]")
        == "Upgrade now"
    )
    assert is_abstention("I don't know")
    assert is_abstention("I do not know from the supplied evidence")


def test_source_builder_rejects_retrieved_records_without_an_identifier(
    advisory_record_factory,
):
    record = advisory_record_factory(document_id="")

    with pytest.raises(CorpusError, match="missing an advisory identifier"):
        build_advisory_sources([retrieval_result(record)])
