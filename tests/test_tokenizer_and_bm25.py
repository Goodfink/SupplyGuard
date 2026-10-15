import json

import pytest

from supplyguard.retrieval.bm25 import BM25Retriever, load_corpus
from supplyguard.retrieval.tokenizer import tokenize


def test_tokenize_preserves_security_and_package_identifiers():
    assert tokenize("CVE-2024-1234 affects @scope/pkg and github.com/org/repo") == [
        "cve-2024-1234",
        "affects",
        "@scope/pkg",
        "and",
        "github.com/org/repo",
    ]


@pytest.mark.parametrize("value", [None, "", "--- ... ///"])
def test_tokenize_empty_values(value):
    assert tokenize(value) == []


def test_bm25_ranks_exact_package_match_first(advisory_record_factory):
    target = advisory_record_factory(
        document_id="GHSA-target-0000-0001",
        package_name="github.com/dhowden/tag",
        summary="Out-of-bounds read in tag parser",
    )
    unrelated = advisory_record_factory(
        document_id="GHSA-other-0000-0002",
        package_name="example-web-framework",
        summary="Out-of-bounds read in an unrelated parser",
    )
    retriever = BM25Retriever()
    retriever.build_index([unrelated, target])

    results = retriever.search("github.com/dhowden/tag out-of-bounds read", top_k=2)

    assert results[0]["document_id"] == target["document_id"]
    assert results[0]["score"] > results[1]["score"]


def test_bm25_rejects_search_before_index_is_built():
    with pytest.raises(ValueError, match="Build the BM25 index"):
        BM25Retriever().search("query")


def test_bm25_rejects_invalid_index_and_search_arguments(advisory_record_factory):
    retriever = BM25Retriever()
    with pytest.raises(ValueError, match="No records"):
        retriever.build_index([])

    retriever.build_index([advisory_record_factory()])
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("query", top_k=0)
    assert retriever.search("---") == []


def test_load_corpus_reads_valid_jsonl(tmp_path, advisory_record_factory):
    record = advisory_record_factory()
    corpus_path = tmp_path / "advisories.jsonl"
    corpus_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    assert load_corpus(corpus_path) == [record]


def test_load_corpus_reports_invalid_json_line(tmp_path):
    corpus_path = tmp_path / "advisories.jsonl"
    corpus_path.write_text('{"retrieval_text": "valid"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON on line 2"):
        load_corpus(corpus_path)


def test_load_corpus_requires_retrieval_text(tmp_path):
    corpus_path = tmp_path / "advisories.jsonl"
    corpus_path.write_text('{"document_id": "GHSA-missing"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing retrieval_text"):
        load_corpus(corpus_path)
