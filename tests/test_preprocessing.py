from datetime import datetime, timezone

from supplyguard.helpers.data_helpers import is_valid_advisory
from supplyguard.preprocess import build_retrieval_text, normalize_advisory


START_DATE = datetime(2022, 1, 1, tzinfo=timezone.utc)
CUTOFF_DATE = datetime(2026, 12, 31, tzinfo=timezone.utc)


def raw_advisory():
    return {
        "id": "GHSA-abcd-1234-5678",
        "aliases": ["CVE-2025-1234"],
        "published": "2025-02-03T10:00:00Z",
        "modified": "2025-02-04T10:00:00Z",
        "summary": "Path traversal in example-package",
        "details": "Attackers can read files outside the intended directory.",
        "affected": [
            {
                "package": {
                    "name": "example-package",
                    "ecosystem": "PyPI",
                    "purl": "pkg:pypi/example-package",
                },
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [
                            {"introduced": "0"},
                            {"fixed": "2.1.0"},
                        ],
                    }
                ],
                "versions": ["2.0.0"],
            }
        ],
        "references": [{"type": "ADVISORY", "url": "https://osv.dev/test"}],
    }


def test_is_valid_advisory_accepts_complete_record():
    assert is_valid_advisory(raw_advisory(), START_DATE, CUTOFF_DATE)


def test_is_valid_advisory_rejects_withdrawn_and_malicious_records():
    withdrawn = raw_advisory()
    withdrawn["withdrawn"] = "2025-02-05T10:00:00Z"
    assert not is_valid_advisory(withdrawn, START_DATE, CUTOFF_DATE)

    malicious = raw_advisory()
    malicious["id"] = "MAL-2025-example"
    assert not is_valid_advisory(malicious, START_DATE, CUTOFF_DATE)


def test_normalize_advisory_extracts_retrieval_fields():
    normalized = normalize_advisory(
        {
            "record": raw_advisory(),
            "source_ecosystem": "PyPI",
            "source_filename": "2025/GHSA-abcd-1234-5678.json",
            "published_year": 2025,
        }
    )

    assert normalized["document_id"] == "GHSA-abcd-1234-5678"
    assert normalized["packages"][0]["name"] == "example-package"
    assert normalized["fixed_versions"] == [
        {"package": "example-package", "version": "2.1.0"}
    ]
    assert "CVE-2025-1234" in normalized["retrieval_text"]
    assert "example-package fixed in 2.1.0" in normalized["retrieval_text"]


def test_build_retrieval_text_omits_empty_optional_sections():
    text = build_retrieval_text(
        advisory_id="GHSA-minimal-0000-0001",
        aliases=[],
        packages=[],
        summary="Minimal advisory",
        details="",
        affected_ranges=[],
        fixed_versions=[],
    )

    assert text == (
        "Advisory ID: GHSA-minimal-0000-0001\n"
        "Summary: Minimal advisory"
    )
