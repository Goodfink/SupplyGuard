import pytest

from supplyguard.evaluation.evaluation_metrics import EvaluationMetrics


def test_factoid_requires_all_facts_and_no_forbidden_terms():
    metrics = EvaluationMetrics()

    correct = metrics.evaluate_answer(
        "Upgrade Caddy to 2.5.0",
        "Caddy is fixed in 2.5.0",
        "factoid",
        ["GHSA-2927-hv3p-f3vp"],
        required_answer_terms=[["2.5.0"]],
        forbidden_answer_terms=["2.4.0"],
        cited_ids=["GHSA-2927-hv3p-f3vp"],
        retrieved_document_ids=["GHSA-2927-hv3p-f3vp"],
    )
    forbidden = metrics.evaluate_answer(
        "Use 2.5.0, not 2.4.0",
        "Caddy is fixed in 2.5.0",
        "factoid",
        ["GHSA-2927-hv3p-f3vp"],
        required_answer_terms=[["2.5.0"]],
        forbidden_answer_terms=["2.4.0"],
        cited_ids=["GHSA-2927-hv3p-f3vp"],
        retrieved_document_ids=["GHSA-2927-hv3p-f3vp"],
    )

    assert correct["correct"]
    assert correct["citations_accurate"]
    assert correct["citations_grounded"]
    assert not forbidden["correct"]


def test_citation_accuracy_requires_exact_set():
    metrics = EvaluationMetrics()

    assert metrics.evaluate_citation_accuracy(
        {"GHSA-one", "GHSA-two"},
        ["GHSA-two", "GHSA-one"],
    )
    assert not metrics.evaluate_citation_accuracy(
        {"GHSA-one", "GHSA-extra"},
        ["GHSA-one"],
    )


def test_unanswerable_response_requires_abstention_without_citation():
    metrics = EvaluationMetrics()

    result = metrics.evaluate_answer(
        "I don't know",
        "I don't know",
        "unanswerable",
        [],
    )

    assert result["correct"]
    assert result["has_appropriate_idontknow"]
    assert result["citations_accurate"]
    assert result["citations_grounded"]


def test_citation_grounding_rejects_advisories_not_in_retrieved_context():
    metrics = EvaluationMetrics()

    result = metrics.evaluate_answer(
        "Upgrade to 2.5.0",
        "Upgrade to 2.5.0",
        "factoid",
        ["GHSA-retrieved"],
        cited_ids=["GHSA-hallucinated"],
        retrieved_document_ids=["GHSA-retrieved"],
    )

    assert not result["citations_accurate"]
    assert not result["citations_grounded"]
    assert result["cited_advisory_ids"] == ["GHSA-hallucinated"]


def test_aggregate_results_reports_question_breakdown():
    metrics = EvaluationMetrics()
    results = [
        metrics.evaluate_answer(
            "Version 2.5.0",
            "2.5.0",
            "factoid",
            ["GHSA-one"],
            required_answer_terms=[["2.5.0"]],
            cited_ids=["GHSA-one"],
            retrieved_document_ids=["GHSA-one"],
        ),
        metrics.evaluate_answer(
            "I don't know",
            "I don't know",
            "unanswerable",
            [],
        ),
    ]

    aggregate = metrics.aggregate_results(results)

    assert aggregate["accuracy"] == pytest.approx(1.0)
    assert aggregate["citation_accuracy"] == pytest.approx(1.0)
    assert aggregate["citation_grounding_accuracy"] == pytest.approx(1.0)
    assert aggregate["idontknow_accuracy"] == pytest.approx(1.0)
    assert aggregate["breakdown"]["factoid"] == {
        "count": 1,
        "correct": 1,
        "accuracy": 1.0,
    }
