import pytest

from supplyguard.evaluation.evaluation_metrics import EvaluationMetrics


def test_factoid_requires_all_facts_and_no_forbidden_terms():
    metrics = EvaluationMetrics()

    correct = metrics.evaluate_answer(
        "Upgrade Caddy to 2.5.0 [GHSA-2927-hv3p-f3vp]",
        "Caddy is fixed in 2.5.0",
        "factoid",
        ["GHSA-2927-hv3p-f3vp"],
        required_answer_terms=[["2.5.0"]],
        forbidden_answer_terms=["2.4.0"],
    )
    forbidden = metrics.evaluate_answer(
        "Use 2.5.0, not 2.4.0 [GHSA-2927-hv3p-f3vp]",
        "Caddy is fixed in 2.5.0",
        "factoid",
        ["GHSA-2927-hv3p-f3vp"],
        required_answer_terms=[["2.5.0"]],
        forbidden_answer_terms=["2.4.0"],
    )

    assert correct["correct"]
    assert correct["citations_accurate"]
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


def test_aggregate_results_reports_question_breakdown():
    metrics = EvaluationMetrics()
    results = [
        metrics.evaluate_answer(
            "Version 2.5.0 [GHSA-one]",
            "2.5.0",
            "factoid",
            ["GHSA-one"],
            required_answer_terms=[["2.5.0"]],
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
    assert aggregate["idontknow_accuracy"] == pytest.approx(1.0)
    assert aggregate["breakdown"]["factoid"] == {
        "count": 1,
        "correct": 1,
        "accuracy": 1.0,
    }
