import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from supplyguard import cli
from supplyguard.config import DEFAULT_CONFIG, PathConfig, with_model_overrides
from supplyguard.errors import ConfigurationError
from supplyguard.evaluation.reporting import (
    build_evaluation_report,
    index_question_results,
    load_evaluation_report,
    save_evaluation_report,
)


def test_legacy_question_list_is_migrated_in_memory(tmp_path):
    report_path = tmp_path / "results.json"
    questions = [{"question_id": "q1", "question": "Example?"}]
    report_path.write_text(json.dumps(questions), encoding="utf-8")

    report = load_evaluation_report(report_path)

    assert report == {"schema_version": 1, "metrics": {}, "questions": questions}


def test_structured_report_round_trips_atomically(tmp_path):
    report_path = tmp_path / "nested" / "results.json"
    report = {
        "schema_version": 1,
        "metrics": {"bm25": {"accuracy": 0.75}},
        "questions": [{"question_id": "q1"}],
    }

    save_evaluation_report(report_path, report)

    assert load_evaluation_report(report_path) == report
    assert list(report_path.parent.glob("*.tmp")) == []


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 999, "metrics": {}, "questions": []},
        {"schema_version": 1, "metrics": [], "questions": []},
        "invalid",
    ],
)
def test_invalid_reports_fail_with_actionable_error(tmp_path, payload):
    report_path = tmp_path / "results.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Evaluation report"):
        load_evaluation_report(report_path)


def test_question_results_require_unique_stable_ids():
    report = {
        "schema_version": 1,
        "metrics": {},
        "questions": [{"question_id": "q1"}, {"question_id": "q1"}],
    }

    with pytest.raises(ConfigurationError, match="Duplicate evaluation question_id"):
        index_question_results(report)


def test_report_captures_runtime_and_dataset_provenance(tmp_path):
    paths = PathConfig.from_project_root(tmp_path)
    paths.processed_data_dir.mkdir(parents=True)
    corpus_path = paths.corpus_path
    corpus_path.write_text('{"document_id":"OSV-1"}\n', encoding="utf-8")
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text('[{"question_id":"q1"}]\n', encoding="utf-8")
    paths = replace(paths, evaluation_file=evaluation_path)
    config = replace(DEFAULT_CONFIG, paths=paths)
    generated_at = datetime(2026, 10, 23, 12, 30, tzinfo=UTC)

    report = build_evaluation_report(
        questions=[{"question_id": "q1"}],
        metrics={"bm25": {"accuracy": 1.0}},
        config=config,
        corpus_path=corpus_path,
        generated_at=generated_at,
    )

    assert report["schema_version"] == 1
    assert report["generated_at"] == "2026-10-23T12:30:00Z"
    assert report["run"]["python_version"]
    assert report["run"]["platform"]
    assert set(report["run"]["dependencies"]) == {
        "supplyguard",
        "numpy",
        "sentence-transformers",
        "torch",
        "transformers",
    }
    assert report["run"]["models"]["language_model_revision"]
    assert report["run"]["retrieval"] == {
        "top_k": 5,
        "max_contexts": 3,
        "bm25_min_score": 10.0,
        "dense_min_score": 0.44,
        "bm25_relative_cutoff": 0.65,
        "dense_relative_cutoff": 0.7,
    }
    assert report["run"]["runtime"]["random_seed"] == 42
    assert report["datasets"]["corpus"] == {
        "path": "data/processed/advisories.jsonl",
        "sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
    }
    assert report["datasets"]["evaluation_set"] == {
        "path": "evaluation.json",
        "sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
    }
    assert report["metrics"]["bm25"]["accuracy"] == 1.0
    assert report["questions"] == [{"question_id": "q1"}]


def test_report_rejects_timestamp_without_timezone(tmp_path):
    paths = PathConfig.from_project_root(tmp_path)
    paths.processed_data_dir.mkdir(parents=True)
    paths.corpus_path.write_text("corpus", encoding="utf-8")
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text("[]", encoding="utf-8")
    config = replace(DEFAULT_CONFIG, paths=replace(paths, evaluation_file=evaluation_path))

    with pytest.raises(ConfigurationError, match="timestamp must include a timezone"):
        build_evaluation_report(
            questions=[],
            metrics={},
            config=config,
            corpus_path=paths.corpus_path,
            generated_at=datetime(2026, 10, 23),
        )


def test_full_evaluation_preserves_baseline_and_writes_aggregate_report(tmp_path, monkeypatch):
    paths = PathConfig.from_project_root(tmp_path)
    paths.processed_data_dir.mkdir(parents=True)
    paths.corpus_path.write_text('{"document_id":"GHSA-one"}\n', encoding="utf-8")
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_set = [
        {
            "question_id": "q1",
            "question": "Which version is fixed?",
            "question_type": "factoid",
            "ground_truth_answer": "2.0.0",
            "ground_truth_chunk_ids": ["GHSA-one"],
            "required_answer_terms": [["2.0.0"]],
            "forbidden_answer_terms": [],
        }
    ]
    evaluation_path.write_text(json.dumps(evaluation_set), encoding="utf-8")
    results_path = tmp_path / "results.json"
    paths = replace(paths, evaluation_file=evaluation_path, results_file=results_path)
    config = with_model_overrides(
        replace(DEFAULT_CONFIG, paths=paths),
        language_model_name="organization/chat-model",
        language_model_revision="chat-commit",
        embedding_model_name="organization/embedding-model",
        embedding_model_revision="embedding-commit",
        device="cuda:0",
    )
    save_evaluation_report(
        results_path,
        {
            "schema_version": 1,
            "metrics": {"no_retrieval": {"accuracy": 0.0}},
            "questions": [],
        },
    )
    monkeypatch.setattr(cli, "load_evaluation_set", lambda: evaluation_set)

    def fake_generate(_question, _bm25, _dense, _llm):
        system_result = {
            "answer": "Version 2.0.0 fixes the issue.",
            "sources": [
                {
                    "advisory_id": "GHSA-one",
                    "url": "https://osv.dev/vulnerability/GHSA-one",
                }
            ],
            "retrieved_chunks": [
                {
                    "document_id": "GHSA-one",
                    "record": {"advisory_id": "GHSA-one"},
                }
            ],
        }
        return {"bm25": system_result, "dense": system_result}

    from supplyguard.llm import answer_generation

    monkeypatch.setattr(answer_generation, "generate_with_both_retrievers", fake_generate)

    cli.run_full_evaluation(
        object(),
        object(),
        object(),
        results_file=results_path,
        config=config,
        corpus_path=paths.corpus_path,
    )

    report = load_evaluation_report(results_path)
    assert report["metrics"]["no_retrieval"]["accuracy"] == 0.0
    assert report["metrics"]["bm25"]["accuracy"] == 1.0
    assert report["metrics"]["dense"]["accuracy"] == 1.0
    assert report["metrics"]["bm25"]["retrieval"]["recall_at_k"] == 1.0
    assert report["metrics"]["dense"]["retrieval"]["mrr_at_k"] == 1.0
    assert report["metrics"]["comparison"]["dense_minus_bm25_accuracy_points"] == 0.0
    assert report["run"]["models"]["language_model_name"] == "organization/chat-model"
    assert report["run"]["models"]["language_model_revision"] == "chat-commit"
    assert report["run"]["models"]["embedding_model_name"] == "organization/embedding-model"
    assert report["run"]["models"]["embedding_model_revision"] == "embedding-commit"
    assert report["run"]["models"]["device"] == "cuda:0"
    assert report["questions"][0]["bm25"]["sources"][0]["advisory_id"] == "GHSA-one"
    assert report["questions"][0]["bm25"]["retrieval_metrics"]["complete_recall"] is True
