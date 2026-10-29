from pathlib import Path

import pytest

from supplyguard import cli
from supplyguard.cli import build_parser, load_evaluation_set, main
from supplyguard.config import DEFAULT_CONFIG, SupplyGuardConfig
from supplyguard.errors import CorpusError, ModelLoadError


@pytest.mark.parametrize(
    ("arguments", "command"),
    [
        (["ask", "What affects example-package?"], "ask"),
        (["evaluate"], "evaluate"),
        (["demo"], "demo"),
        (["serve"], "serve"),
        (["inspect-corpus"], "inspect-corpus"),
    ],
)
def test_parser_exposes_expected_commands(arguments, command):
    assert build_parser().parse_args(arguments).command == command


def test_ask_accepts_custom_corpus_path():
    args = build_parser().parse_args(["ask", "question", "--corpus", "custom.jsonl"])

    assert args.corpus == Path("custom.jsonl")


@pytest.mark.parametrize("command", ["ask", "evaluate", "demo", "serve"])
def test_model_commands_accept_forced_index_rebuild(command):
    arguments = [command]
    if command == "ask":
        arguments.append("question")
    arguments.append("--rebuild-index")

    assert build_parser().parse_args(arguments).rebuild_index is True


def test_ask_accepts_custom_local_model_configuration():
    args = build_parser().parse_args(
        [
            "ask",
            "question",
            "--language-model",
            "/models/local-chat",
            "--language-model-revision",
            "chat-commit",
            "--embedding-model",
            "organization/embeddings",
            "--embedding-model-revision",
            "embedding-commit",
            "--device",
            "cuda:0",
        ]
    )

    assert args.language_model == "/models/local-chat"
    assert args.language_model_revision == "chat-commit"
    assert args.embedding_model == "organization/embeddings"
    assert args.embedding_model_revision == "embedding-commit"
    assert args.device == "cuda:0"


def test_ask_uses_repository_corpus_by_default():
    args = build_parser().parse_args(["ask", "question"])

    assert args.corpus == DEFAULT_CONFIG.paths.corpus_path


def test_serve_defaults_to_local_only_access():
    args = build_parser().parse_args(["serve"])

    assert args.host == "127.0.0.1"
    assert args.port == 8000


def test_main_without_command_prints_help_without_loading_models(capsys):
    assert main([]) == 0
    assert "{ask,evaluate,demo,serve,inspect-corpus}" in capsys.readouterr().out


def test_main_routes_ask_command(monkeypatch):
    systems = (object(), object(), object(), [])
    received = []
    setup_calls = []

    def fake_setup(corpus, config=DEFAULT_CONFIG, rebuild_index=False):
        setup_calls.append((corpus, config, rebuild_index))
        return systems

    monkeypatch.setattr(cli, "setup_system", fake_setup)
    monkeypatch.setattr(
        cli,
        "run_single_query",
        lambda question, bm25, dense, llm: received.append((question, bm25, dense, llm)),
    )

    assert main(["ask", "Which version is fixed?"]) == 0
    assert setup_calls == [(DEFAULT_CONFIG.paths.corpus_path, DEFAULT_CONFIG, False)]
    assert received == [("Which version is fixed?", *systems[:3])]


def test_main_forwards_forced_index_rebuild(monkeypatch):
    systems = (object(), object(), object(), [])
    setup_calls = []
    monkeypatch.setattr(
        cli,
        "setup_system",
        lambda corpus, config=DEFAULT_CONFIG, rebuild_index=False: (
            setup_calls.append((corpus, config, rebuild_index)) or systems
        ),
    )
    monkeypatch.setattr(cli, "run_single_query", lambda *_arguments: None)

    assert main(["ask", "question", "--rebuild-index"]) == 0
    assert setup_calls == [(DEFAULT_CONFIG.paths.corpus_path, DEFAULT_CONFIG, True)]


def test_main_forwards_custom_model_configuration(monkeypatch):
    systems = (object(), object(), object(), [])
    setup_calls = []

    def fake_setup(corpus, config=DEFAULT_CONFIG, rebuild_index=False):
        setup_calls.append((corpus, config, rebuild_index))
        return systems

    monkeypatch.setattr(cli, "setup_system", fake_setup)
    monkeypatch.setattr(cli, "run_single_query", lambda *_arguments: None)

    assert (
        main(
            [
                "ask",
                "question",
                "--language-model",
                "organization/chat-model",
                "--embedding-model",
                "organization/embedding-model",
                "--device",
                "mps",
            ]
        )
        == 0
    )
    _, configured, rebuild_index = setup_calls[0]
    assert isinstance(configured, SupplyGuardConfig)
    assert configured.models.language_model_name == "organization/chat-model"
    assert configured.models.language_model_revision == "main"
    assert configured.models.embedding_model_name == "organization/embedding-model"
    assert configured.models.embedding_model_revision == "main"
    assert configured.models.device == "mps"
    assert rebuild_index is False


def test_main_routes_serve_without_eagerly_loading_models(monkeypatch):
    from supplyguard import api

    setup_calls = []
    server_calls = []
    monkeypatch.setattr(cli, "setup_system", lambda *_args, **_kwargs: setup_calls.append(True))
    monkeypatch.setattr(api, "run_api_server", lambda **settings: server_calls.append(settings))

    assert (
        main(
            [
                "serve",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
                "--language-model",
                "organization/chat-model",
            ]
        )
        == 0
    )
    assert setup_calls == []
    assert len(server_calls) == 1
    assert server_calls[0]["host"] == "0.0.0.0"
    assert server_calls[0]["port"] == 8080
    assert server_calls[0]["corpus_path"] == DEFAULT_CONFIG.paths.corpus_path
    assert server_calls[0]["config"].models.language_model_name == "organization/chat-model"
    assert server_calls[0]["rebuild_index"] is False


def test_evaluate_forwards_custom_models_to_report(monkeypatch, tmp_path):
    systems = (object(), object(), object(), [])
    evaluation_calls = []
    monkeypatch.setattr(cli, "setup_system", lambda *_args, **_kwargs: systems)
    monkeypatch.setattr(
        cli,
        "run_full_evaluation",
        lambda *args, **kwargs: evaluation_calls.append((args, kwargs)),
    )
    results_path = tmp_path / "results.json"

    assert (
        main(
            [
                "evaluate",
                "--results-file",
                str(results_path),
                "--language-model",
                "organization/chat-model",
                "--language-model-revision",
                "chat-commit",
                "--embedding-model",
                "organization/embedding-model",
                "--embedding-model-revision",
                "embedding-commit",
                "--device",
                "cuda:0",
            ]
        )
        == 0
    )
    positional, keyword = evaluation_calls[0]
    assert positional == systems[:3]
    assert keyword["results_file"] == results_path
    assert keyword["corpus_path"] == DEFAULT_CONFIG.paths.corpus_path
    assert keyword["config"].models.language_model_name == "organization/chat-model"
    assert keyword["config"].models.language_model_revision == "chat-commit"
    assert keyword["config"].models.embedding_model_name == "organization/embedding-model"
    assert keyword["config"].models.embedding_model_revision == "embedding-commit"
    assert keyword["config"].models.device == "cuda:0"


def test_evaluation_set_is_available_as_package_data():
    questions = load_evaluation_set()

    assert len(questions) == 14
    assert questions[0]["question_id"] == "q1"


def test_print_sources_displays_verified_advisory_link(capsys):
    cli.print_sources(
        [
            {
                "advisory_id": "GHSA-abcd-1234-5678",
                "url": "https://osv.dev/vulnerability/GHSA-abcd-1234-5678",
            }
        ]
    )

    output = capsys.readouterr().out
    assert "GHSA-abcd-1234-5678" in output
    assert "https://osv.dev/vulnerability/GHSA-abcd-1234-5678" in output


def test_print_evaluation_summary_includes_retrieval_quality(capsys):
    stats = {
        "accuracy": 0.5,
        "correct_answers": 2,
        "total_questions": 4,
        "idontknow_accuracy": 1.0,
        "citation_accuracy": 0.75,
        "citation_grounding_accuracy": 1.0,
        "breakdown": {
            "factoid": {"accuracy": 0.5, "correct": 1, "count": 2},
            "multi_hop": {"accuracy": 0.0, "correct": 0, "count": 1},
            "unanswerable": {"accuracy": 1.0, "correct": 1, "count": 1},
        },
        "retrieval": {
            "top_k": 5,
            "recall_at_k": 0.75,
            "hit_rate_at_k": 1.0,
            "mrr_at_k": 0.625,
            "complete_recall_rate_at_k": 0.5,
            "unanswerable_empty_result_rate": 1.0,
        },
    }

    cli.print_evaluation_summary("BM25 Results", stats)

    output = capsys.readouterr().out
    assert "Retrieval Recall@5: 75.0%" in output
    assert "Retrieval Hit Rate@5: 100.0%" in output
    assert "Retrieval MRR@5: 0.625" in output
    assert "Complete Evidence@5: 50.0%" in output
    assert "Unanswerable Empty-Result Rate: 100.0%" in output


def test_main_reports_typed_errors_with_stable_exit_code(monkeypatch, capsys):
    def fail_setup(_corpus, config=DEFAULT_CONFIG, rebuild_index=False):
        raise ModelLoadError("model cache is unavailable")

    monkeypatch.setattr(cli, "setup_system", fail_setup)

    assert main(["ask", "Which version is fixed?"]) == ModelLoadError.exit_code
    assert capsys.readouterr().err == "SupplyGuard error: model cache is unavailable\n"


def test_invalid_question_fails_before_models_are_loaded(monkeypatch, capsys):
    setup_calls = []
    monkeypatch.setattr(cli, "setup_system", lambda *_args, **_kwargs: setup_calls.append(True))
    oversized_question = "x" * (DEFAULT_CONFIG.models.question_max_chars + 1)

    assert main(["ask", oversized_question]) == 2
    assert setup_calls == []
    assert "Question exceeds the maximum length" in capsys.readouterr().err


def test_missing_corpus_fails_before_model_imports(tmp_path, capsys):
    missing_path = tmp_path / "missing.jsonl"

    assert main(["ask", "question", "--corpus", str(missing_path)]) == CorpusError.exit_code
    assert "Could not read advisory corpus" in capsys.readouterr().err
