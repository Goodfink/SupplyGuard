from pathlib import Path

import pytest

from supplyguard import cli
from supplyguard.cli import build_parser, load_evaluation_set, main
from supplyguard.paths import DEFAULT_CORPUS_PATH


@pytest.mark.parametrize(
    ("arguments", "command"),
    [
        (["ask", "What affects example-package?"], "ask"),
        (["evaluate"], "evaluate"),
        (["demo"], "demo"),
        (["inspect-corpus"], "inspect-corpus"),
    ],
)
def test_parser_exposes_expected_commands(arguments, command):
    assert build_parser().parse_args(arguments).command == command


def test_ask_accepts_custom_corpus_path():
    args = build_parser().parse_args(
        ["ask", "question", "--corpus", "custom.jsonl"]
    )

    assert args.corpus == Path("custom.jsonl")


def test_ask_uses_repository_corpus_by_default():
    args = build_parser().parse_args(["ask", "question"])

    assert args.corpus == DEFAULT_CORPUS_PATH


def test_main_without_command_prints_help_without_loading_models(capsys):
    assert main([]) == 0
    assert "{ask,evaluate,demo,inspect-corpus}" in capsys.readouterr().out


def test_main_routes_ask_command(monkeypatch):
    systems = (object(), object(), object(), [])
    received = []
    monkeypatch.setattr(cli, "setup_system", lambda corpus: systems)
    monkeypatch.setattr(
        cli,
        "run_single_query",
        lambda question, bm25, dense, llm: received.append(
            (question, bm25, dense, llm)
        ),
    )

    assert main(["ask", "Which version is fixed?"]) == 0
    assert received == [("Which version is fixed?", *systems[:3])]


def test_evaluation_set_is_available_as_package_data():
    questions = load_evaluation_set()

    assert len(questions) == 14
    assert questions[0]["question_id"] == "q1"
