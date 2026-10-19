from pathlib import Path

import pytest

from supplyguard.config import (
    DEFAULT_CONFIG,
    ModelConfig,
    PathConfig,
    RetrievalConfig,
    RuntimeConfig,
    with_model_overrides,
)
from supplyguard.errors import ConfigurationError


def test_path_config_derives_all_paths_from_project_root(tmp_path):
    paths = PathConfig.from_project_root(tmp_path)

    assert paths.project_root == tmp_path.resolve()
    assert paths.raw_data_dir == tmp_path.resolve() / "data" / "raw"
    assert paths.dense_index_dir == tmp_path.resolve() / "data" / "indexes" / "dense"
    assert paths.corpus_path == tmp_path.resolve() / "data" / "processed" / "advisories.jsonl"
    assert paths.results_file == tmp_path.resolve() / "evaluation_results.json"
    assert (
        paths.evaluation_file
        == Path(__file__).parents[1] / "src/supplyguard/evaluation/evaluation_set.json"
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RetrievalConfig(top_k=0),
        lambda: RetrievalConfig(dense_relative_cutoff=1.1),
        lambda: ModelConfig(answer_max_tokens=0),
        lambda: ModelConfig(language_model_name=""),
        lambda: RuntimeConfig(torch_threads=0),
    ],
)
def test_invalid_configuration_fails_fast(factory):
    with pytest.raises(ConfigurationError):
        factory()


def test_model_overrides_preserve_pinned_defaults_when_omitted():
    configured = with_model_overrides(DEFAULT_CONFIG)

    assert configured == DEFAULT_CONFIG


def test_custom_models_default_to_main_revision_and_preserve_other_settings():
    configured = with_model_overrides(
        DEFAULT_CONFIG,
        language_model_name="organization/custom-chat-model",
        embedding_model_name="organization/custom-embedding-model",
        device="mps",
    )

    assert configured.models.language_model_name == "organization/custom-chat-model"
    assert configured.models.language_model_revision == "main"
    assert configured.models.embedding_model_name == "organization/custom-embedding-model"
    assert configured.models.embedding_model_revision == "main"
    assert configured.models.device == "mps"
    assert configured.models.answer_max_tokens == DEFAULT_CONFIG.models.answer_max_tokens


def test_explicit_model_revisions_are_preserved():
    configured = with_model_overrides(
        DEFAULT_CONFIG,
        language_model_revision="language-commit",
        embedding_model_revision="embedding-commit",
    )

    assert configured.models.language_model_revision == "language-commit"
    assert configured.models.embedding_model_revision == "embedding-commit"


@pytest.mark.parametrize(
    "overrides",
    [
        {"language_model_name": ""},
        {"language_model_revision": ""},
        {"embedding_model_name": ""},
        {"embedding_model_revision": ""},
        {"device": ""},
    ],
)
def test_empty_model_overrides_fail_fast(overrides):
    with pytest.raises(ConfigurationError):
        with_model_overrides(DEFAULT_CONFIG, **overrides)
