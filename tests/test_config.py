from pathlib import Path

import pytest

from supplyguard.config import ModelConfig, PathConfig, RetrievalConfig, RuntimeConfig
from supplyguard.errors import ConfigurationError


def test_path_config_derives_all_paths_from_project_root(tmp_path):
    paths = PathConfig.from_project_root(tmp_path)

    assert paths.project_root == tmp_path.resolve()
    assert paths.raw_data_dir == tmp_path.resolve() / "data" / "raw"
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
