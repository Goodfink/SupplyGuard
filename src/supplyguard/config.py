"""Validated runtime configuration for SupplyGuard."""

from dataclasses import dataclass, field, replace
from pathlib import Path

from supplyguard.errors import ConfigurationError

PACKAGE_ROOT = Path(__file__).resolve().parent


def _find_project_root() -> Path:
    for candidate in PACKAGE_ROOT.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "data").is_dir():
            return candidate
    return Path.cwd()


@dataclass(frozen=True, slots=True)
class PathConfig:
    """Filesystem locations used by commands and experiments."""

    project_root: Path
    raw_data_dir: Path
    processed_data_dir: Path
    dense_index_dir: Path
    corpus_path: Path
    evaluation_file: Path
    results_file: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> "PathConfig":
        project_root = project_root.resolve()
        raw_data_dir = project_root / "data" / "raw"
        processed_data_dir = project_root / "data" / "processed"
        return cls(
            project_root=project_root,
            raw_data_dir=raw_data_dir,
            processed_data_dir=processed_data_dir,
            dense_index_dir=project_root / "data" / "indexes" / "dense",
            corpus_path=processed_data_dir / "advisories.jsonl",
            evaluation_file=PACKAGE_ROOT / "evaluation" / "evaluation_set.json",
            results_file=project_root / "evaluation_results.json",
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Pinned models and inference settings."""

    language_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    language_model_revision: str = "7ae557604adf67be50417f59c2c2f167def9a775"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    device: str = "cpu"
    prompt_max_length: int = 16_384
    answer_max_tokens: int = 300
    baseline_max_tokens: int = 80
    embedding_batch_size: int = 32

    def __post_init__(self) -> None:
        text_values: dict[str, str] = {
            "language_model_name": self.language_model_name,
            "language_model_revision": self.language_model_revision,
            "embedding_model_name": self.embedding_model_name,
            "embedding_model_revision": self.embedding_model_revision,
            "device": self.device,
        }
        for text_name, text_value in text_values.items():
            if not text_value.strip():
                raise ConfigurationError(f"{text_name} must not be empty")

        positive_values: dict[str, int] = {
            "prompt_max_length": self.prompt_max_length,
            "answer_max_tokens": self.answer_max_tokens,
            "baseline_max_tokens": self.baseline_max_tokens,
            "embedding_batch_size": self.embedding_batch_size,
        }
        for numeric_name, numeric_value in positive_values.items():
            if numeric_value <= 0:
                raise ConfigurationError(f"{numeric_name} must be greater than 0")


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Retrieval limits, thresholds, and relative score cutoffs."""

    top_k: int = 5
    max_contexts: int = 3
    bm25_min_score: float = 10.0
    dense_min_score: float = 0.44
    bm25_relative_cutoff: float = 0.65
    dense_relative_cutoff: float = 0.70

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ConfigurationError("top_k must be greater than 0")
        if self.max_contexts <= 0:
            raise ConfigurationError("max_contexts must be greater than 0")
        if self.bm25_min_score < 0 or self.dense_min_score < 0:
            raise ConfigurationError("retrieval score thresholds must not be negative")
        for name, value in (
            ("bm25_relative_cutoff", self.bm25_relative_cutoff),
            ("dense_relative_cutoff", self.dense_relative_cutoff),
        ):
            if not 0 <= value <= 1:
                raise ConfigurationError(f"{name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Reproducibility settings shared by preprocessing and inference."""

    random_seed: int = 42
    torch_threads: int = 1
    deterministic_algorithms: bool = True

    def __post_init__(self) -> None:
        if self.torch_threads <= 0:
            raise ConfigurationError("torch_threads must be greater than 0")


@dataclass(frozen=True, slots=True)
class SupplyGuardConfig:
    """Complete application configuration."""

    paths: PathConfig = field(
        default_factory=lambda: PathConfig.from_project_root(_find_project_root())
    )
    models: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


DEFAULT_CONFIG = SupplyGuardConfig()


def with_model_overrides(
    config: SupplyGuardConfig,
    *,
    language_model_name: str | None = None,
    language_model_revision: str | None = None,
    embedding_model_name: str | None = None,
    embedding_model_revision: str | None = None,
    device: str | None = None,
) -> SupplyGuardConfig:
    """Return a validated config with optional CLI model selections applied.

    Selecting a different model without a revision uses its ``main`` revision. Users can pass an
    immutable commit hash when they need a fully reproducible run.
    """

    def resolve_revision(name: str | None, revision: str | None, default: str) -> str:
        if revision is not None:
            return revision
        if name is not None:
            return "main"
        return default

    models = replace(
        config.models,
        language_model_name=(
            config.models.language_model_name
            if language_model_name is None
            else language_model_name
        ),
        language_model_revision=resolve_revision(
            language_model_name,
            language_model_revision,
            config.models.language_model_revision,
        ),
        embedding_model_name=(
            config.models.embedding_model_name
            if embedding_model_name is None
            else embedding_model_name
        ),
        embedding_model_revision=resolve_revision(
            embedding_model_name,
            embedding_model_revision,
            config.models.embedding_model_revision,
        ),
        device=config.models.device if device is None else device,
    )
    return replace(config, models=models)
