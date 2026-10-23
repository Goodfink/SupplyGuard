"""Versioned, reproducible evaluation report persistence."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from supplyguard.config import SupplyGuardConfig
from supplyguard.errors import ConfigurationError

REPORT_SCHEMA_VERSION = 1
REPORTED_PACKAGES = (
    "supplyguard",
    "numpy",
    "sentence-transformers",
    "torch",
    "transformers",
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ConfigurationError(f"Could not hash evaluation input '{path}': {error}") from error
    return digest.hexdigest()


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def empty_evaluation_report() -> dict[str, Any]:
    """Return an empty report shape suitable for merging incremental experiment results."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "metrics": {},
        "questions": [],
    }


def load_evaluation_report(path: Path) -> dict[str, Any]:
    """Load the current schema or migrate the legacy top-level question list in memory."""
    if not path.exists():
        return empty_evaluation_report()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"Could not load evaluation report '{path}': {error}") from error

    if isinstance(payload, list):
        report = empty_evaluation_report()
        report["questions"] = payload
        return report

    if not isinstance(payload, dict):
        raise ConfigurationError(f"Evaluation report '{path}' must contain an object or list")
    if payload.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ConfigurationError(
            f"Evaluation report '{path}' uses unsupported schema version "
            f"{payload.get('schema_version')!r}"
        )
    if not isinstance(payload.get("metrics"), dict) or not isinstance(
        payload.get("questions"), list
    ):
        raise ConfigurationError(f"Evaluation report '{path}' has an invalid structure")
    return payload


def index_question_results(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index validated per-question results by their stable question IDs."""
    indexed: dict[str, dict[str, Any]] = {}
    for item in report["questions"]:
        if not isinstance(item, dict) or not isinstance(item.get("question_id"), str):
            raise ConfigurationError("Every evaluation result must have a string question_id")
        question_id = item["question_id"]
        if question_id in indexed:
            raise ConfigurationError(f"Duplicate evaluation question_id: {question_id}")
        indexed[question_id] = item
    return indexed


def build_evaluation_report(
    *,
    questions: list[dict[str, Any]],
    metrics: dict[str, Any],
    config: SupplyGuardConfig,
    corpus_path: Path,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a self-describing report from evaluation outputs and runtime configuration."""
    timestamp = generated_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ConfigurationError("Evaluation report timestamp must include a timezone")

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "run": {
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "machine": platform.machine(),
            "dependencies": {
                package_name: _package_version(package_name) for package_name in REPORTED_PACKAGES
            },
            "models": asdict(config.models),
            "retrieval": asdict(config.retrieval),
            "runtime": asdict(config.runtime),
        },
        "datasets": {
            "corpus": {
                "path": _display_path(corpus_path, config.paths.project_root),
                "sha256": _hash_file(corpus_path),
            },
            "evaluation_set": {
                "path": _display_path(
                    config.paths.evaluation_file,
                    config.paths.project_root,
                ),
                "sha256": _hash_file(config.paths.evaluation_file),
            },
        },
        "metrics": metrics,
        "questions": questions,
    }


def save_evaluation_report(path: Path, report: dict[str, Any]) -> None:
    """Atomically persist a report so interrupted writes cannot corrupt prior results."""
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(report, file, indent=2)
            file.write("\n")
        os.replace(temporary_path, path)
        temporary_path = None
    except (OSError, TypeError, ValueError) as error:
        raise ConfigurationError(f"Could not save evaluation report '{path}': {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
