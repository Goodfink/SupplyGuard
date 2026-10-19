"""Persistent, validated storage for dense retrieval embeddings."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from supplyguard.errors import RetrievalIndexError

CACHE_SCHEMA_VERSION = 1


def hash_file(path: Path) -> str:
    """Return the SHA-256 digest of a corpus file without loading it all at once."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise RetrievalIndexError(f"Could not hash corpus '{path}': {error}") from error
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DenseIndexIdentity:
    """Inputs that must match before cached embeddings can be reused."""

    corpus_sha256: str
    model_name: str
    model_revision: str
    document_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "corpus_sha256": self.corpus_sha256,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "document_ids": list(self.document_ids),
        }

    @property
    def cache_key(self) -> str:
        serialized = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()[:20]


class DenseIndexCache:
    """Load and atomically persist one content-addressed dense index."""

    def __init__(self, directory: Path, identity: DenseIndexIdentity):
        self.directory = directory
        self.identity = identity
        stem = f"dense-{identity.cache_key}"
        self.metadata_path = directory / f"{stem}.json"
        self.embeddings_path = directory / f"{stem}.npy"

    def load(self) -> np.ndarray | None:
        """Return a valid cached embedding matrix, or ``None`` for any cache miss."""
        if not self.metadata_path.is_file() or not self.embeddings_path.is_file():
            return None

        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                return None
            if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
                return None
            for name, expected in self.identity.as_dict().items():
                if metadata.get(name) != expected:
                    return None

            embeddings = np.load(self.embeddings_path, allow_pickle=False)
            if (
                not isinstance(embeddings, np.ndarray)
                or embeddings.ndim != 2
                or embeddings.shape[1] == 0
            ):
                return None
            if embeddings.shape[0] != len(self.identity.document_ids):
                return None
            if list(embeddings.shape) != metadata.get("shape"):
                return None
            if str(embeddings.dtype) != metadata.get("dtype"):
                return None
            if embeddings.dtype.kind != "f" or not np.isfinite(embeddings).all():
                return None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

        return embeddings

    def save(self, embeddings: np.ndarray) -> None:
        """Persist embeddings and metadata using same-directory atomic replacements."""
        matrix = np.asarray(embeddings, dtype=np.float32)
        if (
            matrix.ndim != 2
            or matrix.shape[0] != len(self.identity.document_ids)
            or matrix.shape[1] == 0
        ):
            raise RetrievalIndexError("Dense embeddings do not match the indexed documents")
        if not np.isfinite(matrix).all():
            raise RetrievalIndexError("Dense embeddings contain non-finite values")

        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            **self.identity.as_dict(),
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
        }
        embeddings_temp: Path | None = None
        metadata_temp: Path | None = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.directory, prefix=".dense-", suffix=".npy", delete=False
            ) as file:
                embeddings_temp = Path(file.name)
                np.save(file, matrix, allow_pickle=False)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.directory,
                prefix=".dense-",
                suffix=".json",
                delete=False,
            ) as file:
                metadata_temp = Path(file.name)
                json.dump(metadata, file, indent=2, sort_keys=True)
                file.write("\n")

            os.replace(embeddings_temp, self.embeddings_path)
            embeddings_temp = None
            os.replace(metadata_temp, self.metadata_path)
            metadata_temp = None
        except (OSError, TypeError, ValueError) as error:
            raise RetrievalIndexError(
                f"Could not persist dense index in '{self.directory}': {error}"
            ) from error
        finally:
            if embeddings_temp is not None:
                embeddings_temp.unlink(missing_ok=True)
            if metadata_temp is not None:
                metadata_temp.unlink(missing_ok=True)
