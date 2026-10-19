from pathlib import Path
from typing import Any

import numpy as np

from supplyguard.config import DEFAULT_CONFIG, ModelConfig
from supplyguard.errors import (
    ConfigurationError,
    ModelInferenceError,
    ModelLoadError,
    RetrievalIndexError,
)
from supplyguard.retrieval.index_cache import DenseIndexCache, DenseIndexIdentity, hash_file


class DenseRetriever:
    def __init__(self, config: ModelConfig = DEFAULT_CONFIG.models, model=None):
        """Initialize the dense retriever with the configured embedding model."""
        self.config = config
        self.model = model if model is not None else self._load_model()
        self.model.eval()
        self.records: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.document_ids: list[str] = []
        self.loaded_from_cache = False

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(
                self.config.embedding_model_name,
                revision=self.config.embedding_model_revision,
                device=self.config.device,
            )
        except Exception as error:
            raise ModelLoadError(
                f"Could not load embedding model '{self.config.embedding_model_name}' at revision "
                f"'{self.config.embedding_model_revision}'. Check the model cache and network "
                f"access. Underlying error: {error}"
            ) from error

    def _prepare_records(self, records):
        if not records:
            raise RetrievalIndexError("Cannot build the dense index from an empty corpus")

        self.records = records
        self.document_ids = [record["document_id"] for record in records]
        return [record.get("retrieval_text", "") for record in records]

    def _encode_documents(self, texts_to_embed):
        print(f"Encoding {len(texts_to_embed)} documents...")
        try:
            embeddings = self.model.encode(
                texts_to_embed,
                batch_size=self.config.embedding_batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                precision="float32",
                normalize_embeddings=True,
            )
        except Exception as error:
            raise ModelInferenceError(f"Could not encode the advisory corpus: {error}") from error

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts_to_embed):
            raise RetrievalIndexError("Embedding model returned an invalid dense index shape")
        if not np.isfinite(matrix).all():
            raise RetrievalIndexError("Embedding model returned non-finite values")
        return matrix

    def build_index(self, records):
        """Build an in-memory embedding index without using the persistent cache."""
        texts_to_embed = self._prepare_records(records)
        self.embeddings = self._encode_documents(texts_to_embed)
        self.loaded_from_cache = False
        print(f"Index built with {len(self.embeddings)} embeddings")

    def prepare_index(
        self,
        records,
        *,
        corpus_path: Path,
        cache_dir: Path,
        force_rebuild: bool = False,
    ) -> bool:
        """Load a valid cached index or build and persist a replacement.

        Returns ``True`` when the index was loaded from cache.
        """
        texts_to_embed = self._prepare_records(records)
        identity = DenseIndexIdentity(
            corpus_sha256=hash_file(corpus_path),
            model_name=self.config.embedding_model_name,
            model_revision=self.config.embedding_model_revision,
            document_ids=tuple(self.document_ids),
        )
        cache = DenseIndexCache(cache_dir, identity)

        if not force_rebuild:
            cached_embeddings = cache.load()
            if cached_embeddings is not None:
                self.embeddings = cached_embeddings
                self.loaded_from_cache = True
                print(f"Loaded {len(cached_embeddings)} embeddings from {cache.embeddings_path}")
                return True

        self.embeddings = self._encode_documents(texts_to_embed)
        cache.save(self.embeddings)
        self.loaded_from_cache = False
        print(f"Index built with {len(self.embeddings)} embeddings and cached in {cache_dir}")
        return False

    def search(self, query, top_k=DEFAULT_CONFIG.retrieval.top_k):
        """Search using semantic similarity"""
        if self.embeddings is None:
            raise RetrievalIndexError("Build the dense index before searching")

        if top_k <= 0:
            raise ConfigurationError("top_k must be greater than 0")

        # Encode the query
        try:
            query_embedding = self.model.encode(
                query, convert_to_numpy=True, precision="float32", normalize_embeddings=True
            )
        except Exception as error:
            raise ModelInferenceError(f"Could not encode the search query: {error}") from error

        # Compute similarity (cosine similarity via dot product on normalized vectors)
        similarities = np.dot(self.embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(similarities[idx])
            record = self.records[idx]

            results.append(
                {
                    "rank": rank,
                    "score": score,
                    "document_id": record["document_id"],
                    "chunk_id": record["chunk_id"],
                    "summary": record["summary"],
                    "retrieval_text": record["retrieval_text"],
                    "record": record,
                }
            )

        return results
