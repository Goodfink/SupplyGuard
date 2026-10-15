from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from supplyguard.config import DEFAULT_CONFIG, ModelConfig
from supplyguard.errors import (
    ConfigurationError,
    ModelInferenceError,
    ModelLoadError,
    RetrievalIndexError,
)


class DenseRetriever:
    def __init__(self, config: ModelConfig = DEFAULT_CONFIG.models):
        """Initialize the dense retriever with the configured embedding model."""
        self.config = config
        try:
            self.model = SentenceTransformer(
                config.embedding_model_name,
                revision=config.embedding_model_revision,
                device=config.device,
            )
        except Exception as error:
            raise ModelLoadError(
                f"Could not load embedding model '{config.embedding_model_name}' at revision "
                f"'{config.embedding_model_revision}'. Check the model cache and network access. "
                f"Underlying error: {error}"
            ) from error
        self.model.eval()
        self.records: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.document_ids: list[str] = []

    def build_index(self, records):
        """Build embedding index from records"""
        if not records:
            raise RetrievalIndexError("Cannot build the dense index from an empty corpus")

        self.records = records
        self.document_ids = []
        texts_to_embed = []

        for record in records:
            self.document_ids.append(record["document_id"])
            # Use retrieval_text for embeddings (same as BM25)
            texts_to_embed.append(record.get("retrieval_text", ""))

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
        self.embeddings = embeddings
        print(f"Index built with {len(embeddings)} embeddings")

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
