import json

import numpy as np
from sentence_transformers import SentenceTransformer

from supplyguard.reproducibility import DEVICE

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class DenseRetriever:
    def __init__(self, model_name=MODEL_NAME, model_revision=MODEL_REVISION):
        """Initialize dense retriever with embedding model"""
        self.model = SentenceTransformer(model_name, revision=model_revision, device=DEVICE)
        self.model.eval()
        self.model_name = model_name
        self.model_revision = model_revision
        self.records = []
        self.embeddings = None
        self.document_ids = []

    def build_index(self, records):
        """Build embedding index from records"""
        if not records:
            raise ValueError("No records to index")

        self.records = records
        self.document_ids = []
        texts_to_embed = []

        for record in records:
            self.document_ids.append(record["document_id"])
            # Use retrieval_text for embeddings (same as BM25)
            texts_to_embed.append(record.get("retrieval_text", ""))

        print(f"Encoding {len(texts_to_embed)} documents...")
        self.embeddings = self.model.encode(
            texts_to_embed,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            precision="float32",
            normalize_embeddings=True,
        )
        print(f"Index built with {len(self.embeddings)} embeddings")

    def search(self, query, top_k=5):
        """Search using semantic similarity"""
        if self.embeddings is None:
            raise ValueError("Build the index before searching")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        # Encode the query
        query_embedding = self.model.encode(
            query, convert_to_numpy=True, precision="float32", normalize_embeddings=True
        )

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


def load_corpus(corpus_path):
    """Load JSONL corpus"""
    records = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"Error decoding line: {e}")
                continue
    return records
