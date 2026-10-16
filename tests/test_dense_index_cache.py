import json
from dataclasses import replace

import numpy as np
import pytest

from supplyguard.config import ModelConfig
from supplyguard.retrieval.dense_retrieval import DenseRetriever


class FakeEmbeddingModel:
    def __init__(self):
        self.document_encode_calls = 0
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def encode(self, inputs, **_options):
        if isinstance(inputs, str):
            return self._vector(inputs)
        self.document_encode_calls += 1
        return np.vstack([self._vector(text) for text in inputs])

    @staticmethod
    def _vector(text):
        lowered = text.lower()
        vector = np.array(
            [
                lowered.count("alpha") + 1,
                lowered.count("beta") + 1,
                lowered.count("gamma") + 1,
            ],
            dtype=np.float32,
        )
        return vector / np.linalg.norm(vector)


@pytest.fixture
def records():
    return [
        {
            "document_id": "OSV-ALPHA",
            "chunk_id": "OSV-ALPHA-0",
            "summary": "Alpha advisory",
            "retrieval_text": "alpha alpha package",
        },
        {
            "document_id": "OSV-BETA",
            "chunk_id": "OSV-BETA-0",
            "summary": "Beta advisory",
            "retrieval_text": "beta beta package",
        },
    ]


def write_corpus(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def make_retriever(model_config=None):
    model = FakeEmbeddingModel()
    retriever = DenseRetriever(model_config or ModelConfig(), model=model)
    return retriever, model


def test_second_startup_reuses_cache_without_encoding_documents(tmp_path, records):
    corpus_path = tmp_path / "advisories.jsonl"
    cache_dir = tmp_path / "indexes"
    write_corpus(corpus_path, records)

    first, first_model = make_retriever()
    assert first.prepare_index(records, corpus_path=corpus_path, cache_dir=cache_dir) is False
    assert first_model.document_encode_calls == 1

    second, second_model = make_retriever()
    assert second.prepare_index(records, corpus_path=corpus_path, cache_dir=cache_dir) is True
    assert second.loaded_from_cache is True
    assert second_model.document_encode_calls == 0


def test_corpus_change_invalidates_cached_index(tmp_path, records):
    corpus_path = tmp_path / "advisories.jsonl"
    cache_dir = tmp_path / "indexes"
    write_corpus(corpus_path, records)
    first, _ = make_retriever()
    first.prepare_index(records, corpus_path=corpus_path, cache_dir=cache_dir)

    corpus_path.write_text(corpus_path.read_text() + "\n", encoding="utf-8")
    second, second_model = make_retriever()

    assert second.prepare_index(records, corpus_path=corpus_path, cache_dir=cache_dir) is False
    assert second_model.document_encode_calls == 1


def test_model_revision_change_invalidates_cached_index(tmp_path, records):
    corpus_path = tmp_path / "advisories.jsonl"
    cache_dir = tmp_path / "indexes"
    write_corpus(corpus_path, records)
    config = ModelConfig()
    first, _ = make_retriever(config)
    first.prepare_index(records, corpus_path=corpus_path, cache_dir=cache_dir)

    changed_config = replace(config, embedding_model_revision="different-revision")
    second, second_model = make_retriever(changed_config)

    assert second.prepare_index(records, corpus_path=corpus_path, cache_dir=cache_dir) is False
    assert second_model.document_encode_calls == 1


def test_cached_and_fresh_indexes_return_equivalent_rankings(tmp_path, records):
    corpus_path = tmp_path / "advisories.jsonl"
    cache_dir = tmp_path / "indexes"
    write_corpus(corpus_path, records)
    fresh, _ = make_retriever()
    fresh.prepare_index(records, corpus_path=corpus_path, cache_dir=cache_dir)

    cached, _ = make_retriever()
    cached.prepare_index(records, corpus_path=corpus_path, cache_dir=cache_dir)

    fresh_results = fresh.search("alpha", top_k=2)
    cached_results = cached.search("alpha", top_k=2)
    assert [result["document_id"] for result in cached_results] == [
        result["document_id"] for result in fresh_results
    ]
    assert [result["score"] for result in cached_results] == pytest.approx(
        [result["score"] for result in fresh_results]
    )


def test_corrupt_cache_is_rebuilt_automatically(tmp_path, records):
    corpus_path = tmp_path / "advisories.jsonl"
    cache_dir = tmp_path / "indexes"
    write_corpus(corpus_path, records)
    first, _ = make_retriever()
    first.prepare_index(records, corpus_path=corpus_path, cache_dir=cache_dir)
    next(cache_dir.glob("*.npy")).write_bytes(b"not a numpy array")

    second, second_model = make_retriever()

    assert second.prepare_index(records, corpus_path=corpus_path, cache_dir=cache_dir) is False
    assert second_model.document_encode_calls == 1


def test_force_rebuild_bypasses_valid_cache(tmp_path, records):
    corpus_path = tmp_path / "advisories.jsonl"
    cache_dir = tmp_path / "indexes"
    write_corpus(corpus_path, records)
    first, _ = make_retriever()
    first.prepare_index(records, corpus_path=corpus_path, cache_dir=cache_dir)

    second, second_model = make_retriever()

    assert (
        second.prepare_index(
            records,
            corpus_path=corpus_path,
            cache_dir=cache_dir,
            force_rebuild=True,
        )
        is False
    )
    assert second_model.document_encode_calls == 1


def test_cache_metadata_records_reproducibility_inputs(tmp_path, records):
    corpus_path = tmp_path / "advisories.jsonl"
    cache_dir = tmp_path / "indexes"
    write_corpus(corpus_path, records)
    retriever, _ = make_retriever()
    retriever.prepare_index(records, corpus_path=corpus_path, cache_dir=cache_dir)

    metadata = json.loads(next(cache_dir.glob("*.json")).read_text(encoding="utf-8"))

    assert len(metadata["corpus_sha256"]) == 64
    assert metadata["model_name"] == ModelConfig().embedding_model_name
    assert metadata["model_revision"] == ModelConfig().embedding_model_revision
    assert metadata["document_ids"] == ["OSV-ALPHA", "OSV-BETA"]
    assert metadata["dtype"] == "float32"
