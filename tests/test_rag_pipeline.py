import pytest

from supplyguard.config import DEFAULT_CONFIG, RetrievalConfig, SupplyGuardConfig
from supplyguard.errors import ConfigurationError, ModelGenerationError
from supplyguard.llm.answer_generation import RAGPipeline


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, query, top_k=5):
        self.queries.append((query, top_k))
        return self.results[:top_k]


class FakeLLM:
    def __init__(self, answer="Grounded answer [GHSA-test-0000-0001]"):
        self.answer = answer
        self.calls = []

    def generate(self, prompt, **settings):
        self.calls.append((prompt, settings))
        return self.answer


def retrieval_result(record, score, rank=1):
    return {
        "rank": rank,
        "score": score,
        "document_id": record["document_id"],
        "chunk_id": record["chunk_id"],
        "summary": record["summary"],
        "retrieval_text": record["retrieval_text"],
        "record": record,
    }


def test_generate_answer_uses_filtered_bm25_context(advisory_record_factory):
    record = advisory_record_factory()
    bm25 = FakeRetriever([retrieval_result(record, score=20.0)])
    dense = FakeRetriever([])
    llm = FakeLLM()
    pipeline = RAGPipeline(bm25, dense, llm=llm)

    result = pipeline.generate_answer("What affects example-package?", use_dense=False)

    assert result["answer"] == "Grounded answer"
    assert result["retrieval_method"] == "bm25"
    assert result["num_chunks"] == 1
    assert result["num_sources"] == 1
    assert result["sources"] == [
        {
            "advisory_id": record["document_id"],
            "chunk_id": record["chunk_id"],
            "url": f"https://osv.dev/vulnerability/{record['document_id']}",
            "summary": record["summary"],
            "rank": 1,
            "score": 20.0,
        }
    ]
    assert record["retrieval_text"] in llm.calls[0][0]
    assert llm.calls[0][1] == {"max_tokens": DEFAULT_CONFIG.models.answer_max_tokens}
    assert bm25.queries == [("What affects example-package?", 5)]


def test_unknown_identifier_abstains_without_calling_llm(advisory_record_factory):
    record = advisory_record_factory(document_id="GHSA-known-0000-0001")
    retriever = FakeRetriever([retrieval_result(record, score=20.0)])
    llm = FakeLLM()
    pipeline = RAGPipeline(retriever, retriever, llm=llm)

    result = pipeline.generate_answer("What fixes CVE-2099-99999?", use_dense=False)

    assert result["answer"] == "I don't know"
    assert result["retrieved_chunks"] == []
    assert result["sources"] == []
    assert llm.calls == []


def test_package_mismatch_abstains(advisory_record_factory):
    record = advisory_record_factory(package_name="known-package")
    retriever = FakeRetriever([retrieval_result(record, score=20.0)])
    llm = FakeLLM()
    pipeline = RAGPipeline(retriever, retriever, llm=llm)

    result = pipeline.generate_answer(
        "What affects package missing-package?",
        use_dense=False,
    )

    assert result["answer"] == "I don't know"
    assert llm.calls == []


def test_filter_retrieved_applies_relative_cutoff_and_context_limit(
    advisory_record_factory,
):
    records = [
        advisory_record_factory(document_id=f"GHSA-test-0000-000{index}") for index in range(1, 5)
    ]
    results = [
        retrieval_result(record, score=score, rank=index)
        for index, (record, score) in enumerate(
            zip(records, [20.0, 15.0, 13.0, 12.0], strict=True),
            start=1,
        )
    ]
    retriever = FakeRetriever(results)
    config = SupplyGuardConfig(
        paths=DEFAULT_CONFIG.paths,
        models=DEFAULT_CONFIG.models,
        retrieval=RetrievalConfig(max_contexts=2),
        runtime=DEFAULT_CONFIG.runtime,
    )
    pipeline = RAGPipeline(retriever, retriever, llm=FakeLLM(), config=config)

    filtered = pipeline.filter_retrieved("general question", results, use_dense=False)

    assert [item["score"] for item in filtered] == [20.0, 15.0]


def test_hybrid_retrieval_fuses_filtered_bm25_and_dense_results(advisory_record_factory):
    shared = advisory_record_factory(document_id="GHSA-shared-0000-0001")
    lexical = advisory_record_factory(document_id="GHSA-lexical-0000-0002")
    semantic = advisory_record_factory(document_id="GHSA-semantic-0000-0003")
    bm25 = FakeRetriever(
        [
            retrieval_result(lexical, score=20.0, rank=1),
            retrieval_result(shared, score=15.0, rank=2),
        ]
    )
    dense = FakeRetriever(
        [
            retrieval_result(semantic, score=0.9, rank=1),
            retrieval_result(shared, score=0.8, rank=2),
        ]
    )
    config = SupplyGuardConfig(
        paths=DEFAULT_CONFIG.paths,
        models=DEFAULT_CONFIG.models,
        retrieval=RetrievalConfig(max_contexts=1),
        runtime=DEFAULT_CONFIG.runtime,
    )
    pipeline = RAGPipeline(bm25, dense, llm=FakeLLM(), config=config)

    result = pipeline.generate_answer("What is the fix?", retrieval_method="hybrid")

    assert result["retrieval_method"] == "hybrid"
    assert [item["document_id"] for item in result["retrieved_chunks"]] == ["GHSA-shared-0000-0001"]
    assert result["retrieved_chunks"][0]["retriever_ranks"] == {"bm25": 2, "dense": 2}
    assert bm25.queries == [("What is the fix?", 5)]
    assert dense.queries == [("What is the fix?", 5)]


def test_unknown_retrieval_method_fails_fast():
    pipeline = RAGPipeline(FakeRetriever([]), FakeRetriever([]), llm=FakeLLM())

    with pytest.raises(ConfigurationError, match="Choose bm25, dense, or hybrid"):
        pipeline.generate_answer("question", retrieval_method="unsupported")


def test_model_cannot_cite_an_advisory_that_was_not_retrieved(advisory_record_factory):
    record = advisory_record_factory(document_id="GHSA-retrieved-0000-0001")
    retriever = FakeRetriever([retrieval_result(record, score=20.0)])
    llm = FakeLLM("Use the fixed release [GHSA-hallucinated-0000-9999]")
    pipeline = RAGPipeline(retriever, retriever, llm=llm)

    result = pipeline.generate_answer("What is the fixed release?", use_dense=False)

    assert result["answer"] == "Use the fixed release"
    assert [source["advisory_id"] for source in result["sources"]] == [record["document_id"]]


def test_empty_retrieval_abstains_without_calling_llm():
    retriever = FakeRetriever([])
    llm = FakeLLM()
    pipeline = RAGPipeline(retriever, retriever, llm=llm)

    result = pipeline.generate_answer("What affects missing-package?", use_dense=False)

    assert result["answer"] == "I don't know"
    assert result["retrieved_chunks"] == []
    assert result["sources"] == []
    assert llm.calls == []


def test_model_abstention_has_no_sources_even_when_context_was_retrieved(
    advisory_record_factory,
):
    record = advisory_record_factory()
    retriever = FakeRetriever([retrieval_result(record, score=20.0)])
    pipeline = RAGPipeline(retriever, retriever, llm=FakeLLM("I don't know"))

    result = pipeline.generate_answer("What is the fix?", use_dense=False)

    assert result["answer"] == "I don't know"
    assert result["retrieved_chunks"]
    assert result["sources"] == []


def test_citation_only_model_output_is_rejected(advisory_record_factory):
    record = advisory_record_factory()
    retriever = FakeRetriever([retrieval_result(record, score=20.0)])
    pipeline = RAGPipeline(
        retriever,
        retriever,
        llm=FakeLLM("[GHSA-hallucinated-0000-9999]"),
    )

    with pytest.raises(ModelGenerationError, match="no answer"):
        pipeline.generate_answer("What is the fix?", use_dense=False)


def test_generation_errors_are_not_converted_to_empty_answers(advisory_record_factory):
    class FailingLLM:
        def generate(self, prompt, **settings):
            raise ModelGenerationError("generation failed")

    record = advisory_record_factory()
    retriever = FakeRetriever([retrieval_result(record, score=20.0)])
    pipeline = RAGPipeline(retriever, retriever, llm=FailingLLM())

    with pytest.raises(ModelGenerationError, match="generation failed"):
        pipeline.generate_answer("What affects example-package?", use_dense=False)
