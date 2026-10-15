from src.llm.answer_generation import RAGPipeline


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

    assert result["answer"] == llm.answer
    assert result["retrieval_method"] == "bm25"
    assert result["num_chunks"] == 1
    assert record["retrieval_text"] in llm.calls[0][0]
    assert bm25.queries == [("What affects example-package?", 5)]


def test_unknown_identifier_abstains_without_calling_llm(advisory_record_factory):
    record = advisory_record_factory(document_id="GHSA-known-0000-0001")
    retriever = FakeRetriever([retrieval_result(record, score=20.0)])
    llm = FakeLLM()
    pipeline = RAGPipeline(retriever, retriever, llm=llm)

    result = pipeline.generate_answer("What fixes CVE-2099-99999?", use_dense=False)

    assert result["answer"] == "I don't know"
    assert result["retrieved_chunks"] == []
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
        advisory_record_factory(document_id=f"GHSA-test-0000-000{index}")
        for index in range(1, 5)
    ]
    results = [
        retrieval_result(record, score=score, rank=index)
        for index, (record, score) in enumerate(
            zip(records, [20.0, 15.0, 13.0, 12.0]),
            start=1,
        )
    ]
    retriever = FakeRetriever(results)
    pipeline = RAGPipeline(retriever, retriever, llm=FakeLLM(), max_contexts=2)

    filtered = pipeline.filter_retrieved("general question", results, use_dense=False)

    assert [item["score"] for item in filtered] == [20.0, 15.0]


def test_extract_citations_deduplicates_supported_identifier_types():
    pipeline = RAGPipeline(FakeRetriever([]), FakeRetriever([]), llm=FakeLLM())

    citations = pipeline.extract_citations(
        "[GHSA-abcd-1234-5678] [CVE-2025-1234] "
        "[GHSA-abcd-1234-5678] [GO-2025-1000]"
    )

    assert set(citations) == {
        "GHSA-abcd-1234-5678",
        "CVE-2025-1234",
        "GO-2025-1000",
    }
