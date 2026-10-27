import re

from supplyguard.config import DEFAULT_CONFIG, SupplyGuardConfig
from supplyguard.errors import ConfigurationError, ModelGenerationError
from supplyguard.llm.prompt_template import create_rag_prompt
from supplyguard.retrieval.hybrid import reciprocal_rank_fusion
from supplyguard.sources import build_advisory_sources, is_abstention, remove_inline_citations


class RAGPipeline:
    def __init__(
        self,
        bm25_retriever,
        dense_retriever,
        llm=None,
        config: SupplyGuardConfig = DEFAULT_CONFIG,
    ):
        """Initialize RAG pipeline"""
        self.config = config
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        if llm is None:
            from supplyguard.llm.llm_setup import HuggingFaceLLM

            llm = HuggingFaceLLM(config.models)
        self.llm = llm

    def filter_retrieved(self, question, retrieved, use_dense, result_limit=None):
        if not retrieved:
            return []

        searchable_text = "\n".join(
            result["record"]["retrieval_text"].lower() for result in retrieved
        )
        identifiers = re.findall(r"(?:CVE|GHSA|GO)-[A-Za-z0-9-]+", question, re.IGNORECASE)
        if identifiers and any(
            identifier.lower() not in searchable_text for identifier in identifiers
        ):
            return []

        package_match = re.search(r"\bpackage\s+([@A-Za-z0-9._/:+-]+)", question, re.IGNORECASE)
        if package_match:
            package_name = package_match.group(1).lower()
            package_names = {
                package["name"].lower()
                for result in retrieved
                for package in result["record"].get("packages", [])
            }
            if package_name not in package_names:
                return []

        retrieval_config = self.config.retrieval
        minimum_score = (
            retrieval_config.dense_min_score if use_dense else retrieval_config.bm25_min_score
        )
        relative_cutoff = (
            retrieval_config.dense_relative_cutoff
            if use_dense
            else retrieval_config.bm25_relative_cutoff
        )
        top_score = retrieved[0]["score"]
        score_cutoff = max(minimum_score, top_score * relative_cutoff)
        if result_limit is None:
            result_limit = retrieval_config.max_contexts
        return [result for result in retrieved if result["score"] >= score_cutoff][:result_limit]

    def retrieve(self, question, retrieval_method, top_k):
        """Retrieve and filter context with one strategy or rank fusion."""
        if retrieval_method == "bm25":
            retrieved = self.bm25_retriever.search(question, top_k=top_k)
            return self.filter_retrieved(question, retrieved, use_dense=False)

        if retrieval_method == "dense":
            retrieved = self.dense_retriever.search(question, top_k=top_k)
            return self.filter_retrieved(question, retrieved, use_dense=True)

        if retrieval_method == "hybrid":
            bm25_results = self.bm25_retriever.search(question, top_k=top_k)
            dense_results = self.dense_retriever.search(question, top_k=top_k)
            filtered_bm25 = self.filter_retrieved(
                question,
                bm25_results,
                use_dense=False,
                result_limit=top_k,
            )
            filtered_dense = self.filter_retrieved(
                question,
                dense_results,
                use_dense=True,
                result_limit=top_k,
            )
            return reciprocal_rank_fusion(
                filtered_bm25,
                filtered_dense,
                top_k=self.config.retrieval.max_contexts,
                rrf_k=self.config.retrieval.rrf_k,
            )

        raise ConfigurationError(
            f"Unknown retrieval method '{retrieval_method}'. Choose bm25, dense, or hybrid"
        )

    def generate_answer(
        self,
        question,
        use_dense=None,
        top_k=None,
        *,
        retrieval_method=None,
    ):
        """Generate answer using retrieved context"""
        if top_k is None:
            top_k = self.config.retrieval.top_k

        if retrieval_method is None:
            retrieval_method = "bm25" if use_dense is False else "dense"
        retrieved = self.retrieve(question, retrieval_method, top_k)

        retrieved_chunks = [r["record"] for r in retrieved]

        if retrieved_chunks:
            prompt = create_rag_prompt(question, retrieved_chunks)
            model_answer = self.llm.generate(
                prompt,
                max_tokens=self.config.models.answer_max_tokens,
            )
            answer = remove_inline_citations(model_answer)
            if not answer:
                raise ModelGenerationError(
                    "The language model returned no answer after untrusted citations were removed"
                )
            sources = [] if is_abstention(answer) else build_advisory_sources(retrieved)
        else:
            answer = "I don't know"
            sources = []

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved,
            "retrieval_method": retrieval_method,
            "num_chunks": len(retrieved),
            "num_sources": len(sources),
        }


def generate_with_both_retrievers(question, bm25_retriever, dense_retriever, llm):
    """Generate answers with both retrieval methods for comparison"""

    pipeline = RAGPipeline(bm25_retriever, dense_retriever, llm)

    bm25_result = pipeline.generate_answer(question, use_dense=False)
    dense_result = pipeline.generate_answer(question, use_dense=True)

    return {"question": question, "bm25": bm25_result, "dense": dense_result}
