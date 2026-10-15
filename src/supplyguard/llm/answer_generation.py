import re

from supplyguard.llm.prompt_template import create_rag_prompt

class RAGPipeline:
    def __init__(self, bm25_retriever, dense_retriever, llm=None, bm25_min_score=10.0, dense_min_score=0.44, max_contexts=3):
        """Initialize RAG pipeline"""
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        if llm is None:
            from supplyguard.llm.llm_setup import HuggingFaceLLM

            llm = HuggingFaceLLM()
        self.llm = llm
        self.bm25_min_score = bm25_min_score
        self.dense_min_score = dense_min_score
        self.max_contexts = max_contexts

    def filter_retrieved(self, question, retrieved, use_dense):
        if not retrieved:
            return []

        searchable_text = "\n".join(result["record"]["retrieval_text"].lower() for result in retrieved)
        identifiers = re.findall(r"(?:CVE|GHSA|GO)-[A-Za-z0-9-]+", question, re.IGNORECASE)
        if identifiers and any(identifier.lower() not in searchable_text for identifier in identifiers):
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

        minimum_score = self.dense_min_score if use_dense else self.bm25_min_score
        relative_cutoff = 0.70 if use_dense else 0.65
        top_score = retrieved[0]["score"]
        score_cutoff = max(minimum_score, top_score * relative_cutoff)
        return [result for result in retrieved if result["score"] >= score_cutoff][:self.max_contexts]
    
    def generate_answer(self, question, use_dense=True, top_k=5):
        """Generate answer using retrieved context"""
        
        # Retrieve context
        if use_dense:
            retrieved = self.dense_retriever.search(question, top_k=top_k)
            retrieval_method = "dense"
        else:
            retrieved = self.bm25_retriever.search(question, top_k=top_k)
            retrieval_method = "bm25"

        retrieved = self.filter_retrieved(question, retrieved, use_dense)
        
        # Get full records for context
        retrieved_chunks = [r['record'] for r in retrieved]
        
        if retrieved_chunks:
            prompt = create_rag_prompt(question, retrieved_chunks)
            answer = self.llm.generate(prompt, temperature=0.3, max_tokens=300)
        else:
            answer = "I don't know"
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved,
            "retrieval_method": retrieval_method,
            "num_chunks": len(retrieved)
        }
    
    def extract_citations(self, answer_text):
        """Extract citation IDs from answer"""
        # Find all [GHSA-...] or [GO-...] patterns
        citations = re.findall(r'\[(GHSA-[^\]]+|GO-[^\]]+|CVE-[^\]]+)\]', answer_text)
        return list(set(citations))  # Remove duplicates

def generate_with_both_retrievers(question, bm25_retriever, dense_retriever, llm):
    """Generate answers with both retrieval methods for comparison"""
    
    pipeline = RAGPipeline(bm25_retriever, dense_retriever, llm)
    
    bm25_result = pipeline.generate_answer(question, use_dense=False, top_k=5)
    dense_result = pipeline.generate_answer(question, use_dense=True, top_k=5)
    
    return {
        "question": question,
        "bm25": bm25_result,
        "dense": dense_result
    }
