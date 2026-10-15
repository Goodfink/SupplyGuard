def create_rag_prompt(question, retrieved_chunks):
    """Create a RAG prompt that forces grounding in retrieved context"""
    
    context_parts = []
    for chunk in retrieved_chunks:
        text = chunk['retrieval_text']
        context_parts.append(f"[{chunk['document_id']}]\n{text}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""Answer using only the provided advisories.

If not found, say: "I don't know"

Always cite the advisory ID, e.g., [GHSA-12345]

ADVISORIES:
{context}

QUESTION: {question}

ANSWER:"""
    
    return prompt

def create_eval_prompt(question, retrieved_chunks):
    """Same as RAG prompt (used for evaluation)"""
    return create_rag_prompt(question, retrieved_chunks)