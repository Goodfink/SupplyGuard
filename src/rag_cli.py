import json
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
EVAL_DIR = PROJECT_ROOT / "src" / "evaluation"
RESULTS_FILE = PROJECT_ROOT / "evaluation_results.json"

def format_optional_percentage(value):
    """Format percentage metrics that may be inapplicable."""
    return "N/A" if value is None else f"{value:.1%}"

def print_evaluation_summary(label, stats):
    """Print answer accuracy broken down by question type."""
    factoid = stats["breakdown"]["factoid"]
    multi_hop = stats["breakdown"]["multi_hop"]
    unanswerable = stats["breakdown"]["unanswerable"]

    print(f"\n{label}:")
    print(f"  Overall Answer Accuracy: {stats['accuracy']:.1%} ({stats['correct_answers']}/{stats['total_questions']})")
    print(f"  Factoid Answer Accuracy: {factoid['accuracy']:.1%} ({factoid['correct']}/{factoid['count']})")
    if multi_hop["count"] > 0:
        print(f"  Multi-hop Answer Accuracy: {multi_hop['accuracy']:.1%} ({multi_hop['correct']}/{multi_hop['count']})")
    if unanswerable["count"] > 0:
        print(f"  Abstention Accuracy: {stats['idontknow_accuracy']:.1%} ({unanswerable['correct']}/{unanswerable['count']})")
    print(f"  Citation Accuracy (exact advisory IDs): {format_optional_percentage(stats['citation_accuracy'])}")

def load_evaluation_set():
    """Load evaluation questions"""
    eval_file = EVAL_DIR / "evaluation_set.json"
    with open(eval_file, "r") as f:
        return json.load(f)

def setup_system():
    """Initialize retrieval and LLM systems"""
    from src.llm.llm_setup import HuggingFaceLLM
    from src.retrieval.bm25 import BM25Retriever, load_corpus
    from src.retrieval.dense_retrieval import DenseRetriever

    # Load corpus
    corpus_path = DATA_DIR / "advisories.jsonl"
    records = load_corpus(corpus_path)

    # Setup BM25
    print("\n2. Setting up BM25 retriever...")
    bm25 = BM25Retriever()
    bm25.build_index(records)

    # Setup Dense Retriever
    print("\n3. Setting up Dense retriever...")
    dense = DenseRetriever()
    dense.build_index(records)

    # Setup LLM
    print("\n4. Loading LLM (this takes 2-3 minutes on first run)...")
    try:
        llm = HuggingFaceLLM()
    except Exception as e:
        print(f"   ✗ LLM Error: {e}")
        return None, None, None, None
    
    return bm25, dense, llm, records

def run_single_query(question, bm25, dense, llm):
    """Run a single query with both retrievers."""
    from src.llm.answer_generation import RAGPipeline

    print(f"\nQuestion: {question}\n")
    
    pipeline = RAGPipeline(bm25, dense, llm)
    
    # BM25 result
    print("BM25 Results:")
    bm25_result = pipeline.generate_answer(question, use_dense=False, top_k=5)
    print(f"  Top chunk: {bm25_result['retrieved_chunks'][0]['document_id']}")
    print(f"  Score: {bm25_result['retrieved_chunks'][0]['score']:.4f}")
    
    # Dense result
    print("\nDense Retrieval Results:")
    dense_result = pipeline.generate_answer(question, use_dense=True, top_k=5)
    print(f"  Top chunk: {dense_result['retrieved_chunks'][0]['document_id']}")
    print(f"  Score: {dense_result['retrieved_chunks'][0]['score']:.4f}")
    
    # Show answers
    print("\nBM25 Answer:")
    print(f"  {bm25_result['answer'][:200]}...")
    
    print("\nDense Answer:")
    print(f"  {dense_result['answer'][:200]}...")

def run_full_evaluation(bm25, dense, llm):
    """Run evaluation on all test questions"""
    from src.evaluation.evaluation_metrics import EvaluationMetrics
    from src.llm.answer_generation import generate_with_both_retrievers

    
    evaluation_set = load_evaluation_set()
    metrics = EvaluationMetrics()
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            saved_results = {item["question_id"]: item for item in json.load(f)}
    else:
        saved_results = {}
    
    bm25_results = []
    dense_results = []
    
    for i, question_item in enumerate(evaluation_set, 1):
        print(f"\n[{i}/{len(evaluation_set)}] {question_item['question'][:60]}...")
        
        # Generate with both methods
        result = generate_with_both_retrievers(
            question_item['question'],
            bm25, dense, llm
        )
        
        # Evaluate both
        bm25_eval = metrics.evaluate_answer(
            result['bm25']['answer'],
            question_item['ground_truth_answer'],
            question_item['question_type'],
            question_item['ground_truth_chunk_ids'],
            question_item.get('required_answer_terms'),
            question_item.get('forbidden_answer_terms')
        )
        
        dense_eval = metrics.evaluate_answer(
            result['dense']['answer'],
            question_item['ground_truth_answer'],
            question_item['question_type'],
            question_item['ground_truth_chunk_ids'],
            question_item.get('required_answer_terms'),
            question_item.get('forbidden_answer_terms')
        )
        
        bm25_results.append(bm25_eval)
        dense_results.append(dense_eval)

        saved_result = saved_results.get(question_item["question_id"], {})
        saved_result.update({
            "question_id": question_item["question_id"],
            "question": question_item["question"],
            "question_type": question_item["question_type"],
            "bm25": {
                "answer": result["bm25"]["answer"],
                "correct": bm25_eval["correct"],
                "citations_accurate": bm25_eval["citations_accurate"],
                "retrieved_document_ids": [item["document_id"] for item in result["bm25"]["retrieved_chunks"]]
            },
            "dense": {
                "answer": result["dense"]["answer"],
                "correct": dense_eval["correct"],
                "citations_accurate": dense_eval["citations_accurate"],
                "retrieved_document_ids": [item["document_id"] for item in result["dense"]["retrieved_chunks"]]
            }
        })
        saved_results[question_item["question_id"]] = saved_result
        
        # Print status
        bm25_status = "✓" if bm25_eval["correct"] else "✗"
        dense_status = "✓" if dense_eval["correct"] else "✗"
        print(f"  BM25: {bm25_status} | Dense: {dense_status}")

    ordered_results = [saved_results[item["question_id"]] for item in evaluation_set]
    with open(RESULTS_FILE, "w") as f:
        json.dump(ordered_results, f, indent=2)
    
    bm25_stats = metrics.aggregate_results(bm25_results)
    dense_stats = metrics.aggregate_results(dense_results)
    
    print_evaluation_summary("BM25 Results", bm25_stats)
    print_evaluation_summary("Dense Retrieval Results", dense_stats)
    
    # Comparison
    print(f"\nComparison:")
    diff = (dense_stats['accuracy'] - bm25_stats['accuracy']) * 100
    print(f"  Dense vs BM25: {diff:+.1f}% accuracy")
    print(f"  Results saved to: {RESULTS_FILE}")
    
    return bm25_results, dense_results

def interactive_demo(bm25, dense, llm):
    """Interactive Q&A demo"""
    from src.llm.answer_generation import RAGPipeline

    print("Ask questions about vulnerabilities. Type 'exit' to quit.\n")
    
    pipeline = RAGPipeline(bm25, dense, llm)
    
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == 'exit':
            break
        
        if not question:
            continue
        
        # Generate answer with dense retrieval
        result = pipeline.generate_answer(question, use_dense=True, top_k=5)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources:")
        for chunk in result['retrieved_chunks'][:3]:
            print(f"  - {chunk['document_id']}: {chunk['summary']}")

def main():
    parser = argparse.ArgumentParser(description='Run the SupplyGuard RAG system')
    parser.add_argument('--test-setup', action='store_true', help='Test system setup only')
    parser.add_argument('--query', type=str, help='Test single query')
    parser.add_argument('--evaluate', choices=['all', 'sample'], help='Run evaluation')
    parser.add_argument('--demo', action='store_true', help='Interactive demo')
    
    args = parser.parse_args()
    
    # Setup system
    bm25, dense, llm, records = setup_system()
    
    if bm25 is None:
        sys.exit(1)
    
    # Test setup only
    if args.test_setup:
        return
    
    # Test single query
    if args.query:
        run_single_query(args.query, bm25, dense, llm)
        return
    
    # Run evaluation
    if args.evaluate == 'all':
        run_full_evaluation(bm25, dense, llm)
        return
    
    # Interactive demo
    if args.demo:
        interactive_demo(bm25, dense, llm)
        return
    
    # Default: show help
    parser.print_help()

if __name__ == "__main__":
    main()
