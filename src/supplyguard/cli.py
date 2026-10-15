import argparse
import json
from pathlib import Path

from supplyguard.paths import (
    DEFAULT_CORPUS_PATH,
    DEFAULT_RESULTS_FILE,
    EVALUATION_FILE,
)


def format_optional_percentage(value):
    """Format percentage metrics that may be inapplicable."""
    return "N/A" if value is None else f"{value:.1%}"


def print_evaluation_summary(label, stats):
    """Print answer accuracy broken down by question type."""
    factoid = stats["breakdown"]["factoid"]
    multi_hop = stats["breakdown"]["multi_hop"]
    unanswerable = stats["breakdown"]["unanswerable"]

    print(f"\n{label}:")
    print(
        f"  Overall Answer Accuracy: {stats['accuracy']:.1%} "
        f"({stats['correct_answers']}/{stats['total_questions']})"
    )
    print(
        f"  Factoid Answer Accuracy: {factoid['accuracy']:.1%} "
        f"({factoid['correct']}/{factoid['count']})"
    )
    if multi_hop["count"] > 0:
        print(
            f"  Multi-hop Answer Accuracy: {multi_hop['accuracy']:.1%} "
            f"({multi_hop['correct']}/{multi_hop['count']})"
        )
    if unanswerable["count"] > 0:
        print(
            f"  Abstention Accuracy: {stats['idontknow_accuracy']:.1%} "
            f"({unanswerable['correct']}/{unanswerable['count']})"
        )
    print(
        "  Citation Accuracy (exact advisory IDs): "
        f"{format_optional_percentage(stats['citation_accuracy'])}"
    )


def load_evaluation_set():
    """Load evaluation questions bundled with SupplyGuard."""
    with EVALUATION_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def setup_system(corpus_path=DEFAULT_CORPUS_PATH):
    """Initialize retrieval and LLM systems."""
    from supplyguard.llm.llm_setup import HuggingFaceLLM
    from supplyguard.retrieval.bm25 import BM25Retriever, load_corpus
    from supplyguard.retrieval.dense_retrieval import DenseRetriever

    records = load_corpus(corpus_path)

    print("\n1. Setting up BM25 retriever...")
    bm25 = BM25Retriever()
    bm25.build_index(records)

    print("\n2. Setting up dense retriever...")
    dense = DenseRetriever()
    dense.build_index(records)

    print("\n3. Loading LLM (this takes 2-3 minutes on first run)...")
    try:
        llm = HuggingFaceLLM()
    except Exception as error:
        print(f"   ✗ LLM Error: {error}")
        return None, None, None, None

    return bm25, dense, llm, records


def run_single_query(question, bm25, dense, llm):
    """Run a single query with both retrievers."""
    from supplyguard.llm.answer_generation import RAGPipeline

    print(f"\nQuestion: {question}\n")
    pipeline = RAGPipeline(bm25, dense, llm)

    print("BM25 Results:")
    bm25_result = pipeline.generate_answer(question, use_dense=False, top_k=5)
    if bm25_result["retrieved_chunks"]:
        print(f"  Top chunk: {bm25_result['retrieved_chunks'][0]['document_id']}")
        print(f"  Score: {bm25_result['retrieved_chunks'][0]['score']:.4f}")
    else:
        print("  No matching advisory found")

    print("\nDense Retrieval Results:")
    dense_result = pipeline.generate_answer(question, use_dense=True, top_k=5)
    if dense_result["retrieved_chunks"]:
        print(f"  Top chunk: {dense_result['retrieved_chunks'][0]['document_id']}")
        print(f"  Score: {dense_result['retrieved_chunks'][0]['score']:.4f}")
    else:
        print("  No matching advisory found")

    print("\nBM25 Answer:")
    print(f"  {bm25_result['answer']}")
    print("\nDense Answer:")
    print(f"  {dense_result['answer']}")


def run_full_evaluation(bm25, dense, llm, results_file=DEFAULT_RESULTS_FILE):
    """Run evaluation on all test questions."""
    from supplyguard.evaluation.evaluation_metrics import EvaluationMetrics
    from supplyguard.llm.answer_generation import generate_with_both_retrievers

    evaluation_set = load_evaluation_set()
    metrics = EvaluationMetrics()
    if results_file.exists():
        with results_file.open(encoding="utf-8") as file:
            saved_results = {
                item["question_id"]: item for item in json.load(file)
            }
    else:
        saved_results = {}

    bm25_results = []
    dense_results = []
    for index, question_item in enumerate(evaluation_set, 1):
        print(
            f"\n[{index}/{len(evaluation_set)}] "
            f"{question_item['question'][:60]}..."
        )
        result = generate_with_both_retrievers(
            question_item["question"], bm25, dense, llm
        )

        bm25_eval = metrics.evaluate_answer(
            result["bm25"]["answer"],
            question_item["ground_truth_answer"],
            question_item["question_type"],
            question_item["ground_truth_chunk_ids"],
            question_item.get("required_answer_terms"),
            question_item.get("forbidden_answer_terms"),
        )
        dense_eval = metrics.evaluate_answer(
            result["dense"]["answer"],
            question_item["ground_truth_answer"],
            question_item["question_type"],
            question_item["ground_truth_chunk_ids"],
            question_item.get("required_answer_terms"),
            question_item.get("forbidden_answer_terms"),
        )
        bm25_results.append(bm25_eval)
        dense_results.append(dense_eval)

        saved_result = saved_results.get(question_item["question_id"], {})
        saved_result.update(
            {
                "question_id": question_item["question_id"],
                "question": question_item["question"],
                "question_type": question_item["question_type"],
                "bm25": {
                    "answer": result["bm25"]["answer"],
                    "correct": bm25_eval["correct"],
                    "citations_accurate": bm25_eval["citations_accurate"],
                    "retrieved_document_ids": [
                        item["document_id"]
                        for item in result["bm25"]["retrieved_chunks"]
                    ],
                },
                "dense": {
                    "answer": result["dense"]["answer"],
                    "correct": dense_eval["correct"],
                    "citations_accurate": dense_eval["citations_accurate"],
                    "retrieved_document_ids": [
                        item["document_id"]
                        for item in result["dense"]["retrieved_chunks"]
                    ],
                },
            }
        )
        saved_results[question_item["question_id"]] = saved_result

        bm25_status = "✓" if bm25_eval["correct"] else "✗"
        dense_status = "✓" if dense_eval["correct"] else "✗"
        print(f"  BM25: {bm25_status} | Dense: {dense_status}")

    ordered_results = [
        saved_results[item["question_id"]] for item in evaluation_set
    ]
    with results_file.open("w", encoding="utf-8") as file:
        json.dump(ordered_results, file, indent=2)

    bm25_stats = metrics.aggregate_results(bm25_results)
    dense_stats = metrics.aggregate_results(dense_results)
    print_evaluation_summary("BM25 Results", bm25_stats)
    print_evaluation_summary("Dense Retrieval Results", dense_stats)

    difference = (dense_stats["accuracy"] - bm25_stats["accuracy"]) * 100
    print("\nComparison:")
    print(f"  Dense vs BM25: {difference:+.1f}% accuracy")
    print(f"  Results saved to: {results_file}")
    return bm25_results, dense_results


def interactive_demo(bm25, dense, llm):
    """Run the interactive dense-retrieval Q&A demo."""
    from supplyguard.llm.answer_generation import RAGPipeline

    print("Ask questions about vulnerabilities. Type 'exit' to quit.\n")
    pipeline = RAGPipeline(bm25, dense, llm)
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        result = pipeline.generate_answer(question, use_dense=True, top_k=5)
        print(f"\nAnswer: {result['answer']}")
        print("\nSources:")
        for chunk in result["retrieved_chunks"][:3]:
            print(f"  - {chunk['document_id']}: {chunk['summary']}")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="supplyguard",
        description="Search and evaluate OSV security advisories.",
    )
    subparsers = parser.add_subparsers(dest="command")

    ask_parser = subparsers.add_parser(
        "ask", help="Ask one question using BM25 and dense retrieval."
    )
    ask_parser.add_argument("question", help="Security advisory question")
    ask_parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Path to an advisory JSONL corpus",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Run the complete BM25 and dense evaluation."
    )
    evaluate_parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Path to an advisory JSONL corpus",
    )
    evaluate_parser.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help="Where to write evaluation results",
    )

    demo_parser = subparsers.add_parser(
        "demo", help="Start the interactive dense-retrieval demo."
    )
    demo_parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Path to an advisory JSONL corpus",
    )

    subparsers.add_parser(
        "inspect-corpus", help="Inspect the configured raw OSV archives."
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "inspect-corpus":
        from supplyguard.corpus_inspection import main as inspect_corpus

        inspect_corpus()
        return 0

    bm25, dense, llm, _ = setup_system(args.corpus)
    if bm25 is None:
        return 1

    if args.command == "ask":
        run_single_query(args.question, bm25, dense, llm)
    elif args.command == "evaluate":
        run_full_evaluation(
            bm25, dense, llm, results_file=args.results_file
        )
    elif args.command == "demo":
        interactive_demo(bm25, dense, llm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
