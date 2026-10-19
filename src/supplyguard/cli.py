import argparse
import json
import sys
from pathlib import Path

from supplyguard.config import DEFAULT_CONFIG, SupplyGuardConfig
from supplyguard.errors import ConfigurationError, SupplyGuardError


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
    print(
        "  Citation Grounding (citations present in context): "
        f"{format_optional_percentage(stats['citation_grounding_accuracy'])}"
    )


def print_sources(sources):
    """Print verified OSV source records returned by the RAG pipeline."""
    if not sources:
        return

    print("  Sources:")
    for source in sources:
        print(f"    - {source['advisory_id']}: {source['url']}")


def load_evaluation_set(evaluation_file=DEFAULT_CONFIG.paths.evaluation_file):
    """Load evaluation questions bundled with SupplyGuard."""
    try:
        with evaluation_file.open(encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(
            f"Could not load evaluation set '{evaluation_file}': {error}"
        ) from error


def setup_system(
    corpus_path=DEFAULT_CONFIG.paths.corpus_path,
    config: SupplyGuardConfig = DEFAULT_CONFIG,
    rebuild_index: bool = False,
):
    """Initialize retrieval and LLM systems."""
    from supplyguard.retrieval.bm25 import BM25Retriever, load_corpus

    records = load_corpus(corpus_path)

    print("\n1. Setting up BM25 retriever...")
    bm25 = BM25Retriever()
    bm25.build_index(records)

    print("\n2. Setting up dense retriever...")
    from supplyguard.retrieval.dense_retrieval import DenseRetriever

    dense = DenseRetriever(config.models)
    dense.prepare_index(
        records,
        corpus_path=corpus_path,
        cache_dir=config.paths.dense_index_dir,
        force_rebuild=rebuild_index,
    )

    print("\n3. Loading LLM (this takes 2-3 minutes on first run)...")
    from supplyguard.llm.llm_setup import HuggingFaceLLM

    llm = HuggingFaceLLM(config.models)

    return bm25, dense, llm, records


def run_single_query(question, bm25, dense, llm):
    """Run a single query with both retrievers."""
    from supplyguard.llm.answer_generation import RAGPipeline

    print(f"\nQuestion: {question}\n")
    pipeline = RAGPipeline(bm25, dense, llm)

    print("BM25 Results:")
    bm25_result = pipeline.generate_answer(question, use_dense=False)
    if bm25_result["retrieved_chunks"]:
        print(f"  Top chunk: {bm25_result['retrieved_chunks'][0]['document_id']}")
        print(f"  Score: {bm25_result['retrieved_chunks'][0]['score']:.4f}")
    else:
        print("  No matching advisory found")

    print("\nDense Retrieval Results:")
    dense_result = pipeline.generate_answer(question, use_dense=True)
    if dense_result["retrieved_chunks"]:
        print(f"  Top chunk: {dense_result['retrieved_chunks'][0]['document_id']}")
        print(f"  Score: {dense_result['retrieved_chunks'][0]['score']:.4f}")
    else:
        print("  No matching advisory found")

    print("\nBM25 Answer:")
    print(f"  {bm25_result['answer']}")
    print_sources(bm25_result["sources"])
    print("\nDense Answer:")
    print(f"  {dense_result['answer']}")
    print_sources(dense_result["sources"])


def run_full_evaluation(bm25, dense, llm, results_file=DEFAULT_CONFIG.paths.results_file):
    """Run evaluation on all test questions."""
    from supplyguard.evaluation.evaluation_metrics import EvaluationMetrics
    from supplyguard.llm.answer_generation import generate_with_both_retrievers

    evaluation_set = load_evaluation_set()
    metrics = EvaluationMetrics()
    if results_file.exists():
        with results_file.open(encoding="utf-8") as file:
            saved_results = {item["question_id"]: item for item in json.load(file)}
    else:
        saved_results = {}

    bm25_results = []
    dense_results = []
    for index, question_item in enumerate(evaluation_set, 1):
        print(f"\n[{index}/{len(evaluation_set)}] {question_item['question'][:60]}...")
        result = generate_with_both_retrievers(question_item["question"], bm25, dense, llm)

        bm25_cited_ids = [source["advisory_id"] for source in result["bm25"]["sources"]]
        dense_cited_ids = [source["advisory_id"] for source in result["dense"]["sources"]]
        bm25_retrieved_ids = [item["document_id"] for item in result["bm25"]["retrieved_chunks"]]
        dense_retrieved_ids = [item["document_id"] for item in result["dense"]["retrieved_chunks"]]
        bm25_context_advisory_ids = [
            str(item["record"].get("advisory_id") or item["document_id"])
            for item in result["bm25"]["retrieved_chunks"]
        ]
        dense_context_advisory_ids = [
            str(item["record"].get("advisory_id") or item["document_id"])
            for item in result["dense"]["retrieved_chunks"]
        ]

        bm25_eval = metrics.evaluate_answer(
            result["bm25"]["answer"],
            question_item["ground_truth_answer"],
            question_item["question_type"],
            question_item["ground_truth_chunk_ids"],
            question_item.get("required_answer_terms"),
            question_item.get("forbidden_answer_terms"),
            cited_ids=bm25_cited_ids,
            retrieved_document_ids=bm25_context_advisory_ids,
        )
        dense_eval = metrics.evaluate_answer(
            result["dense"]["answer"],
            question_item["ground_truth_answer"],
            question_item["question_type"],
            question_item["ground_truth_chunk_ids"],
            question_item.get("required_answer_terms"),
            question_item.get("forbidden_answer_terms"),
            cited_ids=dense_cited_ids,
            retrieved_document_ids=dense_context_advisory_ids,
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
                    "citations_grounded": bm25_eval["citations_grounded"],
                    "sources": result["bm25"]["sources"],
                    "retrieved_document_ids": bm25_retrieved_ids,
                },
                "dense": {
                    "answer": result["dense"]["answer"],
                    "correct": dense_eval["correct"],
                    "citations_accurate": dense_eval["citations_accurate"],
                    "citations_grounded": dense_eval["citations_grounded"],
                    "sources": result["dense"]["sources"],
                    "retrieved_document_ids": dense_retrieved_ids,
                },
            }
        )
        saved_results[question_item["question_id"]] = saved_result

        bm25_status = "✓" if bm25_eval["correct"] else "✗"
        dense_status = "✓" if dense_eval["correct"] else "✗"
        print(f"  BM25: {bm25_status} | Dense: {dense_status}")

    ordered_results = [saved_results[item["question_id"]] for item in evaluation_set]
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

        result = pipeline.generate_answer(question, use_dense=True)
        print(f"\nAnswer: {result['answer']}")
        print_sources(result["sources"])


def build_parser():
    parser = argparse.ArgumentParser(
        prog="supplyguard",
        description="Search and evaluate OSV security advisories.",
    )
    subparsers = parser.add_subparsers(dest="command")

    def add_index_arguments(command_parser):
        command_parser.add_argument(
            "--corpus",
            type=Path,
            default=DEFAULT_CONFIG.paths.corpus_path,
            help="Path to an advisory JSONL corpus",
        )
        command_parser.add_argument(
            "--rebuild-index",
            action="store_true",
            help="Re-encode the corpus even when a valid dense index is cached",
        )

    ask_parser = subparsers.add_parser(
        "ask", help="Ask one question using BM25 and dense retrieval."
    )
    ask_parser.add_argument("question", help="Security advisory question")
    add_index_arguments(ask_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Run the complete BM25 and dense evaluation."
    )
    add_index_arguments(evaluate_parser)
    evaluate_parser.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_CONFIG.paths.results_file,
        help="Where to write evaluation results",
    )

    demo_parser = subparsers.add_parser("demo", help="Start the interactive dense-retrieval demo.")
    add_index_arguments(demo_parser)

    subparsers.add_parser("inspect-corpus", help="Inspect the configured raw OSV archives.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "inspect-corpus":
            from supplyguard.corpus_inspection import main as inspect_corpus

            inspect_corpus()
            return 0

        bm25, dense, llm, _ = setup_system(args.corpus, rebuild_index=args.rebuild_index)

        if args.command == "ask":
            run_single_query(args.question, bm25, dense, llm)
        elif args.command == "evaluate":
            run_full_evaluation(bm25, dense, llm, results_file=args.results_file)
        elif args.command == "demo":
            interactive_demo(bm25, dense, llm)
        return 0
    except SupplyGuardError as error:
        print(f"SupplyGuard error: {error}", file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
