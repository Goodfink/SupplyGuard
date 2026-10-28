import argparse
import json
import sys
from pathlib import Path
from typing import Any

from supplyguard.config import DEFAULT_CONFIG, SupplyGuardConfig, with_model_overrides
from supplyguard.errors import ConfigurationError, SupplyGuardError
from supplyguard.llm.input_validation import validate_question

RETRIEVAL_LABELS = {
    "bm25": "BM25",
    "dense": "Dense Retrieval",
    "hybrid": "Hybrid Retrieval (RRF)",
}


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
    retrieval = stats.get("retrieval")
    if retrieval:
        top_k = retrieval["top_k"]
        print(f"  Retrieval Recall@{top_k}: {format_optional_percentage(retrieval['recall_at_k'])}")
        print(
            f"  Retrieval Hit Rate@{top_k}: "
            f"{format_optional_percentage(retrieval['hit_rate_at_k'])}"
        )
        mrr = retrieval["mrr_at_k"]
        print(f"  Retrieval MRR@{top_k}: {'N/A' if mrr is None else f'{mrr:.3f}'}")
        print(
            f"  Complete Evidence@{top_k}: "
            f"{format_optional_percentage(retrieval['complete_recall_rate_at_k'])}"
        )
        print(
            "  Unanswerable Empty-Result Rate: "
            f"{format_optional_percentage(retrieval['unanswerable_empty_result_rate'])}"
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

    print("\nModel configuration:")
    print(
        f"  Language: {config.models.language_model_name} "
        f"(revision: {config.models.language_model_revision})"
    )
    print(
        f"  Embedding: {config.models.embedding_model_name} "
        f"(revision: {config.models.embedding_model_revision})"
    )
    print(f"  Device: {config.models.device}")

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
    """Run a single query with lexical, semantic, and hybrid retrieval."""
    from supplyguard.llm.answer_generation import RETRIEVAL_METHODS, RAGPipeline

    print(f"\nQuestion: {question}\n")
    pipeline = RAGPipeline(bm25, dense, llm)

    results = {
        retrieval_method: pipeline.generate_answer(
            question,
            retrieval_method=retrieval_method,
        )
        for retrieval_method in RETRIEVAL_METHODS
    }

    for retrieval_method, result in results.items():
        print(f"{RETRIEVAL_LABELS[retrieval_method]} Results:")
        if result["retrieved_chunks"]:
            print(f"  Top chunk: {result['retrieved_chunks'][0]['document_id']}")
            print(f"  Score: {result['retrieved_chunks'][0]['score']:.4f}")
        else:
            print("  No matching advisory found")
        print()

    for retrieval_method, result in results.items():
        print(f"{RETRIEVAL_LABELS[retrieval_method]} Answer:")
        print(f"  {result['answer']}")
        print_sources(result["sources"])
        print()


def run_full_evaluation(
    bm25,
    dense,
    llm,
    results_file=DEFAULT_CONFIG.paths.results_file,
    config: SupplyGuardConfig = DEFAULT_CONFIG,
    corpus_path=DEFAULT_CONFIG.paths.corpus_path,
):
    """Run evaluation on all test questions."""
    from supplyguard.evaluation.evaluation_metrics import EvaluationMetrics
    from supplyguard.evaluation.reporting import (
        build_evaluation_report,
        index_question_results,
        load_evaluation_report,
        save_evaluation_report,
    )
    from supplyguard.evaluation.retrieval_metrics import (
        aggregate_retrieval_results,
        evaluate_retrieval,
    )
    from supplyguard.llm.answer_generation import (
        RETRIEVAL_METHODS,
        generate_with_all_retrievers,
    )

    evaluation_set = load_evaluation_set()
    metrics = EvaluationMetrics()
    existing_report = load_evaluation_report(results_file)
    saved_results = index_question_results(existing_report)

    answer_results: dict[str, list[dict[str, Any]]] = {
        retrieval_method: [] for retrieval_method in RETRIEVAL_METHODS
    }
    retrieval_results: dict[str, list[dict[str, Any]]] = {
        retrieval_method: [] for retrieval_method in RETRIEVAL_METHODS
    }
    for index, question_item in enumerate(evaluation_set, 1):
        print(f"\n[{index}/{len(evaluation_set)}] {question_item['question'][:60]}...")
        result = generate_with_all_retrievers(question_item["question"], bm25, dense, llm)
        saved_system_results = {}
        statuses = []

        for retrieval_method in RETRIEVAL_METHODS:
            system_result = result[retrieval_method]
            cited_ids = [source["advisory_id"] for source in system_result["sources"]]
            retrieved_ids = [item["document_id"] for item in system_result["retrieved_chunks"]]
            context_advisory_ids = [
                str(item["record"].get("advisory_id") or item["document_id"])
                for item in system_result["retrieved_chunks"]
            ]
            retrieval_eval = evaluate_retrieval(
                context_advisory_ids,
                question_item["ground_truth_chunk_ids"],
                question_item["question_type"],
            )
            answer_eval = metrics.evaluate_answer(
                system_result["answer"],
                question_item["ground_truth_answer"],
                question_item["question_type"],
                question_item["ground_truth_chunk_ids"],
                question_item.get("required_answer_terms"),
                question_item.get("forbidden_answer_terms"),
                cited_ids=cited_ids,
                retrieved_document_ids=context_advisory_ids,
            )
            answer_results[retrieval_method].append(answer_eval)
            retrieval_results[retrieval_method].append(retrieval_eval)
            saved_system_results[retrieval_method] = {
                "answer": system_result["answer"],
                "correct": answer_eval["correct"],
                "citations_accurate": answer_eval["citations_accurate"],
                "citations_grounded": answer_eval["citations_grounded"],
                "sources": system_result["sources"],
                "retrieved_document_ids": retrieved_ids,
                "retrieval_metrics": retrieval_eval,
            }
            status = "✓" if answer_eval["correct"] else "✗"
            statuses.append(f"{RETRIEVAL_LABELS[retrieval_method]}: {status}")

        saved_result = saved_results.get(question_item["question_id"], {})
        saved_result.update(
            {
                "question_id": question_item["question_id"],
                "question": question_item["question"],
                "question_type": question_item["question_type"],
                **saved_system_results,
            }
        )
        saved_results[question_item["question_id"]] = saved_result

        print("  " + " | ".join(statuses))

    ordered_results = [saved_results[item["question_id"]] for item in evaluation_set]
    system_stats = {}
    for retrieval_method in RETRIEVAL_METHODS:
        stats = metrics.aggregate_results(answer_results[retrieval_method])
        stats["retrieval"] = aggregate_retrieval_results(
            retrieval_results[retrieval_method],
            top_k=config.retrieval.top_k,
        )
        system_stats[retrieval_method] = stats

    bm25_accuracy = system_stats["bm25"]["accuracy"]
    dense_accuracy = system_stats["dense"]["accuracy"]
    hybrid_accuracy = system_stats["hybrid"]["accuracy"]
    report_metrics = {
        **existing_report["metrics"],
        **system_stats,
        "comparison": {
            "dense_minus_bm25_accuracy_points": (dense_accuracy - bm25_accuracy) * 100,
            "hybrid_minus_bm25_accuracy_points": (hybrid_accuracy - bm25_accuracy) * 100,
            "hybrid_minus_dense_accuracy_points": (hybrid_accuracy - dense_accuracy) * 100,
        },
    }
    report = build_evaluation_report(
        questions=ordered_results,
        metrics=report_metrics,
        config=config,
        corpus_path=corpus_path,
    )
    save_evaluation_report(results_file, report)

    for retrieval_method in RETRIEVAL_METHODS:
        print_evaluation_summary(
            f"{RETRIEVAL_LABELS[retrieval_method]} Results",
            system_stats[retrieval_method],
        )

    print("\nComparison:")
    print(f"  Dense vs BM25: {(dense_accuracy - bm25_accuracy) * 100:+.1f}% accuracy")
    print(f"  Hybrid vs BM25: {(hybrid_accuracy - bm25_accuracy) * 100:+.1f}% accuracy")
    print(f"  Hybrid vs Dense: {(hybrid_accuracy - dense_accuracy) * 100:+.1f}% accuracy")
    print(f"  Results saved to: {results_file}")
    return tuple(answer_results[method] for method in RETRIEVAL_METHODS)


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

    def add_runtime_arguments(command_parser):
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
        command_parser.add_argument(
            "--language-model",
            help="Hugging Face model ID or local path for answer generation",
        )
        command_parser.add_argument(
            "--language-model-revision",
            help="Language-model commit, tag, or branch (defaults to main for a custom model)",
        )
        command_parser.add_argument(
            "--embedding-model",
            help="Sentence Transformers model ID or local path for dense retrieval",
        )
        command_parser.add_argument(
            "--embedding-model-revision",
            help="Embedding-model commit, tag, or branch (defaults to main for a custom model)",
        )
        command_parser.add_argument(
            "--device",
            help="Inference device understood by PyTorch, such as cpu, cuda, cuda:0, or mps",
        )

    ask_parser = subparsers.add_parser(
        "ask", help="Ask one question using BM25, dense, and hybrid retrieval."
    )
    ask_parser.add_argument("question", help="Security advisory question")
    add_runtime_arguments(ask_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Run the complete BM25, dense, and hybrid evaluation."
    )
    add_runtime_arguments(evaluate_parser)
    evaluate_parser.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_CONFIG.paths.results_file,
        help="Where to write evaluation results",
    )

    demo_parser = subparsers.add_parser("demo", help="Start the interactive dense-retrieval demo.")
    add_runtime_arguments(demo_parser)

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

        config = with_model_overrides(
            DEFAULT_CONFIG,
            language_model_name=args.language_model,
            language_model_revision=args.language_model_revision,
            embedding_model_name=args.embedding_model,
            embedding_model_revision=args.embedding_model_revision,
            device=args.device,
        )
        if args.command == "ask":
            args.question = validate_question(
                args.question,
                max_chars=config.models.question_max_chars,
            )
        bm25, dense, llm, _ = setup_system(
            args.corpus,
            config=config,
            rebuild_index=args.rebuild_index,
        )

        if args.command == "ask":
            run_single_query(args.question, bm25, dense, llm)
        elif args.command == "evaluate":
            run_full_evaluation(
                bm25,
                dense,
                llm,
                results_file=args.results_file,
                config=config,
                corpus_path=args.corpus,
            )
        elif args.command == "demo":
            interactive_demo(bm25, dense, llm)
        return 0
    except SupplyGuardError as error:
        print(f"SupplyGuard error: {error}", file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
