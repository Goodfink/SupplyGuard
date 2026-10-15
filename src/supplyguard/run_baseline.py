import json

from supplyguard.config import DEFAULT_CONFIG


def main():
    """Run the no-retrieval factoid baseline."""
    from supplyguard.evaluation.evaluation_metrics import EvaluationMetrics
    from supplyguard.llm.llm_setup import HuggingFaceLLM

    with DEFAULT_CONFIG.paths.evaluation_file.open(encoding="utf-8") as file:
        evaluation_set = json.load(file)

    diagnostic_items = [item for item in evaluation_set if item["question_type"] == "factoid"]
    llm = HuggingFaceLLM(DEFAULT_CONFIG.models)
    metrics = EvaluationMetrics()

    if DEFAULT_CONFIG.paths.results_file.exists():
        with DEFAULT_CONFIG.paths.results_file.open(encoding="utf-8") as file:
            saved_results = {item["question_id"]: item for item in json.load(file)}
    else:
        saved_results = {}

    for index, question_item in enumerate(diagnostic_items, 1):
        question = question_item["question"]
        print(f"[{index}/{len(diagnostic_items)}] Q: {question}")

        diagnostic_prompt = f"""Answer the question briefly.
If you do not know, say "I don't know."

Question: {question}
Answer:"""
        result_text = llm.generate(
            diagnostic_prompt,
            max_tokens=DEFAULT_CONFIG.models.baseline_max_tokens,
        )

        evaluation = metrics.evaluate_answer(
            result_text,
            question_item["ground_truth_answer"],
            question_item["question_type"],
            [],
            question_item.get("required_answer_terms"),
            question_item.get("forbidden_answer_terms"),
            require_citations=False,
        )
        saved_result = saved_results.get(question_item["question_id"], {})
        manual_correct = saved_result.get("no_retrieval", {}).get("manual_correct")
        saved_result.update(
            {
                "question_id": question_item["question_id"],
                "question": question,
                "question_type": question_item["question_type"],
                "no_retrieval": {
                    "answer": result_text,
                    "correct": evaluation["correct"],
                    "citations_accurate": evaluation["citations_accurate"],
                    "manual_correct": manual_correct,
                },
            }
        )
        saved_results[question_item["question_id"]] = saved_result
        print(f"    A: {result_text}\n")

    results = [
        saved_results[item["question_id"]]
        for item in evaluation_set
        if item["question_id"] in saved_results
    ]
    with DEFAULT_CONFIG.paths.results_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    automatic_correct = sum(
        saved_results[item["question_id"]]["no_retrieval"]["correct"] for item in diagnostic_items
    )
    print(
        f"Automatic accuracy: {automatic_correct}/{len(diagnostic_items)} "
        f"({automatic_correct / len(diagnostic_items):.1%})"
    )
    print(f"Results saved to: {DEFAULT_CONFIG.paths.results_file}")


if __name__ == "__main__":
    main()
