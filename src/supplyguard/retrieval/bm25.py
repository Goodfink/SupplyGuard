import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from supplyguard.paths import DEFAULT_CORPUS_PATH

from .tokenizer import tokenize

type AdvisoryRecord = dict[str, Any]
type SearchResult = dict[str, Any]


def load_corpus(corpus_path: Path) -> list[AdvisoryRecord]:
    records: list[AdvisoryRecord] = []

    with corpus_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error

            if not record.get("retrieval_text"):
                raise ValueError(f"Record on line {line_number} is missing retrieval_text")

            records.append(record)

        return records


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

        self.records: list[AdvisoryRecord] = []
        self.document_lengths: list[int] = []
        self.average_document_length = 0.0

        self.document_frequencies: Counter[str] = Counter()
        self.inverse_document_frequencies: dict[str, float] = {}
        self.postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)

    def build_index(self, records: list[AdvisoryRecord]) -> None:
        if not records:
            raise ValueError("No records found")

        self.records = records
        self.document_lengths = []
        self.document_frequencies = Counter()
        self.inverse_document_frequencies = {}
        self.postings = defaultdict(list)

        for document_index, record in enumerate(records):
            document_tokens = tokenize(record["retrieval_text"])

            term_frequencies = Counter(document_tokens)
            self.document_lengths.append(len(document_tokens))

            for term, frequency in term_frequencies.items():
                self.document_frequencies[term] += 1
                self.postings[term].append((document_index, frequency))

        total_document_length = sum(self.document_lengths)
        self.average_document_length = total_document_length / len(records)
        number_of_documents = len(records)

        for term, document_frequency in self.document_frequencies.items():
            self.inverse_document_frequencies[term] = math.log(
                1 + (number_of_documents - document_frequency + 0.5) / (document_frequency + 0.5)
            )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self.records:
            raise ValueError("Build the BM25 index before searching")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = [0.0] * len(self.records)

        for term in query_tokens:
            inverse_document_frequencies = self.inverse_document_frequencies.get(term)

            if inverse_document_frequencies is None:
                continue

            for document_index, term_frequency in self.postings[term]:
                document_length = self.document_lengths[document_index]
                length_normalization = (
                    1 - self.b + self.b * (document_length / self.average_document_length)
                )
                numerator = term_frequency * (self.k1 + 1)
                denominator = term_frequency + self.k1 * length_normalization
                scores[document_index] += inverse_document_frequencies * numerator / denominator

        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        results: list[SearchResult] = []

        for document_index in ranked_indices:
            score = scores[document_index]

            if score <= 0:
                continue

            record = self.records[document_index]

            results.append(
                {
                    "rank": len(results) + 1,
                    "score": score,
                    "document_id": record["document_id"],
                    "chunk_id": record["chunk_id"],
                    "summary": record["summary"],
                    "retrieval_text": record["retrieval_text"],
                    "record": record,
                }
            )

            if len(results) == top_k:
                break

        return results


def print_results(results: list[SearchResult]) -> None:

    if not results:
        print("No matching results found")
        return

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Document ID: {result['document_id']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Summary: {result['summary']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search OSV advisories using BM25")

    parser.add_argument("--query", required=True, help="Question or search query")
    parser.add_argument(
        "--top_k", required=True, type=int, default=5, help="Number of results to return"
    )
    parser.add_argument(
        "--corpus", type=Path, default=DEFAULT_CORPUS_PATH, help="Path to the advisories.jsonl"
    )
    args = parser.parse_args()

    records = load_corpus(args.corpus)
    retriever = BM25Retriever()
    retriever.build_index(records)
    results = retriever.search(args.query, args.top_k)

    print_results(results)


if __name__ == "__main__":
    main()
