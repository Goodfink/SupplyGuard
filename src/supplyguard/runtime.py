"""Shared construction of SupplyGuard's retrieval and inference runtime."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from supplyguard.config import DEFAULT_CONFIG, SupplyGuardConfig

type ProgressReporter = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class RuntimeSystem:
    """The initialized components shared by CLI and API entry points."""

    bm25: Any
    dense: Any
    llm: Any
    records: list[dict[str, Any]]


def load_runtime(
    corpus_path: Path = DEFAULT_CONFIG.paths.corpus_path,
    *,
    config: SupplyGuardConfig = DEFAULT_CONFIG,
    rebuild_index: bool = False,
    report_progress: ProgressReporter | None = None,
) -> RuntimeSystem:
    """Load the corpus, retrieval indexes, and local language model once."""

    from supplyguard.retrieval.bm25 import BM25Retriever, load_corpus

    records = load_corpus(corpus_path)

    if report_progress is not None:
        report_progress("1. Setting up BM25 retriever...")
    bm25 = BM25Retriever()
    bm25.build_index(records)

    if report_progress is not None:
        report_progress("2. Setting up dense retriever...")
    from supplyguard.retrieval.dense_retrieval import DenseRetriever

    dense = DenseRetriever(config.models)
    dense.prepare_index(
        records,
        corpus_path=corpus_path,
        cache_dir=config.paths.dense_index_dir,
        force_rebuild=rebuild_index,
    )

    if report_progress is not None:
        report_progress("3. Loading LLM (this takes 2-3 minutes on first run)...")
    from supplyguard.llm.llm_setup import HuggingFaceLLM

    llm = HuggingFaceLLM(config.models)
    return RuntimeSystem(bm25=bm25, dense=dense, llm=llm, records=records)
