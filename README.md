# SupplyGuard

[![Quality](https://github.com/Goodfink/SupplyGuard/actions/workflows/quality.yml/badge.svg)](https://github.com/Goodfink/SupplyGuard/actions/workflows/quality.yml)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)

SupplyGuard is a retrieval-augmented generation system for answering questions about open-source software security advisories. It compares BM25 and dense semantic retrieval using the same corpus, local language model, prompt, generation settings, and evaluation set.

## System

- Corpus: 1,000 advisories from the [OSV database](https://osv.dev/)
- Classical retrieval: BM25
- Dense retrieval: `sentence-transformers/all-MiniLM-L6-v2`
- Local language model: `Qwen/Qwen2.5-0.5B-Instruct`
- Sources: programmatically attached OSV links restricted to retrieved advisories
- Evaluation: 10 factoid, 2 multi-hop, and 2 unanswerable questions
- Baseline: the 10 factoid questions answered without retrieval

The tracked corpus is stored in `data/processed/advisories.jsonl`. The evaluation set is stored in `src/supplyguard/evaluation/evaluation_set.json`.

## Setup

Use Python 3.13.5.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The first run downloads the pinned Qwen and MiniLM model revisions from Hugging Face. It also
encodes the advisory corpus and stores the dense index under `data/indexes/dense`. Later runs reuse
that index without re-encoding. SupplyGuard automatically builds a new index when the corpus,
ordered document IDs, embedding model, or pinned model revision changes.

Force a clean dense-index rebuild when troubleshooting or benchmarking:

```bash
supplyguard ask "Which version is fixed?" --rebuild-index
```

### Choose local models

The pinned defaults provide the reproducible baseline, but each model command can run compatible
Hugging Face models already on the user's system or download them on first use:

```bash
supplyguard ask "Which version fixes this vulnerability?" \
  --language-model Qwen/Qwen2.5-1.5B-Instruct \
  --embedding-model sentence-transformers/all-mpnet-base-v2 \
  --device cpu
```

SupplyGuard accepts either Hugging Face model IDs or local model-directory paths. Custom language
models must be compatible causal chat models, and custom embedding models must be compatible with
Sentence Transformers. Models that require executing custom remote code are intentionally not
supported. When a custom model is selected without a revision, SupplyGuard uses its `main` branch.
For reproducible evaluations, pass immutable Hugging Face commit hashes with
`--language-model-revision` and `--embedding-model-revision`.

Changing the embedding model or revision selects a separate validated dense-index cache. Use
`--rebuild-index` only when an explicit rebuild of that matching cache is desired.

Runtime paths, pinned model revisions, retrieval thresholds, and generation limits are centralized in `src/supplyguard/config.py`. Invalid configuration, corpus, index, and model states produce concise CLI errors with non-zero exit codes instead of partial `None` results.

## Tests

Run the offline unit test suite without downloading either model:

```bash
python -m pytest
```

Run the complete local quality suite:

```bash
ruff format --check src tests
ruff check src tests
mypy
```

## Reproduce Results

Run all no-retrieval, BM25, and dense experiments with one command:

```bash
supplyguard-experiments
```

Results are written atomically to a versioned `evaluation_results.json` report. The report contains
aggregate metrics and per-question evidence together with model revisions, retrieval settings,
runtime settings, Python/platform details, and SHA-256 fingerprints for the corpus and evaluation
set. Existing legacy result lists are migrated automatically on the next experiment run.

Random seeds, dependency versions, model revisions, CPU execution, and deterministic generation are
fixed for reproducibility.

## Demo

Ask one question using both retrievers:

```bash
supplyguard ask "Which version fixes the vulnerability in github.com/dhowden/tag?"
```

Answers include structured source records with advisory IDs and canonical OSV links. The language model cannot add a source that was not present in its retrieved context, and abstentions include no sources.

Run the interactive dense-retrieval demo:

```bash
supplyguard demo
```

## Results

| Condition | Shared factoids | Full evaluation | Unanswerable |
|---|---:|---:|---:|
| No retrieval | 0/10 (0%) | N/A | N/A |
| BM25 RAG | 7/10 (70%) | 10/14 (71.4%) | 2/2 (100%) |
| Dense RAG | 7/10 (70%) | 9/14 (64.3%) | 2/2 (100%) |

Retrieval substantially improves factoid accuracy over the no-retrieval baseline. BM25 performs better on the multi-hop questions, while both systems correctly abstain on the unanswerable questions.

## Limitations

- The 0.5B model sometimes extracts the wrong fact from correctly retrieved evidence.
- Sources identify the retrieved evidence supplied to the model, but they do not yet provide claim-level faithfulness guarantees.
- Dense retrieval misses one required advisory on a multi-hop question.
