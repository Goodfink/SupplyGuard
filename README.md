# SupplyGuard

SupplyGuard is a retrieval-augmented generation system for answering questions about open-source software security advisories. It compares BM25 and dense semantic retrieval using the same corpus, local language model, prompt, generation settings, and evaluation set.

## System

- Corpus: 1,000 advisories from the [OSV database](https://osv.dev/)
- Classical retrieval: BM25
- Dense retrieval: `sentence-transformers/all-MiniLM-L6-v2`
- Local language model: `Qwen/Qwen2.5-0.5B-Instruct`
- Evaluation: 10 factoid, 2 multi-hop, and 2 unanswerable questions
- Baseline: the 10 factoid questions answered without retrieval

The tracked corpus is stored in `data/processed/advisories.jsonl`. The evaluation set is stored in `src/evaluation/evaluation_set.json`.

## Setup

Use Python 3.13.5.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The first run downloads the pinned Qwen and MiniLM model revisions from Hugging Face. Later runs use the local model cache.

## Tests

Run the offline unit test suite without downloading either model:

```bash
python -m pytest
```

## Reproduce Results

Run all no-retrieval, BM25, and dense experiments with one command:

```bash
supplyguard-experiments
```

Results are written to `evaluation_results.json`. Random seeds, dependency versions, model revisions, CPU execution, and deterministic generation are fixed for reproducibility.

## Demo

Ask one question using both retrievers:

```bash
supplyguard ask "Which version fixes the vulnerability in github.com/dhowden/tag?"
```

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
- Generated inline citations are inconsistent.
- Dense retrieval misses one required advisory on a multi-hop question.
