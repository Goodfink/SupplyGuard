# SupplyGuard

[![Quality](https://github.com/Goodfink/SupplyGuard/actions/workflows/quality.yml/badge.svg)](https://github.com/Goodfink/SupplyGuard/actions/workflows/quality.yml)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)

SupplyGuard is a retrieval-augmented generation system for answering questions about open-source software security advisories. It supports BM25, dense semantic retrieval, and hybrid rank fusion over the same local advisory corpus.

## System

- Corpus: 1,000 advisories from the [OSV database](https://osv.dev/)
- Classical retrieval: BM25
- Dense retrieval: `sentence-transformers/all-MiniLM-L6-v2`
- Hybrid retrieval: Reciprocal Rank Fusion over filtered BM25 and dense results
- Local language model: `Qwen/Qwen2.5-0.5B-Instruct`
- Sources: programmatically attached OSV links restricted to retrieved advisories
- Evaluation: answer accuracy, citation quality, Recall@5, hit rate, MRR, and complete evidence
- Evaluation set: 10 factoid, 2 multi-hop, and 2 unanswerable questions
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

### RAG security boundaries

SupplyGuard validates and bounds questions before loading models or querying an index. Retrieved
advisories are serialized as size-limited JSON data, while the grounding policy is sent separately
in the model's system role. The question precedes evidence so tokenizer truncation preserves the
user's request, and each selected advisory receives part of the context budget. Advisory text is
explicitly treated as untrusted data; source links are constructed from retrieved records rather
than accepted from model output.

These controls reduce prompt-injection exposure but do not claim that a language model is immune to
all adversarial text. Applications using generated answers for automated decisions should apply
additional review and policy controls.

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

Audit the fully resolved Python dependency graph against the PyPA advisory database:

```bash
pip install -e ".[security]"
python -m pip_audit . --progress-spinner off
```

GitHub Actions also runs this audit, CodeQL's extended Python security queries, and pull-request
dependency review. Dependabot checks Python, Docker, and GitHub Actions dependencies weekly.

## Reproduce Results

Run all no-retrieval, BM25, dense, and hybrid experiments with one command:

```bash
supplyguard-experiments
```

Results are written atomically to a versioned `evaluation_results.json` report. The report contains
aggregate metrics and per-question evidence together with model revisions, retrieval settings,
runtime settings, Python/platform details, and SHA-256 fingerprints for the corpus and evaluation
set. Existing legacy result lists are migrated automatically on the next experiment run.

Random seeds, dependency versions, model revisions, CPU execution, and deterministic generation are
fixed for reproducibility.

The evaluation reports document-level retrieval quality separately from generated-answer quality.
Recall@5 measures how much required evidence was retrieved, hit rate measures whether any required
advisory was found, and MRR rewards placing the first relevant advisory near the top. Complete
evidence requires every relevant advisory, which makes multi-hop misses visible. Unanswerable
questions are excluded from relevance metrics and instead report how often retrieval correctly
returns no result.

## Demo

Ask one question using BM25, dense, and hybrid retrieval:

```bash
supplyguard ask "Which version fixes the vulnerability in github.com/dhowden/tag?"
```

Answers include structured source records with advisory IDs and canonical OSV links. The language model cannot add a source that was not present in its retrieved context, and abstentions include no sources.

Run the interactive dense-retrieval demo:

```bash
supplyguard demo
```

### Local API

Start the HTTP API with the same corpus, model, device, and index options as the CLI:

```bash
supplyguard serve --host 127.0.0.1 --port 8000
```

Models and retrieval indexes are initialized once during service startup. Ask a question with BM25,
dense, or hybrid retrieval (the default):

```bash
curl --request POST http://127.0.0.1:8000/v1/ask \
  --header "Content-Type: application/json" \
  --data '{"question":"Which version fixes this vulnerability?","retrieval_method":"hybrid"}'
```

Operational probes are available at `/healthz` and `/readyz`. Interactive OpenAPI documentation is
available at `http://127.0.0.1:8000/docs`. The server binds to localhost by default; use
`--host 0.0.0.0` only when network access is intentional and protected by an appropriate gateway.

### Container deployment

Build and start the API with Docker Compose:

```bash
docker compose up --build
```

The first startup downloads the pinned language and embedding models and builds the dense index, so
readiness can take several minutes. Named volumes retain both caches across container replacements.
Once startup completes, open `http://127.0.0.1:8000/docs` or call the API normally. Set a different
host port without changing the container image:

```bash
SUPPLYGUARD_PORT=8080 docker compose up --build
```

The production image uses a digest-pinned Python base, installs the project in an isolated virtual
environment, and runs as a non-root user. Compose binds the service to localhost, drops all Linux
capabilities, prevents privilege escalation, and makes the container filesystem read-only except
for the model cache, index cache, and temporary filesystem.

Stop the service without deleting its caches:

```bash
docker compose down
```

Delete the downloaded models and generated index as well:

```bash
docker compose down --volumes
```

## Results

The table below records the original BM25 and dense reference run. Current experiment reports also
include hybrid answer quality, citation quality, and retrieval metrics under the same configuration.

| Condition | Shared factoids | Full evaluation | Unanswerable |
|---|---:|---:|---:|
| No retrieval | 0/10 (0%) | N/A | N/A |
| BM25 RAG | 7/10 (70%) | 10/14 (71.4%) | 2/2 (100%) |
| Dense RAG | 7/10 (70%) | 9/14 (64.3%) | 2/2 (100%) |

Retrieval substantially improves factoid accuracy over the no-retrieval baseline. BM25 performs better on the multi-hop questions, while both systems correctly abstain on the unanswerable questions.

## Limitations

- The 0.5B model sometimes extracts the wrong fact from correctly retrieved evidence.
- Sources identify the retrieved evidence supplied to the model, but they do not yet provide claim-level faithfulness guarantees.
- Prompt isolation and bounded context reduce injection risk but cannot guarantee model behavior.
- Dense retrieval misses one required advisory on a multi-hop question.
