# Contributing

This project was built through real debugging sessions. BM25 returning zeros on small corpora. FAISS refusing to delete vectors. CI lint failing three times before all 25 errors were fixed. This guide exists so you don't repeat those mistakes.

## Setup

### What you need

- Python 3.11 or higher. 3.12 works. 3.10 does not because pydantic-settings requires 3.11.
- Git
- About 2 GB of free disk. The embedding model downloads on first run.
- Docker is optional. Only needed if you want to test the container build.
- No GPU required. Everything runs on CPU in dummy mode.

### Install

```bash
git clone https://github.com/EnigmaEngineer/Production-RAG-Enterprise-Document-Assistant.git
cd Production-RAG-Enterprise-Document-Assistant
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pytest ruff autoflake
```

### Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

This catches problems before they reach CI. Without it you will discover that ruff requires explicit `X as X` re-exports in `__init__.py` files after your code is already pushed. We learned this the hard way.

### Run the API

```bash
RAG_USE_DUMMY_LLM=true python -m uvicorn src.api.app:app --reload --port 8000
```

Open http://localhost:8000/docs for the Swagger UI.

On Windows PowerShell:
```powershell
$env:RAG_USE_DUMMY_LLM="true"
python -m uvicorn src.api.app:app --reload --port 8000
```

### Run tests

```bash
# Unit tests. Fast. No model download.
python -m pytest tests/test_pipeline.py -v

# Integration tests. Downloads the embedding model on first run. Takes about 30 seconds.
python -m pytest tests/test_integration.py -v
```

Do not run both test files in a single pytest invocation. The embedding model loaded by integration tests plus the FAISS indexes from unit tests will exceed 4 GB and OOM on constrained environments. Run them separately.

### Run the backtest

```bash
python evaluation/backtest.py \
  --config config.yaml \
  --generate-synthetic \
  --single-config recursive_512_no_rerank
```

## Things that will bite you

### BM25 returns zero scores on tiny corpora

The BM25Okapi IDF formula produces zero for any term that appears in 50% or more of documents. With only 2 documents every term gets IDF of roughly zero. You need at least 4 documents before BM25 produces meaningful scores. This is not a bug. It is how Okapi BM25 works.

### FAISS does not support point deletion

You cannot remove a single vector from `IndexFlatIP`. When a document is updated the code rebuilds the entire FAISS index from scratch. This takes about 10 seconds at 200K vectors. During rebuild the index is locked and concurrent reads block. If you need fast deletion use the Qdrant backend instead.

### The embedding model takes 30 to 60 seconds to load

On first run sentence-transformers downloads `BAAI/bge-base-en-v1.5` from HuggingFace. On subsequent runs it loads from cache in about 5 seconds. If you are in an air-gapped environment set `TRANSFORMERS_OFFLINE=1` and make sure the cache directory has the model files.

### Ruff and autoflake disagree on init files

Autoflake with `--ignore-init-module-imports` will leave `__init__.py` re-exports alone. Ruff will flag them as F401 unused imports unless you use the explicit `X as X` pattern. Every `__init__.py` in this project uses that pattern. Do the same for any new modules.

### The reranker takes 4 seconds per query on CPU

The cross-encoder model `BAAI/bge-reranker-v2-m3` is 568M parameters. On a GitHub Actions runner it takes about 4 seconds per batch of 15 pairs. On a GPU it takes about 200ms. The nightly CI workflow sets `GATE_P95_MS=15000` to account for this. Do not lower that threshold unless you are running on GPU.

## Coding rules

### No magic numbers

Every tunable value lives in `src/config.py` as a `Settings` field. All settings are overridable via environment variables with the `RAG_` prefix. If you add a new threshold or limit put it in Settings and document it in `config.yaml`.

### No bare except blocks

Always catch specific exceptions. Log errors with structured fields through structlog. Never `except: pass`.

### No unused imports

Autoflake and ruff both check for this. In `__init__.py` files use the explicit re-export pattern:

```python
# correct
from src.chunking.engine import RecursiveChunker as RecursiveChunker

# incorrect. ruff will flag this as F401.
from src.chunking.engine import RecursiveChunker
```

### Type hints on public functions

Every public function and method needs type hints on its arguments and return value.

### Timeouts on all external calls

Every call to the LLM, vector store or reranker must have an explicit timeout. Use `asyncio.wait_for` or the timeout parameter on the client. If a call times out the system must degrade gracefully. The reranker falls back to RRF scores. The LLM falls back to returning raw chunks. The vector store falls back to BM25-only retrieval.

## How to make changes

### Branch names

```
feature/add-pdf-ingestion
fix/bm25-zero-scores
docs/update-api-examples
test/add-reranker-timeout-test
```

### Commit messages

Be specific about what changed and why.

```
# good
Add PDF text extraction utility using PyPDF2
Fix BM25 returning zero scores on 2-document corpora
Add integration test verifying top citation matches correct doc_id

# bad
Update files
Fix bug
WIP
```

### Before you open a PR

Run these locally. If any of them fail CI will also fail.

```bash
ruff check src/ evaluation/ tests/
ruff format --check src/ evaluation/ tests/
python -m pytest tests/test_pipeline.py -v
python -m pytest tests/test_integration.py -v
```

### Adding a new module

1. Create `src/your_module/engine.py` with the implementation
2. Create `src/your_module/__init__.py` with explicit `X as X` re-exports
3. Write tests in `tests/test_your_module.py`
4. Wire it into the pipeline in `src/api/pipeline.py`
5. Add config values to `src/config.py` and `config.yaml`
6. Update the README if the API surface changes

### Adding a test

Unit tests go in `tests/test_pipeline.py`. These must not load external models or make network calls.

Integration tests go in `tests/test_integration.py`. These may load the embedding model and run the full pipeline.

Every test must assert a specific behavior. No `assert True`. No tests that just check a function runs without crashing. Test the actual output.

```python
# good. verifies the correct document is retrieved.
assert top_citation.doc_id == k8s_doc_id

# bad. proves nothing.
assert response is not None
```

## CI pipeline

Every push runs 5 jobs. All must pass.

| Job | What it checks | Typical time |
|-----|---------------|-------------|
| Lint and Format | autoflake, ruff lint, ruff format | 13s |
| Unit Tests | 20 tests across all components | 90s |
| Integration Tests | Ingest then query with real embeddings | 2m |
| Backtest | Evaluation metrics with gate checks | 2m |
| Docker Build | Image builds and health endpoint responds | 2.5m |

The nightly workflow runs the full 5-config evaluation matrix with the cross-encoder reranker and concurrent load testing. It takes about 8 minutes.

## Where to look when something breaks

- `docs/DESIGN_DECISIONS.md` explains why each technology was chosen and when to migrate
- `docs/INCIDENT_RETROSPECTIVES.md` documents specific failures and how the code prevents them
- `evaluation/README.md` explains how to interpret metrics and set gate thresholds
- `src/api/README.md` covers production challenges in the API layer
- `src/retrieval/README.md` covers retrieval-specific edge cases
- `deploy/README.md` covers Kubernetes deployment gotchas
