# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [1.0.0] - 2026-04-07

Initial production release. A complete RAG system with hybrid retrieval, cross-encoder reranking, citations, Kubernetes deployment and an evaluation framework that gates deployments on quality metrics.

### Added

**Core RAG pipeline**
- Hybrid retrieval combining FAISS vector search and BM25 keyword matching
- Reciprocal Rank Fusion to merge results from both retrieval sources
- Cross-encoder reranking using BAAI/bge-reranker-v2-m3 with automatic fallback to RRF scores on timeout
- Citation tracking through the entire pipeline with document IDs, page numbers, chunk indexes and relevance scores
- Three chunking strategies selectable at ingestion time: recursive character, semantic via embedding similarity, fixed overlap
- OpenAI-compatible LLM client for vLLM integration
- Dummy LLM client for testing without GPU

**API layer**
- FastAPI application with Pydantic v2 schemas
- Endpoints for ingestion, querying, index statistics, health probes and Prometheus metrics
- Bearer token authentication middleware
- Request ID correlation across all pipeline stages via structlog contextvars
- Circuit breakers with configurable failure thresholds for LLM, vector store and reranker
- Exponential backoff retry with jitter on LLM calls

**Evaluation framework**
- Retrieval metrics including Hit Rate at K, MRR, NDCG at K, Precision at K, Recall at K
- Generation metrics including Token F1 and ROUGE-L with pure-Python fallback when rouge-score is not installed
- LLM-as-judge scoring stub for GPT-4 or Claude integration
- Concurrent load testing via asyncio and ThreadPoolExecutor
- Production readiness gate checks with configurable thresholds
- Per-difficulty stratified reporting across easy, medium, hard and adversarial queries
- JSONL test set format with expected_doc_ids, relevant_chunks, expected_answer and difficulty fields

**Deployment**
- Multi-stage Dockerfile for the API server running as non-root
- Separate Dockerfile for vLLM sidecar
- Docker Compose with profiles for local development
- Kubernetes manifests including Deployment, Service, HPA, Ingress, ConfigMap, Secret and Namespace
- Kustomize base configuration plus production overlay
- Full Helm chart with toggleable Qdrant and vLLM sub-charts
- Startup, readiness and liveness probes with correct initialDelaySeconds for model loading
- Graceful shutdown with 45 second terminationGracePeriodSeconds and preStop sleep

**CI/CD**
- Five GitHub Actions jobs running on every push: lint, unit tests, integration tests, backtest, Docker build smoke test
- Nightly scheduled workflow running the full 5-config evaluation matrix with concurrent load testing
- Pre-commit hooks for autoflake, ruff, vulture, mypy and file hygiene
- HuggingFace model cache sharing between CI jobs

**Configuration**
- All settings configurable via environment variables with RAG_ prefix
- YAML configuration file support via config.yaml
- Evaluation gate thresholds loadable from config.yaml
- Priority chain: environment variables override YAML which overrides defaults

**Testing**
- 20 unit tests covering chunkers, retrievers, reranker, citations and LLM client
- 5 integration tests exercising the full pipeline including subprocess execution of the backtest script
- BM25 test uses 4-document corpus to produce non-zero IDF scores
- Integration test verifies top citation references the correct document across two topically distinct ingested files

**Documentation**
- README with Mermaid architecture diagrams for system overview, query pipeline sequence and Kubernetes topology
- API usage examples with curl commands and sample JSON responses
- CONTRIBUTING guide with known gotchas and coding rules
- DESIGN_DECISIONS document covering five major tradeoff tables and a risk matrix
- INCIDENT_RETROSPECTIVES documenting four production failures and their resolutions
- Per-module READMEs for API, retrieval, evaluation and deployment

### Known issues

- FAISS IndexFlatIP does not support point deletion. Document updates trigger a full index rebuild which locks the index for about 10 seconds at 200K vectors. Migration path to Qdrant is documented.
- Cross-encoder reranker takes about 4 seconds per batch of 15 pairs on CPU. On GPU this drops to about 200ms. CI nightly workflow uses a 15 second latency threshold to account for CPU-only execution.
- BM25 IDF produces zero scores when the corpus has fewer than about 4 documents. This is inherent to the Okapi BM25 formula and not a bug.
- Embedding model cold start takes 30 to 60 seconds on first run. Subsequent starts load from HuggingFace cache in about 5 seconds.

### Dependencies

Pinned in requirements.txt. Key versions:
- Python 3.11 or higher
- FastAPI 0.115
- Pydantic 2.9
- sentence-transformers 3.3
- FAISS 1.8 (faiss-cpu)
- rank-bm25 0.2.2
- vLLM 0.6.x (for production LLM serving)

[Unreleased]: https://github.com/EnigmaEngineer/Production-RAG-Enterprise-Document-Assistant/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/EnigmaEngineer/Production-RAG-Enterprise-Document-Assistant/releases/tag/v1.0.0
