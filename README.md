# RAG Enterprise Document Assistant

Production-grade Retrieval-Augmented Generation system with hybrid retrieval,
cross-encoder reranking, citation support, and Kubernetes deployment.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client                               │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTPS
                    ┌───────▼────────┐
                    │    Ingress     │
                    │  (nginx/ALB)   │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │   FastAPI      │ ← Prometheus /metrics
                    │  Orchestrator  │ ← /healthz/ready
                    └──┬────┬────┬──┘   /healthz/live
                       │    │    │
          ┌────────────┘    │    └────────────┐
          │                 │                 │
   ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
   │  Retrieval  │  │  Reranker   │  │    vLLM     │
   │ FAISS+BM25  │  │ CrossEnc.   │  │ Llama 3.1  │
   │  (hybrid)   │  │ bge-reranker│  │   8B-Inst   │
   └─────────────┘  └─────────────┘  └─────────────┘
```

## Quick Start

```bash
# 1. Clone and install
git clone <repo> && cd rag-enterprise
pip install -r requirements.txt

# 2. Run locally (dummy LLM, no GPU needed)
RAG_USE_DUMMY_LLM=true python -m uvicorn src.api.app:app --reload

# 3. Ingest a document
curl -X POST http://localhost:8000/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Kubernetes automates deployment...", "filename": "k8s.pdf"}'

# 4. Query
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does Kubernetes work?", "top_k": 5}'

# 5. Run evaluation
./scripts/backtest.sh
```

## Docker

```bash
# API only (CPU, dummy LLM)
docker compose up api

# Full stack with GPU
docker compose --profile gpu up

# With Qdrant
docker compose --profile qdrant up
```

## Kubernetes

```bash
# Kustomize
kubectl apply -k deploy/k8s/base/

# Helm
helm install rag deploy/helm/rag-assistant/ \
  --set secrets.apiKey=my-secret-key \
  --set vllm.enabled=true
```

## Project Structure

```
rag-enterprise/
├── src/
│   ├── api/          # FastAPI app, routes, pipeline orchestrator
│   ├── chunking/     # Recursive, semantic, fixed-overlap chunkers
│   ├── retrieval/    # FAISS, BM25, hybrid retriever, embeddings
│   ├── reranker/     # Cross-encoder reranker with RRF fallback
│   ├── citation/     # Citation formatter for LLM prompts
│   ├── llm/          # vLLM client (OpenAI-compatible)
│   ├── models/       # Pydantic schemas
│   ├── utils/        # Logging, circuit breakers, retries
│   └── config.py     # All settings via environment variables
├── deploy/
│   ├── k8s/          # Kubernetes manifests + kustomize
│   └── helm/         # Helm chart with full templating
├── evaluation/       # Backtesting framework
├── docs/             # Design decisions, incident retrospectives
├── scripts/          # backtest.sh, utilities
└── tests/            # Unit and integration tests
```

## Configuration

All settings are configurable via environment variables with the `RAG_` prefix.
See `src/config.py` for the complete list with defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_CHUNK_STRATEGY` | `recursive` | `recursive`, `semantic`, or `fixed_overlap` |
| `RAG_CHUNK_SIZE` | `512` | Tokens per chunk |
| `RAG_CHUNK_OVERLAP` | `128` | Overlap tokens between chunks |
| `RAG_VECTOR_BACKEND` | `faiss` | `faiss` or `qdrant` |
| `RAG_RERANKER_ENABLED` | `true` | Enable cross-encoder reranking |
| `RAG_LLM_BASE_URL` | `http://vllm:8000/v1` | vLLM endpoint |
| `RAG_USE_DUMMY_LLM` | `true` | Use mock LLM for testing |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/ingest` | Ingest a document |
| `POST` | `/v1/query` | Query with RAG pipeline |
| `GET` | `/v1/stats` | Index statistics |
| `GET` | `/healthz/ready` | Readiness probe |
| `GET` | `/healthz/live` | Liveness probe |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Swagger UI |
