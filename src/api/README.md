# API Server (`src/api/`)

FastAPI orchestrator that wires together the full RAG pipeline:
ingest → chunk → embed → retrieve → rerank → generate → cite.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/v1/ingest` | Bearer | Chunk + embed + index a document |
| `POST` | `/v1/query` | Bearer | Full RAG query with citations |
| `GET` | `/v1/stats` | Bearer | Index statistics |
| `GET` | `/healthz/ready` | None | Readiness probe (503 until pipeline loaded) |
| `GET` | `/healthz/live` | None | Liveness heartbeat |
| `GET` | `/metrics` | None | Prometheus metrics |

## Running Locally

```bash
RAG_USE_DUMMY_LLM=true RAG_LOG_LEVEL=debug \
  python -m uvicorn src.api.app:app --reload --port 8000
```

## Production Metrics & Challenges Solved

### How to Measure

**Latency (p50/p95/p99):**
Prometheus histograms are emitted by `prometheus-fastapi-instrumentator` on every
request. Three sub-histograms break down the pipeline: `rag_retrieval_latency_seconds`,
`rag_rerank_latency_seconds`, `rag_generation_latency_seconds`. Use PromQL:
`histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))`.
Every response also includes an `X-Response-Time-Ms` header for client-side tracking.

**Throughput (req/s):**
`rate(http_requests_total[1m])` in Prometheus. The `/v1/stats` endpoint reports live
index sizes. Load testing with `wrk` or `locust` targeting the `/v1/query` endpoint
is the recommended benchmarking method.

**Cost reduction (vLLM continuous batching):**
vLLM's continuous batching typically yields 2–3× throughput over naive sequential
inference. Measure with: `vllm:num_requests_running` (gauge) and
`vllm:avg_generation_throughput_toks_per_s`. Compare against single-request baseline.
AWQ 4-bit quantization reduces GPU memory from 16 GB to 6 GB, enabling L4/T4 GPUs
instead of A100s — roughly 4× cost reduction per GPU-hour.

**Uptime:**
Readiness probe (`/healthz/ready`) checks that the embedding model, vector index,
and BM25 index are all loaded. Kubernetes only routes traffic to ready pods.
Liveness probe (`/healthz/live`) catches deadlocks. `terminationGracePeriodSeconds: 45`
with a `preStop` sleep of 5s ensures in-flight requests complete before shutdown.

### Challenges Solved

**1. Cold start on embedding model load (30–60s startup)**
The sentence-transformers model download and initialization takes 30–60 seconds on
first boot. We solved this by: (a) including a `startupProbe` with `failureThreshold: 30`
allowing up to 5 minutes for first-time model downloads, (b) returning HTTP 503 from
the readiness probe until the pipeline is fully initialized, and (c) recommending a
PersistentVolume for the HuggingFace cache (`~/.cache/huggingface`) so subsequent
restarts load from local disk in ~5 seconds.

**2. Request correlation across async pipeline stages**
A single `/v1/query` request touches retrieval, reranking, and LLM generation — each
with different latency profiles. We inject an `X-Request-ID` header in middleware and
bind it to structlog's context variables. Every log line from every pipeline stage
carries the same request ID. This made it possible to trace a slow query through all
three stages without distributed tracing infrastructure.

**3. Memory pressure from FAISS + BM25 co-resident in the API process**
Both the FAISS index and the BM25 tokenized corpus live in the same Python process.
At 200K chunks with 768-dim vectors, FAISS uses ~600 MB and BM25 uses ~400 MB. We
set Kubernetes resource limits at 4 GB with requests at 2 GB, and added a Prometheus
alert when RSS exceeds 80% of the limit. The design document includes a migration
trigger to Qdrant (persistent, out-of-process) when the corpus exceeds 1M chunks.
