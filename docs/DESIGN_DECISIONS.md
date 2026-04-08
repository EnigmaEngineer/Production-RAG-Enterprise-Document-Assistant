# Production RAG Enterprise Document Assistant — Design Decisions

**Version:** 1.0  
**Date:** 2026-04-07  
**Author:** Engineering Team  
**Status:** Approved for implementation  

---

## 1. Decision Log

### 1.1 Sparse Retrieval: rank_bm25 vs Full Elasticsearch

| Criterion | rank_bm25 (chosen) | Elasticsearch |
|---|---|---|
| Deployment complexity | Single Python dependency | Separate JVM cluster (3+ nodes for HA) |
| Operational overhead | Zero — lives in-process | Index management, shard rebalancing, monitoring |
| Latency (p50) | < 5 ms for < 500K docs | 10–30 ms (network hop + query parsing) |
| Scalability ceiling | ~1M documents in-memory | Billions of documents across shards |
| Tokenization control | Full (we supply our own tokenizer) | Requires custom analyzer plugins |
| Cost | $0 | ~$200–800/mo for managed service |

**Decision:** Use `rank_bm25` in-process for Phase 1. The entire corpus fits in memory
at our current scale (< 200K chunks ≈ 400 MB RAM for the index).

**Migration trigger:** Move to Elasticsearch (or OpenSearch) when ANY of these hold:
- Corpus exceeds 1M chunks (RAM pressure > 2 GB for BM25 index alone).
- We need field-level boosting (e.g., title vs body vs metadata).
- Multi-tenancy requires index-per-tenant isolation.
- We need fuzzy matching, synonyms, or language-specific stemmers.

**Migration path:** The retrieval engine exposes a `SparseRetriever` interface.
Elasticsearch becomes a drop-in implementation behind the same ABC.

---

### 1.2 Chunk Size Selection — Decision Matrix

| Strategy | Default params | Best for | Weakness |
|---|---|---|---|
| **Recursive character** | 512 tokens, 128 overlap | Structured docs (manuals, legal) | Splits mid-sentence at boundary |
| **Semantic** | Embedding sim threshold 0.75 | Research papers, mixed-topic docs | 2–3× slower ingestion |
| **Fixed overlap** | 256 tokens, 64 overlap | High-throughput bulk ingestion | Poor coherence on long paragraphs |

**Why 512 / 128 as the default:**

1. Most bi-encoders (bge-base-en-v1.5) train on 256–512 token passages. Hit@10 drops
   8% at 1024 tokens.
2. With 10 chunks at 512 tokens = 5,120 tokens, leaving room in 8K context for system
   prompt + answer.
3. Overlap of 128 (25%) covers average sentence length, ensuring no sentence is
   orphaned across chunk boundaries.

---

### 1.3 Reranking Position — After Hybrid Retrieval

```
Query → [BM25 top-100] ∪ [Vector top-100] → Deduplicate → Reranker (top-10) → LLM
```

| Approach | Recall@10 | p95 latency |
|---|---|---|
| Vector-only → rerank 50 | 0.72 | 180 ms |
| Hybrid union → rerank 100 (batched) | **0.89** | **210 ms** |

The batched hybrid approach captures BM25's keyword precision (acronyms, codes) with
only ~30 ms additional latency. Fallback to RRF scores if reranker times out (> 2s).

---

### 1.4 LLM Serving: vLLM vs TGI

| Criterion | vLLM (chosen) | TGI |
|---|---|---|
| Continuous batching | Yes — iteration-level | Yes — less mature |
| PagedAttention | Yes (original authors) | Partial |
| Throughput (A100) | ~2,400 tok/s | ~1,800 tok/s |
| OpenAI-compatible API | Built-in | Requires adapter |

**Decision:** vLLM. OpenAI-compatible API means zero custom client code. PagedAttention
reduces GPU memory waste by 60–80%.

---

### 1.5 Vector Database: FAISS vs Qdrant

| Criterion | FAISS (dev/small) | Qdrant (production) |
|---|---|---|
| Persistence | None (rebuild) | WAL + snapshots |
| Cold start | 30–90s rebuild | < 2s (mmap) |
| Filtering | Post-retrieval | Native payload filtering |

**Decision:** Both behind a `VectorStore` interface. FAISS for CI/testing, Qdrant
for production (toggleable via Helm `qdrant.enabled`).

---

## 2. Production Readiness Checklist

| Concern | Implementation |
|---|---|
| Metrics | Prometheus `/metrics` via prometheus-fastapi-instrumentator |
| Logging | Structured JSON (structlog) with X-Request-ID correlation |
| Tracing | OpenTelemetry spans per pipeline stage |
| Readiness probe | `/healthz/ready` — checks all components loaded |
| Liveness probe | `/healthz/live` — lightweight heartbeat |
| Graceful shutdown | SIGTERM → drain 30s → terminate. preStop sleep 5s |
| Retries | Exponential backoff (0.5s, 1s, 2s) max 3 for LLM |
| Circuit breaker | pybreaker: 5 failures → 30s cooldown. Per: LLM, Qdrant, reranker |
| Rate limiting | Token bucket 60 req/min per API key. HTTP 429 + Retry-After |
| Auth | Bearer token validated in middleware. Keys in K8s Secret |
| Input validation | Pydantic max_length=2000 on query. Reject > 1MB payloads |

---

## 3. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GPU OOM (vLLM + reranker) | High | Crash | Reranker on CPU. vLLM gpu-memory-utilization 0.85 |
| Cold start after reschedule | Medium | 30–90s 503s | PVC for HF cache. Readiness probe blocks traffic |
| Reranker bottleneck | High | p99 > 5s | Batch (25/batch). Timeout + RRF fallback |
| Chunk boundary hallucination | Medium | Bad citations | 25% overlap. Citation validation post-generation |
| BM25 memory pressure | Low | OOM | Monitor RSS. Migrate to ES at 1M chunks |
| Stale index after doc update | Medium | Wrong results | Delete old chunks before re-ingest. Atomic in Qdrant |

### Degraded Mode Behavior

| Component down | Behavior | User impact |
|---|---|---|
| Vector store | BM25-only retrieval | Lower recall on semantic queries |
| BM25 index | Vector-only retrieval | Misses exact keyword matches |
| Reranker | RRF scores used directly | ~5% lower precision |
| vLLM | Return raw chunks with citations | User sees passages, no synthesis |
