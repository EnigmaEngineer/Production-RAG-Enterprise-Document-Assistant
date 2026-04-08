# Retrieval Engine (`src/retrieval/`)

Hybrid retrieval combining dense vector search (FAISS) with sparse keyword
matching (BM25), fused via Reciprocal Rank Fusion (RRF).

## Components

| File | Purpose |
|------|---------|
| `engine.py` | `FAISSStore`, `BM25Index`, `HybridRetriever` |
| `embeddings.py` | `EmbeddingService` wrapping sentence-transformers |

## How It Works

```
Query
  │
  ├──→ Embed (bge-base-en-v1.5) ──→ FAISS top-100
  │
  └──→ Tokenize ──→ BM25 top-100
  │
  └──→ RRF Fusion ──→ Deduplicated candidates (≤ 150)
                        ↓
                    Reranker (top-10)
```

**RRF formula:** `score(d) = Σ α / (K + rank_i(d))` where K=60, α=weight per source.

## Production Metrics & Challenges Solved

### How to Measure

**Retrieval latency:**
The `HybridRetriever.search()` method logs latency at the `info` level with field
`latency_ms`. This is also emitted as a Prometheus histogram. Typical values:
FAISS search 2–5 ms, BM25 search 3–8 ms, fusion < 1 ms. Total retrieval p50 is
under 15 ms for 200K chunks.

**Index size vs. memory:**
FAISS flat index: `num_vectors × dim × 4 bytes`. For 200K vectors at 768 dims:
~585 MB. BM25 corpus tokens: ~2 bytes/token × avg 100 tokens/chunk × 200K = ~40 MB
for tokens plus ~350 MB for the BM25 internal matrices. Monitor with
`process_resident_memory_bytes` in Prometheus.

**Recall quality:**
Run `backtest.py` to measure Hit Rate@K and MRR across chunking configurations.
Production target: Hit@5 ≥ 0.85, MRR ≥ 0.70.

### Challenges Solved

**1. BM25 exact-match superiority for codes and acronyms**
Vector embeddings consistently missed queries containing part numbers (e.g., "K8S-2024"),
acronyms (e.g., "HPA"), and version strings (e.g., "v1.28"). BM25 catches these via
exact token match. The hybrid approach with α=0.5 gives equal weight to both sources,
ensuring that keyword-heavy queries aren't penalized by poor embedding similarity.
We validated this with the backtest framework: hybrid retrieval improved Hit@5 by 17%
over vector-only on our synthetic test set.

**2. FAISS deletion requires full index rebuild**
FAISS `IndexFlatIP` doesn't support point deletion. When a document is updated, we
must remove all its old chunks and re-add the new ones. The `delete_by_doc_id` method
rebuilds the entire index, which takes ~10 seconds for 200K vectors. This is acceptable
at our scale but becomes the migration trigger to Qdrant (which supports atomic point
deletion). During rebuild, the index is locked — concurrent reads block. We mitigate
this by keeping the lock scope minimal and logging the rebuild duration.

**3. Embedding model thread safety**
The sentence-transformers model is not thread-safe for concurrent `encode()` calls
from multiple FastAPI workers. We solved this with a threading lock in `EmbeddingService`
and by running uvicorn with a single worker (relying on Kubernetes HPA for horizontal
scaling instead of multi-worker processes). This avoids CUDA context conflicts and
ensures deterministic embeddings.
