# Production Incident Retrospectives

Documenting production incidents (real and anticipated) and their resolutions.
Each entry follows the format: Timeline → Root Cause → Resolution → Prevention.

---

## INC-001: Reranker OOM During Traffic Spike

**Severity:** P1 — Service degradation  
**Duration:** 12 minutes  
**Date:** Anticipated (pre-mortem analysis)

### Timeline

1. Traffic spike (3× normal) causes concurrent reranker requests to queue.
2. Each reranker batch loads 25 query-document pairs into GPU memory.
3. With 8 concurrent requests, 200 pairs load simultaneously.
4. GPU OOM kills the reranker process.
5. Circuit breaker opens after 5 consecutive failures.
6. System falls back to RRF scores (degraded but functional).

### Root Cause

The cross-encoder reranker and vLLM share a single GPU. Under load, the reranker's
memory usage is unpredictable because the input text lengths vary. vLLM's
PagedAttention reserves memory dynamically, and the reranker doesn't coordinate
with vLLM's memory allocator.

### Resolution

1. Move reranker to CPU-only mode using ONNX Runtime.
2. Set `RAG_RERANKER_TIMEOUT_S=2.0` to bound latency.
3. Circuit breaker automatically falls back to RRF scores.

### Prevention

- Separate GPU workloads: vLLM gets exclusive GPU access.
- Reranker runs on CPU with dedicated resource limits (4 vCPU, 8 GB RAM).
- Load test with 5× expected peak before production deployment.
- Add Prometheus alert: `reranker_latency_p99 > 1.5s` triggers investigation.

---

## INC-002: FAISS Index Corruption After Pod Restart

**Severity:** P2 — Data loss (temporary)  
**Duration:** 45 minutes (time to re-ingest)

### Timeline

1. Kubernetes reschedules the API pod to a different node.
2. FAISS index (in-memory only) is lost.
3. BM25 index (also in-memory) is lost.
4. Readiness probe correctly returns 503.
5. Team manually triggers re-ingestion of all documents.

### Root Cause

Neither FAISS nor BM25 indexes are persisted to disk. The `emptyDir` volume is
node-local and doesn't survive pod rescheduling.

### Resolution

1. Implemented FAISS index serialization to a numpy file on PVC.
2. BM25 corpus persistence to a JSON file on the same PVC.
3. On startup, the pipeline checks for persisted indexes and loads them.
4. If no persisted index exists, the system starts empty and re-ingestion is required.

### Prevention

- Migrate to Qdrant for persistent vector storage (WAL + snapshots).
- Implement a document registry (PostgreSQL or SQLite) that tracks ingested documents.
- Add an automated re-ingestion job that runs on startup if the index is empty.
- Set `PersistentVolumeClaim` with `ReadWriteOnce` access mode.

---

## INC-003: Chunk Boundary Hallucination in Citations

**Severity:** P3 — Incorrect information  
**Duration:** Ongoing (quality issue)

### Timeline

1. User queries about a specific policy clause.
2. The relevant information spans two chunks at a boundary.
3. Chunk A contains the beginning of the clause; Chunk B contains the end.
4. With no overlap, neither chunk contains the complete information.
5. The LLM hallucinates a plausible but incorrect completion.
6. The citation references Chunk A, which doesn't contain the hallucinated text.

### Root Cause

Chunking at a fixed character or token boundary can split sentences and paragraphs
at arbitrary points. Without overlap, context is lost at boundaries.

### Resolution

1. Default chunk overlap set to 25% (128 tokens for 512-token chunks).
2. Recursive chunker prioritizes splitting at paragraph and sentence boundaries.
3. Citation formatter validates that cited text actually exists in the chunk.
4. LLM system prompt explicitly instructs: "If passages are incomplete, say so."

### Prevention

- Increase overlap to 30% for legal and policy documents.
- Implement semantic chunking for documents where topic boundaries matter.
- Add post-generation validation: check that every factual claim in the answer
  can be traced to a specific chunk passage.
- Weekly quality review of a random sample of production queries and answers.

---

## INC-004: Embedding Model Download Timeout in Air-Gapped Environment

**Severity:** P1 — Service unable to start  
**Duration:** 2 hours

### Timeline

1. Deploy to a new Kubernetes cluster with restricted internet access.
2. API pod starts, tries to download `BAAI/bge-base-en-v1.5` from HuggingFace.
3. Download times out (corporate firewall blocks huggingface.co).
4. Startup probe eventually kills the pod.
5. Pod enters CrashLoopBackoff.

### Root Cause

The sentence-transformers library downloads model weights on first use. In
air-gapped or restricted network environments, this download fails silently
and the service cannot start.

### Resolution

1. Pre-download the model during Docker image build (`RUN python -c "from
   sentence_transformers import SentenceTransformer; SentenceTransformer('...')"`)
2. Mount the HuggingFace cache from a PVC pre-populated by an init container.
3. Set `TRANSFORMERS_OFFLINE=1` environment variable in production.

### Prevention

- Include model weights in the Docker image for air-gapped deployments.
- Document the required network egress rules (huggingface.co, cdn-lfs.huggingface.co).
- Add a CI check that verifies the Docker image can start with `TRANSFORMERS_OFFLINE=1`.
- Create an internal model registry mirror for approved models.
