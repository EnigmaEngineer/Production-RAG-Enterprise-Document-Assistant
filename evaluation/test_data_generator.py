"""
evaluation/test_data_generator.py — Generate synthetic JSONL test sets.

Each sample contains:
  - query:                 the question
  - expected_doc_ids:      list of doc IDs that should appear in retrieval
  - relevant_chunks:       list of substring snippets that a correct retrieval
                           must surface (for chunk-level recall)
  - expected_answer:       the gold-standard answer for generation metrics
  - difficulty:            easy | medium | hard  (for stratified reporting)
"""

from __future__ import annotations

import json
from pathlib import Path


# ─── Corpus ────────────────────────────────────────────────────────────────

SYNTHETIC_DOCUMENTS = [
    {
        "doc_id": "doc_001",
        "title": "Kubernetes Architecture Overview",
        "filename": "k8s_arch.pdf",
        "pages": [
            "Kubernetes is an open-source container orchestration platform. It automates deployment, scaling, and management of containerized applications. The control plane consists of the API server, etcd, scheduler, and controller manager.",
            "Worker nodes run kubelet and kube-proxy. Pods are the smallest deployable units. Each pod contains one or more containers that share networking and storage. Services provide stable network endpoints for pods.",
            "Horizontal Pod Autoscaler automatically scales the number of pods based on observed CPU utilization or custom metrics. It checks metrics every 15 seconds by default. The scaling algorithm uses a simple ratio calculation.",
        ],
    },
    {
        "doc_id": "doc_002",
        "title": "Vector Database Comparison",
        "filename": "vector_db.pdf",
        "pages": [
            "FAISS is a library for efficient similarity search developed by Meta. It supports various index types including flat, IVF, and HNSW. FAISS operates entirely in memory and does not provide built-in persistence.",
            "Qdrant is a vector search engine with built-in persistence and filtering. It uses HNSW for approximate nearest neighbor search. Qdrant supports payload filtering which allows metadata-based pre-filtering before vector search.",
            "Pinecone is a managed vector database service. It provides automatic scaling and requires no infrastructure management. However, it introduces vendor lock-in and network latency due to its cloud-only architecture.",
        ],
    },
    {
        "doc_id": "doc_003",
        "title": "RAG System Best Practices",
        "filename": "rag_guide.pdf",
        "pages": [
            "Retrieval Augmented Generation combines information retrieval with language model generation. The key challenge is ensuring retrieval quality directly impacts generation quality. Hybrid retrieval using both dense and sparse methods improves recall.",
            "Chunking strategy significantly affects retrieval quality. Recursive character splitting preserves document structure. Semantic chunking groups related sentences by embedding similarity. Overlap between chunks prevents information loss at boundaries.",
            "Reranking with cross-encoder models improves precision after initial retrieval. Cross-encoders process query-document pairs jointly, capturing fine-grained interactions. The BAAI bge-reranker models provide strong multilingual reranking performance.",
        ],
    },
    {
        "doc_id": "doc_004",
        "title": "Production Monitoring Guide",
        "filename": "monitoring.pdf",
        "pages": [
            "Prometheus collects time-series metrics via pull-based scraping. Applications expose metrics on a /metrics endpoint. Key metrics for RAG systems include retrieval latency, reranking latency, and LLM generation time.",
            "Grafana provides dashboards for visualizing Prometheus metrics. Set up alerts for p99 latency exceeding SLO thresholds. Monitor error rates by type: retrieval failures, reranker timeouts, and LLM circuit breaker trips.",
            "Distributed tracing with OpenTelemetry enables end-to-end request visibility. Each pipeline stage (retrieve, rerank, generate) should be a separate span. Trace sampling at 10% is sufficient for most production workloads.",
        ],
    },
    {
        "doc_id": "doc_005",
        "title": "LLM Serving Architectures",
        "filename": "llm_serving.pdf",
        "pages": [
            "vLLM uses PagedAttention to manage GPU memory efficiently. It allocates KV cache in non-contiguous pages, reducing memory waste by 60-80%. This enables higher batch sizes and better throughput compared to static allocation.",
            "Continuous batching processes new requests without waiting for the entire batch to complete. This reduces the time-to-first-token for new arrivals and improves overall GPU utilization. vLLM implements iteration-level scheduling for this purpose.",
            "Speculative decoding uses a smaller draft model to propose tokens that are then verified by the main model in parallel. This can improve throughput by 2-3x for certain workloads while maintaining output quality.",
        ],
    },
]


# ─── Test samples ──────────────────────────────────────────────────────────

SYNTHETIC_QUERIES = [
    # ── Easy: answer is almost verbatim in one chunk ──
    {
        "query": "What are the main components of the Kubernetes control plane?",
        "expected_doc_ids": ["doc_001"],
        "relevant_chunks": ["API server, etcd, scheduler, and controller manager"],
        "expected_answer": "The Kubernetes control plane consists of the API server, etcd, scheduler, and controller manager.",
        "difficulty": "easy",
    },
    {
        "query": "How does FAISS handle data persistence?",
        "expected_doc_ids": ["doc_002"],
        "relevant_chunks": [
            "FAISS operates entirely in memory and does not provide built-in persistence"
        ],
        "expected_answer": "FAISS operates entirely in memory and does not provide built-in persistence.",
        "difficulty": "easy",
    },
    {
        "query": "What is the benefit of hybrid retrieval in RAG systems?",
        "expected_doc_ids": ["doc_003"],
        "relevant_chunks": [
            "Hybrid retrieval using both dense and sparse methods improves recall"
        ],
        "expected_answer": "Hybrid retrieval using both dense and sparse methods improves recall in RAG systems.",
        "difficulty": "easy",
    },
    {
        "query": "What tracing sample rate is recommended for production?",
        "expected_doc_ids": ["doc_004"],
        "relevant_chunks": [
            "Trace sampling at 10% is sufficient for most production workloads"
        ],
        "expected_answer": "Trace sampling at 10 percent is sufficient for most production workloads.",
        "difficulty": "easy",
    },
    # ── Medium: requires paraphrasing or combining information ──
    {
        "query": "How does Qdrant support metadata filtering during vector search?",
        "expected_doc_ids": ["doc_002"],
        "relevant_chunks": [
            "payload filtering which allows metadata-based pre-filtering before vector search"
        ],
        "expected_answer": "Qdrant supports payload filtering which allows metadata-based pre-filtering before vector search, filtering results based on metadata before the vector similarity computation.",
        "difficulty": "medium",
    },
    {
        "query": "Why does chunk overlap matter for RAG quality?",
        "expected_doc_ids": ["doc_003"],
        "relevant_chunks": [
            "Overlap between chunks prevents information loss at boundaries"
        ],
        "expected_answer": "Overlap between chunks prevents information loss at boundaries during retrieval, ensuring that sentences split across chunk boundaries are still fully represented in at least one chunk.",
        "difficulty": "medium",
    },
    {
        "query": "How does the Horizontal Pod Autoscaler decide when to scale?",
        "expected_doc_ids": ["doc_001"],
        "relevant_chunks": [
            "observed CPU utilization or custom metrics",
            "checks metrics every 15 seconds",
        ],
        "expected_answer": "The HPA automatically scales pods based on observed CPU utilization or custom metrics, checking every 15 seconds using a ratio-based algorithm.",
        "difficulty": "medium",
    },
    {
        "query": "How does PagedAttention improve LLM serving efficiency?",
        "expected_doc_ids": ["doc_005"],
        "relevant_chunks": [
            "allocates KV cache in non-contiguous pages",
            "reducing memory waste by 60-80%",
        ],
        "expected_answer": "PagedAttention manages GPU memory by allocating the KV cache in non-contiguous pages, reducing memory waste by 60-80% and enabling higher batch sizes.",
        "difficulty": "medium",
    },
    # ── Hard: requires cross-document reasoning or negation ──
    {
        "query": "Compare FAISS and Qdrant in terms of persistence and filtering capabilities.",
        "expected_doc_ids": ["doc_002"],
        "relevant_chunks": [
            "FAISS operates entirely in memory and does not provide built-in persistence",
            "Qdrant is a vector search engine with built-in persistence and filtering",
            "payload filtering which allows metadata-based pre-filtering",
        ],
        "expected_answer": "FAISS operates in memory without built-in persistence, while Qdrant provides built-in persistence via WAL and supports payload filtering for metadata-based pre-filtering before vector search.",
        "difficulty": "hard",
    },
    {
        "query": "What are the drawbacks of using Pinecone compared to self-hosted solutions?",
        "expected_doc_ids": ["doc_002"],
        "relevant_chunks": [
            "vendor lock-in and network latency due to its cloud-only architecture"
        ],
        "expected_answer": "Pinecone introduces vendor lock-in and network latency due to its cloud-only architecture, unlike self-hosted options like FAISS or Qdrant.",
        "difficulty": "hard",
    },
    {
        "query": "What monitoring should I set up for a RAG system in production?",
        "expected_doc_ids": ["doc_004"],
        "relevant_chunks": [
            "retrieval latency, reranking latency, and LLM generation time",
            "p99 latency exceeding SLO thresholds",
            "retrieval failures, reranker timeouts, and LLM circuit breaker trips",
        ],
        "expected_answer": "Monitor retrieval latency, reranking latency, and LLM generation time. Set Grafana alerts for p99 latency exceeding SLO thresholds and track error rates by type including retrieval failures, reranker timeouts, and circuit breaker trips.",
        "difficulty": "hard",
    },
    {
        "query": "How does continuous batching improve over traditional batching for LLM inference?",
        "expected_doc_ids": ["doc_005"],
        "relevant_chunks": [
            "processes new requests without waiting for the entire batch to complete",
            "reduces the time-to-first-token",
            "iteration-level scheduling",
        ],
        "expected_answer": "Continuous batching processes new requests without waiting for the full batch to complete, reducing time-to-first-token and improving GPU utilization through iteration-level scheduling.",
        "difficulty": "hard",
    },
    # ── Adversarial: unanswerable from the corpus ──
    {
        "query": "What is the maximum context window of GPT-4 Turbo?",
        "expected_doc_ids": [],
        "relevant_chunks": [],
        "expected_answer": "",
        "difficulty": "adversarial",
    },
    {
        "query": "How do I configure Elasticsearch sharding for a RAG pipeline?",
        "expected_doc_ids": [],
        "relevant_chunks": [],
        "expected_answer": "",
        "difficulty": "adversarial",
    },
]


def generate_test_data(
    output_path: str = "evaluation/test_data.jsonl",
    num_samples: int | None = None,
    include_adversarial: bool = True,
) -> tuple[str, int]:
    """
    Write JSONL test file.

    Returns (path, count) of samples written.
    """
    samples = SYNTHETIC_QUERIES
    if not include_adversarial:
        samples = [s for s in samples if s["difficulty"] != "adversarial"]
    if num_samples is not None:
        samples = samples[:num_samples]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return output_path, len(samples)


def get_corpus() -> list[dict]:
    """Return the synthetic document corpus."""
    return SYNTHETIC_DOCUMENTS
