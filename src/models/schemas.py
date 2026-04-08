"""
Shared Pydantic models used across the application.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Document & chunk models
# ---------------------------------------------------------------------------


class DocumentMetadata(BaseModel):
    """Metadata attached to every ingested document."""

    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""
    title: str = ""
    page_number: int | None = None
    chunk_index: int = 0
    chunk_strategy: str = "recursive"
    total_chunks: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A single text chunk with its metadata and optional embedding."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    embedding: list[float] | None = None

    # Retrieval scores (populated during search)
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Request body for document ingestion."""

    text: str = Field(..., max_length=500_000)
    filename: str = "unknown.txt"
    title: str = ""
    chunk_strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 128
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    num_chunks: int
    chunk_strategy: str
    message: str = "Ingestion successful"


class QueryRequest(BaseModel):
    """Request body for RAG query."""

    query: str = Field(..., max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    rerank: bool = True
    filters: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """A citation pointing back to a source chunk."""

    chunk_id: str
    doc_id: str
    filename: str
    page_number: int | None
    chunk_index: int
    text_snippet: str = Field(description="The first 200 chars of the chunk")
    relevance_score: float


class QueryResponse(BaseModel):
    """Full response to a RAG query, including citations."""

    answer: str
    citations: list[Citation]
    query: str
    model: str
    latency_ms: float
    retrieval_latency_ms: float
    rerank_latency_ms: float
    generation_latency_ms: float
    num_chunks_retrieved: int
    num_chunks_after_rerank: int


# ---------------------------------------------------------------------------
# Health models
# ---------------------------------------------------------------------------


class HealthCheck(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    components: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation models
# ---------------------------------------------------------------------------


class EvalSample(BaseModel):
    """A single evaluation sample from a JSONL test set."""

    query: str
    expected_doc_ids: list[str] = Field(default_factory=list)
    expected_answer: str = ""
    expected_chunks: list[str] = Field(default_factory=list)


class EvalMetrics(BaseModel):
    """Aggregated evaluation metrics."""

    hit_rate_at_k: float = 0.0
    mrr: float = 0.0
    answer_correctness: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    num_samples: int = 0
    config: dict[str, Any] = Field(default_factory=dict)
