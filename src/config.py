"""
Application configuration.

Priority (highest wins): environment variables > config.yaml > defaults.
All env vars use the RAG_ prefix (e.g. RAG_CHUNK_SIZE=256).
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    FIXED_OVERLAP = "fixed_overlap"


class VectorBackend(str, Enum):
    FAISS = "faiss"
    QDRANT = "qdrant"


def _load_yaml_overrides(path: str | Path | None = None) -> dict:
    """Load a YAML config file and return a flat dict of overrides."""
    if path is None:
        path = os.getenv("RAG_CONFIG_FILE")
    if not path or not Path(path).is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    # Flatten: skip nested keys like 'gates' (handled by eval separately)
    return {k: v for k, v in data.items() if not isinstance(v, dict)}


class Settings(BaseSettings):
    """All settings are overridable via environment variables or config.yaml."""

    # --- API ---
    app_name: str = "RAG Enterprise Document Assistant"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    api_key: str = Field(
        default="dev-key-change-me", description="Bearer token for auth"
    )
    request_timeout_s: float = 30.0

    # --- Chunking ---
    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 128
    semantic_threshold: float = 0.75

    # --- Embedding ---
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    embedding_batch_size: int = 64

    # --- Vector store ---
    vector_backend: VectorBackend = VectorBackend.FAISS
    faiss_index_path: str = "/data/faiss_index"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "documents"

    # --- BM25 ---
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_top_k: int = 100

    # --- Retrieval ---
    vector_top_k: int = 100
    rerank_top_k: int = 10
    hybrid_alpha: float = 0.5

    # --- Reranker ---
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_batch_size: int = 25
    reranker_timeout_s: float = 2.0
    reranker_enabled: bool = True

    # --- LLM ---
    llm_base_url: str = "http://vllm:8000/v1"
    llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1
    llm_timeout_s: float = 20.0

    # --- Circuit breaker ---
    cb_fail_max: int = 5
    cb_reset_timeout_s: int = 30

    # --- Rate limiting ---
    rate_limit_rpm: int = 60

    model_config = {"env_prefix": "RAG_", "case_sensitive": False}


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Create Settings with optional YAML file overrides."""
    overrides = _load_yaml_overrides(config_path)
    return Settings(**overrides)


def load_gate_thresholds(config_path: str | Path | None = None) -> dict[str, float]:
    """Load evaluation gate thresholds from a YAML config file."""
    defaults = {
        "hit_rate_at_5": 0.75,
        "mrr": 0.65,
        "ndcg_at_5": 0.60,
        "latency_p95_ms": 2000.0,
        "error_rate": 0.05,
    }
    if not config_path or not Path(config_path).is_file():
        return defaults
    try:
        import yaml
    except ImportError:
        return defaults
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    gates = data.get("gates", {})
    for k, v in gates.items():
        if k in defaults:
            defaults[k] = float(v)
    return defaults


# Singleton — import this everywhere
settings = load_settings()
