"""
FastAPI application — the main entry point.

Routes:
  POST /v1/ingest   — ingest a document
  POST /v1/query    — ask a question (RAG pipeline)
  GET  /v1/stats    — index statistics
  GET  /healthz/ready — readiness probe
  GET  /healthz/live  — liveness probe
  GET  /metrics      — Prometheus metrics
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_fastapi_instrumentator import Instrumentator

from src.config import settings
from src.models.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    HealthCheck,
)
from src.api.pipeline import pipeline
from src.utils.logging import setup_logging, get_logger

log = get_logger(__name__)
security = HTTPBearer(auto_error=False)


# ── Lifespan ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    setup_logging(log_level=settings.log_level, json_format=True)
    log.info("app.starting", version=settings.app_version)
    pipeline.startup()
    yield
    log.info("app.shutting_down")
    pipeline.shutdown()


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    excluded_handlers=["/healthz/*", "/metrics"],
).instrument(app).expose(app, endpoint="/metrics")


# ── Auth dependency ─────────────────────────────────────────────────────────


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> str:
    """Validate Bearer token against configured API key."""
    if settings.api_key == "dev-key-change-me":
        return "dev"  # Skip auth in development
    if credentials is None or credentials.credentials != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return credentials.credentials


# ── Middleware ──────────────────────────────────────────────────────────────


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Inject X-Request-ID for correlation."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    import structlog

    structlog.contextvars.bind_contextvars(request_id=request_id)

    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(round(elapsed, 1))

    log.info(
        "http.request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=round(elapsed, 1),
    )
    structlog.contextvars.unbind_contextvars("request_id")
    return response


# ── Health routes ───────────────────────────────────────────────────────────


@app.get("/healthz/live", response_model=HealthCheck, tags=["health"])
async def liveness():
    """Liveness probe — returns 200 if process is alive."""
    return HealthCheck(status="ok", version=settings.app_version)


@app.get("/healthz/ready", response_model=HealthCheck, tags=["health"])
async def readiness():
    """Readiness probe — returns 200 only when all components are loaded."""
    if not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    components = {
        "pipeline": "ready" if pipeline.is_ready else "not_ready",
    }
    return HealthCheck(
        status="ok",
        version=settings.app_version,
        components=components,
    )


# ── Business routes ────────────────────────────────────────────────────────


@app.post("/v1/ingest", response_model=IngestResponse, tags=["documents"])
async def ingest_document(
    request: IngestRequest,
    _api_key: str = Depends(verify_api_key),
):
    """
    Ingest a document: chunk it, embed it, add to vector + BM25 indexes.
    """
    if not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    try:
        result = pipeline.ingest(request)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.error("ingest.error", error=str(exc))
        raise HTTPException(status_code=500, detail="Ingestion failed")


@app.post("/v1/query", response_model=QueryResponse, tags=["query"])
async def query_documents(
    request: QueryRequest,
    _api_key: str = Depends(verify_api_key),
):
    """
    Query the RAG pipeline: retrieve, rerank, generate answer with citations.
    """
    if not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    try:
        result = pipeline.query(request)
        return result
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    except Exception as exc:
        log.error("query.error", error=str(exc))
        raise HTTPException(status_code=500, detail="Query failed")


@app.get("/v1/stats", tags=["admin"])
async def get_stats(_api_key: str = Depends(verify_api_key)):
    """Return index statistics."""
    return pipeline.stats()
