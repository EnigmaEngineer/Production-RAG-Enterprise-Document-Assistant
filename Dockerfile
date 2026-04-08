# ============================================================================
# RAG Enterprise API Server
# Multi-stage build for minimal production image
# ============================================================================

# ── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: run as non-root
RUN groupadd -r rag && useradd -r -g rag -d /app -s /sbin/nologin rag

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY pyproject.toml .

# Create data directories
RUN mkdir -p /data/faiss_index && chown -R rag:rag /app /data

# Switch to non-root user
USER rag

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8000/healthz/live'); exit(0 if r.status_code==200 else 1)"

# Environment defaults
ENV RAG_HOST=0.0.0.0 \
    RAG_PORT=8000 \
    RAG_LOG_LEVEL=info \
    RAG_USE_DUMMY_LLM=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Graceful shutdown: uvicorn respects SIGTERM
STOPSIGNAL SIGTERM

CMD ["python", "-m", "uvicorn", "src.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-graceful-shutdown", "30"]
