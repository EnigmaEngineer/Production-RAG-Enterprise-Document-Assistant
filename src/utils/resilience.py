"""
Resilience utilities: circuit breakers and retry policies.
"""

from __future__ import annotations

import pybreaker
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)

from src.config import settings


# ---------------------------------------------------------------------------
# Circuit breakers — one per external dependency
# ---------------------------------------------------------------------------

llm_breaker = pybreaker.CircuitBreaker(
    fail_max=settings.cb_fail_max,
    reset_timeout=settings.cb_reset_timeout_s,
    name="llm",
)

vector_store_breaker = pybreaker.CircuitBreaker(
    fail_max=settings.cb_fail_max,
    reset_timeout=settings.cb_reset_timeout_s,
    name="vector_store",
)

reranker_breaker = pybreaker.CircuitBreaker(
    fail_max=settings.cb_fail_max,
    reset_timeout=settings.cb_reset_timeout_s,
    name="reranker",
)


# ---------------------------------------------------------------------------
# Retry decorator for LLM calls
# ---------------------------------------------------------------------------

llm_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=0.5, max=4, jitter=0.5),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
    reraise=True,
)
