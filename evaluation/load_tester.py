"""
evaluation/load_tester.py — Concurrent throughput and latency measurement.

Simulates production load by firing queries in parallel using
asyncio + ThreadPoolExecutor.  Measures:
  - Sustained throughput (QPS) under concurrency
  - Latency percentiles under load (which differ from sequential)
  - Error rate under saturation

Usage (standalone):
    python -m evaluation.load_tester --concurrency 4 --duration 30

Usage (from backtest.py):
    from evaluation.load_tester import run_load_test
    results = run_load_test(pipeline, queries, concurrency=4, duration_s=10)
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable


from evaluation.metrics import latency_stats


@dataclass
class LoadTestResult:
    """Results from a load test run."""

    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def throughput_qps(self) -> float:
        if self.duration_s <= 0:
            return 0.0
        return self.successful / self.duration_s

    @property
    def error_rate(self) -> float:
        if self.total_requests <= 0:
            return 0.0
        return self.failed / self.total_requests

    def latency_percentiles(self) -> dict[str, float]:
        return latency_stats(self.latencies_ms)

    def summary(self) -> dict:
        lat = self.latency_percentiles()
        return {
            "total_requests": self.total_requests,
            "successful": self.successful,
            "failed": self.failed,
            "error_rate": round(self.error_rate, 4),
            "duration_s": round(self.duration_s, 2),
            "throughput_qps": round(self.throughput_qps, 2),
            "latency_p50_ms": round(lat["p50"], 1),
            "latency_p95_ms": round(lat["p95"], 1),
            "latency_p99_ms": round(lat["p99"], 1),
            "latency_mean_ms": round(lat["mean"], 1),
            "latency_max_ms": round(lat["max"], 1),
        }


def _execute_query(query_fn: Callable, query: str) -> tuple[float, str | None]:
    """
    Run a single query, return (latency_ms, error_or_None).
    query_fn should accept a string and return any result or raise.
    """
    t0 = time.perf_counter()
    try:
        query_fn(query)
        elapsed = (time.perf_counter() - t0) * 1000
        return elapsed, None
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return elapsed, str(exc)


async def run_load_test(
    query_fn: Callable,
    queries: list[str],
    concurrency: int = 4,
    num_iterations: int | None = None,
    duration_s: float | None = None,
) -> LoadTestResult:
    """
    Run a load test against the query function.

    Either num_iterations or duration_s must be set.
      - num_iterations: run each query N times total, round-robin
      - duration_s:     keep sending queries for N seconds

    Args:
        query_fn:        callable(str) -> Any that executes a RAG query
        queries:         list of query strings to cycle through
        concurrency:     max concurrent queries
        num_iterations:  total number of queries to execute
        duration_s:      alternatively, run for this many seconds
    """
    result = LoadTestResult()

    if not queries:
        return result

    # Default: run each query once
    if num_iterations is None and duration_s is None:
        num_iterations = len(queries)

    loop = asyncio.get_event_loop()

    # Build the work queue
    if num_iterations is not None:
        work_queries = [queries[i % len(queries)] for i in range(num_iterations)]
    else:
        work_queries = None  # we'll generate on-the-fly

    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        if work_queries is not None:
            # Fixed number of iterations
            sem = asyncio.Semaphore(concurrency)

            async def _bounded_query(q: str):
                async with sem:
                    lat, err = await loop.run_in_executor(
                        pool, _execute_query, query_fn, q
                    )
                    result.total_requests += 1
                    result.latencies_ms.append(lat)
                    if err:
                        result.failed += 1
                        result.errors.append(err)
                    else:
                        result.successful += 1

            tasks = [_bounded_query(q) for q in work_queries]
            await asyncio.gather(*tasks)

        else:
            # Duration-based: keep sending until time runs out
            idx = 0
            sem = asyncio.Semaphore(concurrency)

            async def _timed_worker():
                nonlocal idx
                while time.perf_counter() - t_start < duration_s:
                    q = queries[idx % len(queries)]
                    idx += 1
                    async with sem:
                        lat, err = await loop.run_in_executor(
                            pool, _execute_query, query_fn, q
                        )
                        result.total_requests += 1
                        result.latencies_ms.append(lat)
                        if err:
                            result.failed += 1
                            result.errors.append(err)
                        else:
                            result.successful += 1

            # Launch workers
            workers = [_timed_worker() for _ in range(concurrency)]
            await asyncio.gather(*workers)

    result.duration_s = time.perf_counter() - t_start
    return result


def run_load_test_sync(
    query_fn: Callable,
    queries: list[str],
    concurrency: int = 4,
    num_iterations: int | None = None,
    duration_s: float | None = None,
) -> LoadTestResult:
    """Synchronous wrapper around run_load_test for non-async callers."""
    return asyncio.run(
        run_load_test(query_fn, queries, concurrency, num_iterations, duration_s)
    )
