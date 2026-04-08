#!/usr/bin/env python3
"""
backtest.py — Production evaluation framework for the RAG pipeline.

Computes:
  Retrieval :  Hit Rate@3, Hit Rate@5, MRR, NDCG@3, NDCG@5,
               Precision@5, Recall@5
  Generation:  Token F1, ROUGE-L F1, LLM-as-Judge (opt-in via --llm-judge)
  System    :  p50/p95/p99 latency, throughput under concurrency,
               error rate, per-stage timing breakdown

Outputs:
  * Formatted tables to stdout
  * evaluation/results.json  — full metrics + per-query breakdown
  * exit code 0 if all gate thresholds pass, 1 otherwise (CI-friendly)

Usage:
  # Generate synthetic data + run default matrix
  python evaluation/backtest.py --generate-synthetic

  # Run on your own test set
  python evaluation/backtest.py --test-file my_tests.jsonl

  # Run with load testing (4 concurrent workers, 3 rounds)
  python evaluation/backtest.py --generate-synthetic --load-test --concurrency 4

  # Enable LLM-as-judge scoring (requires OPENAI_API_KEY)
  python evaluation/backtest.py --generate-synthetic --llm-judge

  # Single config (skip matrix)
  python evaluation/backtest.py --generate-synthetic --single-config recursive_512_rerank
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import (
    hit_rate_at_k,
    reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    token_f1,
    rouge_l,
    llm_judge_score,
    latency_stats,
)
from evaluation.test_data_generator import (
    generate_test_data,
    get_corpus,
)
from evaluation.load_tester import run_load_test_sync

from src.chunking.engine import get_chunker
from src.retrieval.engine import FAISSStore, BM25Index, HybridRetriever
from src.retrieval.embeddings import get_embedding_service
from src.reranker.engine import get_reranker
from src.llm.client import DummyLLMClient


# ═══════════════════════════════════════════════════════════════════════════
#  Production readiness gate thresholds
#  (override with env: GATE_MRR=0.80, GATE_HIT5=0.90, etc.)
# ═══════════════════════════════════════════════════════════════════════════

GATE_THRESHOLDS = {
    "hit_rate_at_5": float(os.getenv("GATE_HIT5", "0.75")),
    "mrr": float(os.getenv("GATE_MRR", "0.65")),
    "ndcg_at_5": float(os.getenv("GATE_NDCG5", "0.60")),
    "latency_p95_ms": float(os.getenv("GATE_P95_MS", "2000")),
    "error_rate": float(os.getenv("GATE_ERROR_RATE", "0.05")),
}


# ═══════════════════════════════════════════════════════════════════════════
#  Per-query result container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class QueryResult:
    """Full evaluation result for a single query."""
    query: str
    difficulty: str
    expected_doc_ids: list[str]
    retrieved_doc_ids: list[str]

    # Retrieval metrics
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    rr: float = 0.0
    ndcg_3: float = 0.0
    ndcg_5: float = 0.0
    precision_5: float = 0.0
    recall_5: float = 0.0

    # Generation metrics
    token_f1_score: float = 0.0
    token_precision: float = 0.0
    token_recall: float = 0.0
    rouge_l_f1: float = 0.0
    rouge_l_precision: float = 0.0
    rouge_l_recall: float = 0.0
    llm_judge: float = 0.0
    llm_judge_reasoning: str = ""

    # Timing (ms)
    total_latency_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0

    # Generated answer
    answer: str = ""
    config_label: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "difficulty": self.difficulty,
            "expected_doc_ids": self.expected_doc_ids,
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "hit_at_3": self.hit_at_3,
            "hit_at_5": self.hit_at_5,
            "mrr": self.rr,
            "ndcg_at_3": self.ndcg_3,
            "ndcg_at_5": self.ndcg_5,
            "precision_at_5": self.precision_5,
            "recall_at_5": self.recall_5,
            "token_f1": self.token_f1_score,
            "rouge_l_f1": self.rouge_l_f1,
            "llm_judge_score": self.llm_judge,
            "latency_ms": self.total_latency_ms,
            "retrieval_ms": self.retrieval_ms,
            "rerank_ms": self.rerank_ms,
            "generation_ms": self.generation_ms,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline builder (constructs a fresh pipeline for each config)
# ═══════════════════════════════════════════════════════════════════════════

class EvalPipeline:
    """
    Lightweight pipeline that shares the embedding service across configs
    but creates fresh indexes for each evaluation run.
    """

    def __init__(self, embed_svc, use_reranker: bool = False):
        self.embed_svc = embed_svc
        self.vector_store = FAISSStore(dim=embed_svc.dim)
        self.bm25_index = BM25Index()
        self.retriever = HybridRetriever(
            self.vector_store, self.bm25_index, embed_svc.encode
        )
        self.reranker = get_reranker(enabled=use_reranker, device="cpu")
        self.llm = DummyLLMClient()

    def ingest_corpus(
        self,
        documents: list[dict],
        chunk_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> int:
        """Ingest all documents, returns total chunk count."""
        chunker = get_chunker(
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        total = 0
        for doc in documents:
            for page_num, page_text in enumerate(doc["pages"]):
                chunks = chunker.chunk(
                    text=page_text,
                    doc_id=doc["doc_id"],
                    filename=doc["filename"],
                    title=doc["title"],
                )
                for c in chunks:
                    c.metadata.page_number = page_num + 1

                texts = [c.text for c in chunks]
                embeddings = self.embed_svc.encode_batch(texts)
                for c, emb in zip(chunks, embeddings):
                    c.embedding = emb.tolist()

                self.retriever.ingest(chunks)
                total += len(chunks)
        return total

    def query(self, query_text: str, top_k: int = 5) -> dict:
        """Execute a full RAG query, returning structured timing data."""
        t0 = time.perf_counter()

        # Retrieve
        t_ret = time.perf_counter()
        candidates = self.retriever.search(query_text, vector_top_k=50, bm25_top_k=50)
        retrieval_ms = (time.perf_counter() - t_ret) * 1000

        # Rerank
        t_rr = time.perf_counter()
        reranked = self.reranker.rerank(query_text, candidates, top_k=top_k)
        rerank_ms = (time.perf_counter() - t_rr) * 1000

        # Generate
        t_gen = time.perf_counter()
        answer, _ = self.llm.generate(query_text, reranked)
        gen_ms = (time.perf_counter() - t_gen) * 1000

        total_ms = (time.perf_counter() - t0) * 1000

        return {
            "answer": answer,
            "retrieved_doc_ids": [c.metadata.doc_id for c in reranked],
            "chunks": reranked,
            "total_ms": total_ms,
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "generation_ms": gen_ms,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Core evaluation runner
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_config(
    test_samples: list[dict],
    documents: list[dict],
    embed_svc,
    chunk_strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    use_reranker: bool = False,
    top_k: int = 5,
    label: str = "",
    use_llm_judge: bool = False,
) -> list[QueryResult]:
    """Run full evaluation for one configuration."""
    cfg_label = label or f"{chunk_strategy}_{chunk_size}_{'rr' if use_reranker else 'norr'}"

    print(f"\n{'─'*72}")
    print(f"  Config: {cfg_label}")
    print(f"  Chunk:  {chunk_strategy}  size={chunk_size}  overlap={chunk_overlap}")
    print(f"  Reranker: {'ON' if use_reranker else 'OFF'}")
    print(f"{'─'*72}")

    pipeline = EvalPipeline(embed_svc, use_reranker=use_reranker)
    total_chunks = pipeline.ingest_corpus(
        documents, chunk_strategy, chunk_size, chunk_overlap
    )
    print(f"  Ingested {len(documents)} docs -> {total_chunks} chunks")

    results: list[QueryResult] = []

    for sample in test_samples:
        query = sample["query"]
        expected_ids = sample.get("expected_doc_ids", [])
        expected_answer = sample.get("expected_answer", "")
        difficulty = sample.get("difficulty", "unknown")

        qr = pipeline.query(query, top_k=top_k)
        doc_ids = qr["retrieved_doc_ids"]

        result = QueryResult(
            query=query,
            difficulty=difficulty,
            expected_doc_ids=expected_ids,
            retrieved_doc_ids=doc_ids[:top_k],
            config_label=cfg_label,
            answer=qr["answer"],
            total_latency_ms=qr["total_ms"],
            retrieval_ms=qr["retrieval_ms"],
            rerank_ms=qr["rerank_ms"],
            generation_ms=qr["generation_ms"],
        )

        # ── Retrieval metrics (skip for adversarial) ──────────────────
        if expected_ids:
            result.hit_at_3 = hit_rate_at_k(doc_ids, expected_ids, k=3)
            result.hit_at_5 = hit_rate_at_k(doc_ids, expected_ids, k=5)
            result.rr = reciprocal_rank(doc_ids, expected_ids)
            result.ndcg_3 = ndcg_at_k(doc_ids, expected_ids, k=3)
            result.ndcg_5 = ndcg_at_k(doc_ids, expected_ids, k=5)
            result.precision_5 = precision_at_k(doc_ids, expected_ids, k=5)
            result.recall_5 = recall_at_k(doc_ids, expected_ids, k=5)

        # ── Generation metrics ────────────────────────────────────────
        if expected_answer:
            tf1 = token_f1(qr["answer"], expected_answer)
            result.token_f1_score = tf1["f1"]
            result.token_precision = tf1["precision"]
            result.token_recall = tf1["recall"]

            rl = rouge_l(qr["answer"], expected_answer)
            result.rouge_l_f1 = rl["f1"]
            result.rouge_l_precision = rl["precision"]
            result.rouge_l_recall = rl["recall"]

            if use_llm_judge:
                judge = llm_judge_score(query, qr["answer"], expected_answer)
                result.llm_judge = judge["score"]
                result.llm_judge_reasoning = judge.get("reasoning", "")

        results.append(result)

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Aggregation
# ═══════════════════════════════════════════════════════════════════════════

def aggregate(results: list[QueryResult]) -> dict[str, Any]:
    """Aggregate per-query results into config-level metrics."""
    if not results:
        return {}

    answerable = [r for r in results if r.expected_doc_ids]
    adversarial = [r for r in results if not r.expected_doc_ids]

    lat = latency_stats([r.total_latency_ms for r in results])
    ret_lat = latency_stats([r.retrieval_ms for r in results])
    rr_lat = latency_stats([r.rerank_ms for r in results])
    gen_lat = latency_stats([r.generation_ms for r in results])

    m: dict[str, Any] = {
        "config": results[0].config_label,
        "num_samples": len(results),
        "num_answerable": len(answerable),
        "num_adversarial": len(adversarial),
    }

    if answerable:
        m.update({
            "hit_rate_at_3": float(np.mean([r.hit_at_3 for r in answerable])),
            "hit_rate_at_5": float(np.mean([r.hit_at_5 for r in answerable])),
            "mrr": float(np.mean([r.rr for r in answerable])),
            "ndcg_at_3": float(np.mean([r.ndcg_3 for r in answerable])),
            "ndcg_at_5": float(np.mean([r.ndcg_5 for r in answerable])),
            "precision_at_5": float(np.mean([r.precision_5 for r in answerable])),
            "recall_at_5": float(np.mean([r.recall_5 for r in answerable])),
            "token_f1": float(np.mean([r.token_f1_score for r in answerable
                                       if r.token_f1_score > 0] or [0.0])),
            "rouge_l_f1": float(np.mean([r.rouge_l_f1 for r in answerable
                                         if r.rouge_l_f1 > 0] or [0.0])),
        })
        judges = [r.llm_judge for r in answerable if r.llm_judge > 0]
        m["llm_judge_avg"] = float(np.mean(judges)) if judges else None
    else:
        for k in ["hit_rate_at_3", "hit_rate_at_5", "mrr", "ndcg_at_3",
                   "ndcg_at_5", "precision_at_5", "recall_at_5",
                   "token_f1", "rouge_l_f1", "llm_judge_avg"]:
            m[k] = 0.0

    m.update({
        "latency_p50_ms": lat["p50"],
        "latency_p95_ms": lat["p95"],
        "latency_p99_ms": lat["p99"],
        "latency_mean_ms": lat["mean"],
        "retrieval_p50_ms": ret_lat["p50"],
        "rerank_p50_ms": rr_lat["p50"],
        "generation_p50_ms": gen_lat["p50"],
        "throughput_qps": 1000.0 / lat["mean"] if lat["mean"] > 0 else 0,
    })

    # Per-difficulty breakdown
    difficulty_groups = defaultdict(list)
    for r in results:
        difficulty_groups[r.difficulty].append(r)

    m["by_difficulty"] = {}
    for diff, group in difficulty_groups.items():
        ans = [r for r in group if r.expected_doc_ids]
        m["by_difficulty"][diff] = {
            "count": len(group),
            "hit_at_5": float(np.mean([r.hit_at_5 for r in ans])) if ans else None,
            "mrr": float(np.mean([r.rr for r in ans])) if ans else None,
            "token_f1": float(np.mean([r.token_f1_score for r in ans
                                       if r.token_f1_score > 0] or [0.0])) if ans else None,
        }

    return m


# ═══════════════════════════════════════════════════════════════════════════
#  Gate check
# ═══════════════════════════════════════════════════════════════════════════

def check_gates(metrics: dict, thresholds: dict | None = None) -> list[dict]:
    """Compare metrics against production readiness thresholds."""
    thresholds = thresholds or GATE_THRESHOLDS
    checks: list[dict] = []
    for metric_key, threshold in thresholds.items():
        value = metrics.get(metric_key)
        if value is None:
            checks.append({"metric": metric_key, "value": None,
                           "threshold": threshold, "passed": False,
                           "reason": "metric not computed"})
            continue
        if "latency" in metric_key or "error" in metric_key:
            passed = value <= threshold
        else:
            passed = value >= threshold
        checks.append({"metric": metric_key, "value": round(value, 4),
                        "threshold": threshold, "passed": passed})
    return checks


# ═══════════════════════════════════════════════════════════════════════════
#  Pretty printing
# ═══════════════════════════════════════════════════════════════════════════

def print_retrieval_table(all_metrics: list[dict]) -> None:
    print("\n" + "="*88)
    print("  RETRIEVAL METRICS")
    print("="*88)
    header = (f"{'Config':<30} {'Hit@3':>6} {'Hit@5':>6} {'MRR':>6} "
              f"{'NDCG@3':>7} {'NDCG@5':>7} {'P@5':>6} {'R@5':>6}")
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        print(f"{m['config']:<30} "
              f"{m.get('hit_rate_at_3',0):>6.3f} {m.get('hit_rate_at_5',0):>6.3f} "
              f"{m.get('mrr',0):>6.3f} {m.get('ndcg_at_3',0):>7.3f} "
              f"{m.get('ndcg_at_5',0):>7.3f} {m.get('precision_at_5',0):>6.3f} "
              f"{m.get('recall_at_5',0):>6.3f}")
    print("-" * len(header))


def print_generation_table(all_metrics: list[dict]) -> None:
    print("\n" + "="*60)
    print("  GENERATION METRICS")
    print("="*60)
    has_judge = any(m.get("llm_judge_avg") for m in all_metrics)
    header = f"{'Config':<30} {'Token-F1':>9} {'ROUGE-L':>8}"
    if has_judge:
        header += f" {'LLM-Judge':>10}"
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        line = f"{m['config']:<30} {m.get('token_f1',0):>9.3f} {m.get('rouge_l_f1',0):>8.3f}"
        if has_judge:
            j = m.get("llm_judge_avg")
            line += f" {j:>10.2f}" if j else f" {'N/A':>10}"
        print(line)
    print("-" * len(header))


def print_latency_table(all_metrics: list[dict]) -> None:
    print("\n" + "="*110)
    print("  LATENCY & THROUGHPUT")
    print("="*110)
    print(f"{'':<30} {'--- end-to-end (ms) ---':>28} {'':>6}  {'--- stage breakdown (ms) ---':>30}")
    header = (f"{'Config':<30} {'p50':>7} {'p95':>7} {'p99':>7} "
              f"{'mean':>7} {'QPS':>6}  {'ret':>8} {'rerank':>7} {'gen':>8}")
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        print(f"{m['config']:<30} "
              f"{m.get('latency_p50_ms',0):>7.1f} {m.get('latency_p95_ms',0):>7.1f} "
              f"{m.get('latency_p99_ms',0):>7.1f} {m.get('latency_mean_ms',0):>7.1f} "
              f"{m.get('throughput_qps',0):>6.1f}  "
              f"{m.get('retrieval_p50_ms',0):>8.1f} {m.get('rerank_p50_ms',0):>7.1f} "
              f"{m.get('generation_p50_ms',0):>8.1f}")
    print("-" * len(header))


def print_gate_results(checks: list[dict]) -> bool:
    print("\n" + "="*72)
    print("  PRODUCTION READINESS GATES")
    print("="*72)
    all_pass = True
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        if not c["passed"]:
            all_pass = False
        val = f"{c['value']:.4f}" if c["value"] is not None else "N/A"
        direction = "<=" if ("latency" in c["metric"] or "error" in c["metric"]) else ">="
        print(f"  [{status}]  {c['metric']:<20}  actual={val:<10}  threshold {direction} {c['threshold']}")
    verdict = "ALL GATES PASSED" if all_pass else "SOME GATES FAILED"
    print(f"\n  Result: {verdict}")
    return all_pass


def print_difficulty_breakdown(metrics: dict) -> None:
    by_diff = metrics.get("by_difficulty", {})
    if not by_diff:
        return
    print("\n  Per-difficulty breakdown:")
    for diff, vals in sorted(by_diff.items()):
        h5 = f"{vals['hit_at_5']:.3f}" if vals.get("hit_at_5") is not None else "N/A"
        mrr = f"{vals['mrr']:.3f}" if vals.get("mrr") is not None else "N/A"
        f1 = f"{vals['token_f1']:.3f}" if vals.get("token_f1") is not None else "N/A"
        print(f"    {diff:<14}  n={vals['count']:<3}  Hit@5={h5}  MRR={mrr}  F1={f1}")


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration matrix
# ═══════════════════════════════════════════════════════════════════════════

EVAL_CONFIGS = [
    {"label": "recursive_512_rerank",    "chunk_strategy": "recursive",     "chunk_size": 512, "chunk_overlap": 128, "use_reranker": True},
    {"label": "recursive_512_no_rerank", "chunk_strategy": "recursive",     "chunk_size": 512, "chunk_overlap": 128, "use_reranker": False},
    {"label": "recursive_256_rerank",    "chunk_strategy": "recursive",     "chunk_size": 256, "chunk_overlap": 64,  "use_reranker": True},
    {"label": "fixed_512_rerank",        "chunk_strategy": "fixed_overlap", "chunk_size": 512, "chunk_overlap": 128, "use_reranker": True},
    {"label": "fixed_256_rerank",        "chunk_strategy": "fixed_overlap", "chunk_size": 256, "chunk_overlap": 64,  "use_reranker": True},
]


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Backtesting & Evaluation")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml (loads gate thresholds and pipeline settings)")
    parser.add_argument("--test-file", "--test-set", type=str, help="JSONL test file path")
    parser.add_argument("--generate-synthetic", action="store_true")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="evaluation/results.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--single-config", type=str, default=None)
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--load-test", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--load-iterations", type=int, default=None)
    parser.add_argument("--no-gates", action="store_true")
    args = parser.parse_args()

    # Load gate thresholds from config.yaml if provided
    if args.config:
        from src.config import load_gate_thresholds
        custom_gates = load_gate_thresholds(args.config)
        GATE_THRESHOLDS.update(custom_gates)
        print(f"  Loaded config: {args.config}")

    print("=" * 72)
    print("  RAG Enterprise — Evaluation Framework v2.0")
    print("=" * 72)

    # ── Load test data ────────────────────────────────────────────────
    if args.generate_synthetic or not args.test_file:
        test_file = "evaluation/test_data.jsonl"
        _, count = generate_test_data(test_file, num_samples=args.num_samples)
        print(f"\n  Generated synthetic test data: {test_file} ({count} samples)")
    else:
        test_file = args.test_file

    with open(test_file) as f:
        test_samples = [json.loads(line) for line in f if line.strip()]
    print(f"  Loaded {len(test_samples)} test samples")

    documents = get_corpus()
    print(f"  Corpus: {len(documents)} documents, {sum(len(d['pages']) for d in documents)} pages")

    # ── Initialize shared embedding service once ──────────────────────
    print("\n  Loading embedding model (one-time)...")
    embed_svc = get_embedding_service()
    _ = embed_svc.encode("warmup")
    print(f"  Embedding model ready (dim={embed_svc.dim})")

    # ── Select configs ────────────────────────────────────────────────
    if args.single_config:
        configs = [c for c in EVAL_CONFIGS if c["label"] == args.single_config]
        if not configs:
            avail = [c['label'] for c in EVAL_CONFIGS]
            print(f"  ERROR: config '{args.single_config}' not found. Available: {avail}")
            sys.exit(1)
    else:
        configs = EVAL_CONFIGS

    # ── Run evaluation matrix ─────────────────────────────────────────
    all_metrics: list[dict] = []
    all_detailed: list[dict] = []
    best_config_label = ""

    for cfg in configs:
        results = evaluate_config(
            test_samples=test_samples, documents=documents, embed_svc=embed_svc,
            chunk_strategy=cfg["chunk_strategy"], chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"], use_reranker=cfg["use_reranker"],
            top_k=args.top_k, label=cfg["label"], use_llm_judge=args.llm_judge,
        )
        metrics = aggregate(results)
        all_metrics.append(metrics)
        all_detailed.append({
            "config": cfg,
            "metrics": {k: v for k, v in metrics.items() if k != "by_difficulty"},
            "by_difficulty": metrics.get("by_difficulty", {}),
            "per_query": [r.to_dict() for r in results],
        })
        if not best_config_label or metrics.get("mrr", 0) >= max(
            (m.get("mrr", 0) for m in all_metrics[:-1]), default=0
        ):
            best_config_label = cfg["label"]

    # ── Print tables ──────────────────────────────────────────────────
    print_retrieval_table(all_metrics)
    print_generation_table(all_metrics)
    print_latency_table(all_metrics)

    best_metrics = next(m for m in all_metrics if m["config"] == best_config_label)
    print_difficulty_breakdown(best_metrics)

    # ── Load test ─────────────────────────────────────────────────────
    load_result: dict | None = None
    if args.load_test:
        print("\n" + "="*72)
        print("  CONCURRENT LOAD TEST")
        print("="*72)

        best_cfg = next(c for c in configs if c["label"] == best_config_label)
        load_pipeline = EvalPipeline(embed_svc, use_reranker=best_cfg["use_reranker"])
        load_pipeline.ingest_corpus(
            documents, best_cfg["chunk_strategy"],
            best_cfg["chunk_size"], best_cfg["chunk_overlap"],
        )

        queries = [s["query"] for s in test_samples if s.get("expected_doc_ids")]
        n_iter = args.load_iterations or (len(queries) * 3)

        print(f"  Config:      {best_config_label}")
        print(f"  Concurrency: {args.concurrency}")
        print(f"  Iterations:  {n_iter}")
        print(f"  Queries:     {len(queries)} unique\n")

        lt = run_load_test_sync(
            query_fn=lambda q: load_pipeline.query(q, top_k=args.top_k),
            queries=queries, concurrency=args.concurrency,
            num_iterations=n_iter,
        )
        load_result = lt.summary()

        print(f"  Total requests:  {load_result['total_requests']}")
        print(f"  Successful:      {load_result['successful']}")
        print(f"  Failed:          {load_result['failed']}")
        print(f"  Error rate:      {load_result['error_rate']:.2%}")
        print(f"  Duration:        {load_result['duration_s']:.1f}s")
        print(f"  Throughput:      {load_result['throughput_qps']:.1f} QPS")
        print(f"  Latency p50:     {load_result['latency_p50_ms']:.1f} ms")
        print(f"  Latency p95:     {load_result['latency_p95_ms']:.1f} ms")
        print(f"  Latency p99:     {load_result['latency_p99_ms']:.1f} ms")
        print(f"  Latency max:     {load_result['latency_max_ms']:.1f} ms")

    # ── Gate checks ───────────────────────────────────────────────────
    gates_passed = True
    if not args.no_gates:
        gate_metrics = dict(best_metrics)
        if load_result:
            gate_metrics["error_rate"] = load_result["error_rate"]
            gate_metrics["latency_p95_ms"] = load_result["latency_p95_ms"]
        checks = check_gates(gate_metrics)
        gates_passed = print_gate_results(checks)

    # ── Save results ──────────────────────────────────────────────────
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_file": test_file, "num_samples": len(test_samples),
            "num_configs": len(configs), "top_k": args.top_k,
            "gate_thresholds": GATE_THRESHOLDS, "gates_passed": gates_passed,
        },
        "configs": all_detailed,
    }
    if load_result:
        output["load_test"] = load_result

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {args.output}")

    best = max(all_metrics, key=lambda m: m.get("mrr", 0))
    print(f"\n  Best config by MRR: {best['config']}")
    print(f"    MRR={best.get('mrr',0):.3f}  Hit@5={best.get('hit_rate_at_5',0):.3f}  "
          f"NDCG@5={best.get('ndcg_at_5',0):.3f}  ROUGE-L={best.get('rouge_l_f1',0):.3f}")

    if not args.no_gates and not gates_passed:
        print("\n  Exiting with code 1 (gate failure)")
        sys.exit(1)

    print("\n  Done.")


if __name__ == "__main__":
    main()
