"""
evaluation/metrics.py — Retrieval and generation metric computations.

All functions are stateless and operate on plain Python types so they can be
used independently of the RAG pipeline (e.g. in CI, notebooks, or external
evaluation harnesses).

Retrieval metrics
─────────────────
  hit_rate_at_k     — binary: ≥1 relevant doc in top-K?
  mrr               — 1 / rank of first relevant doc
  ndcg_at_k         — normalised discounted cumulative gain (graded relevance)
  precision_at_k    — fraction of top-K that are relevant
  recall_at_k       — fraction of relevant docs found in top-K

Generation metrics
──────────────────
  token_f1          — unigram precision/recall F1  (fast, no deps)
  rouge_l           — longest-common-subsequence F1 (requires rouge-score)
  llm_judge_score   — GPT-4/Claude relevance score stub (0-5 Likert scale)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence


# ═══════════════════════════════════════════════════════════════════════════
#  Retrieval metrics
# ═══════════════════════════════════════════════════════════════════════════

def hit_rate_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """Return 1.0 if at least one relevant doc appears in retrieved[:k], else 0.0."""
    relevant = set(relevant_ids)
    return 1.0 if any(rid in relevant for rid in retrieved_ids[:k]) else 0.0


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
) -> float:
    """1 / (position of first relevant doc).  0.0 if none found."""
    relevant = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(relevances: Sequence[float], k: int) -> float:
    """Discounted Cumulative Gain up to position k."""
    return sum(
        rel / math.log2(pos + 2)  # pos+2 because enumerate starts at 0
        for pos, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
    relevance_grades: dict[str, float] | None = None,
) -> float:
    """
    Normalised Discounted Cumulative Gain @ k.

    Args:
        retrieved_ids:    ordered list of doc IDs returned by retrieval
        relevant_ids:     set of doc IDs considered relevant
        k:                cutoff
        relevance_grades: optional mapping doc_id → grade (default: 1.0 for
                          every relevant doc, 0.0 otherwise → binary relevance)
    Returns:
        NDCG score in [0, 1].  Returns 0.0 when no relevant docs exist.
    """
    if not relevant_ids:
        return 0.0

    relevant = set(relevant_ids)

    # Build relevance vector for retrieved list
    if relevance_grades:
        gains = [relevance_grades.get(rid, 0.0) for rid in retrieved_ids[:k]]
    else:
        gains = [1.0 if rid in relevant else 0.0 for rid in retrieved_ids[:k]]

    dcg = _dcg(gains, k)

    # Ideal ordering: sort all relevant grades descending
    if relevance_grades:
        ideal = sorted(
            [relevance_grades.get(rid, 0.0) for rid in relevant_ids],
            reverse=True,
        )
    else:
        ideal = [1.0] * len(relevant_ids)

    idcg = _dcg(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """Fraction of top-K results that are relevant."""
    relevant = set(relevant_ids)
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for r in top_k if r in relevant) / len(top_k)


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """Fraction of relevant docs that appear in top-K."""
    if not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    top_k = set(retrieved_ids[:k])
    return len(relevant & top_k) / len(relevant)


# ═══════════════════════════════════════════════════════════════════════════
#  Generation / answer-quality metrics
# ═══════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenizer with punctuation stripping."""
    return re.findall(r"\b\w+\b", text.lower())


def token_f1(prediction: str, reference: str) -> dict[str, float]:
    """
    Unigram token-level precision, recall, and F1.

    Returns dict with keys: precision, recall, f1.
    """
    pred_tokens = Counter(_tokenize(prediction))
    ref_tokens = Counter(_tokenize(reference))

    if not ref_tokens:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if not pred_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Intersection count (handles repeated tokens correctly)
    common = sum((pred_tokens & ref_tokens).values())

    precision = common / sum(pred_tokens.values())
    recall = common / sum(ref_tokens.values())
    if precision + recall == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_l(prediction: str, reference: str) -> dict[str, float]:
    """
    ROUGE-L (longest common subsequence) F1.

    Uses the `rouge-score` library if available, otherwise falls back to a
    pure-Python LCS implementation.

    Returns dict with keys: precision, recall, f1.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        rl = scores["rougeL"]
        return {"precision": rl.precision, "recall": rl.recall, "f1": rl.fmeasure}
    except ImportError:
        # Pure-Python fallback
        return _rouge_l_fallback(prediction, reference)


def _rouge_l_fallback(prediction: str, reference: str) -> dict[str, float]:
    """LCS-based ROUGE-L without external dependencies."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Dynamic programming LCS length
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    precision = lcs_len / m if m > 0 else 0.0
    recall = lcs_len / n if n > 0 else 0.0
    if precision + recall == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def llm_judge_score(
    query: str,
    prediction: str,
    reference: str,
    model: str = "gpt-4o",
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict[str, float | str]:
    """
    Use an LLM (GPT-4, Claude, etc.) to score answer quality on a 1–5 scale.

    This is a *stub* that shows the integration pattern.  To use it for real:
      1. pip install openai
      2. Set OPENAI_API_KEY (or pass api_key)
      3. Call with model="gpt-4o" or model="claude-sonnet-4-20250514"

    The prompt asks the judge to evaluate:
      - Relevance:    Does the answer address the query?
      - Faithfulness:  Is everything in the answer supported by the reference?
      - Completeness:  Does it cover the key points from the reference?

    Returns:
        {"score": float 1-5, "reasoning": str, "error": str|None}
    """
    judge_prompt = f"""You are an expert evaluator for a document question-answering system.

Given a query, a reference answer, and a generated answer, score the generated
answer on a 1–5 scale:

  1 = Completely wrong or irrelevant
  2 = Partially relevant but mostly incorrect
  3 = Relevant but missing key information or partially incorrect
  4 = Mostly correct and relevant with minor omissions
  5 = Fully correct, relevant, and complete

Query:     {query}
Reference: {reference}
Generated: {prediction}

Respond with ONLY a JSON object: {{"score": <int 1-5>, "reasoning": "<brief explanation>"}}"""

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key or "not-set",
            base_url=api_base,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=200,
            temperature=0.0,
        )
        import json
        text = response.choices[0].message.content or ""
        # Strip markdown code fences if present
        text = re.sub(r"```json\s*|\s*```", "", text).strip()
        parsed = json.loads(text)
        return {
            "score": float(parsed.get("score", 0)),
            "reasoning": parsed.get("reasoning", ""),
            "error": None,
        }
    except ImportError:
        return {
            "score": 0.0,
            "reasoning": "openai package not installed — LLM judge unavailable",
            "error": "missing_dependency",
        }
    except Exception as exc:
        return {
            "score": 0.0,
            "reasoning": str(exc),
            "error": "api_error",
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Latency statistics helper
# ═══════════════════════════════════════════════════════════════════════════

def latency_stats(latencies_ms: Sequence[float]) -> dict[str, float]:
    """Compute p50/p95/p99/mean/min/max from a list of latencies."""
    import numpy as np
    arr = np.array(latencies_ms, dtype=np.float64)
    if len(arr) == 0:
        return {k: 0.0 for k in ["min", "max", "mean", "p50", "p95", "p99", "std"]}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "std": float(np.std(arr)),
    }
