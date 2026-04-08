#!/usr/bin/env bash
# ==========================================================================
# backtest.sh — Run the RAG evaluation pipeline
#
# Usage:
#   ./scripts/backtest.sh                           # full matrix, synthetic data
#   ./scripts/backtest.sh --single-config recursive_512_rerank
#   ./scripts/backtest.sh --test-file data.jsonl     # custom test set
#   ./scripts/backtest.sh --load-test --concurrency 8
#   ./scripts/backtest.sh --llm-judge               # needs OPENAI_API_KEY
#
# Gate thresholds (override via env):
#   GATE_MRR=0.80  GATE_HIT5=0.90  GATE_P95_MS=500  ./scripts/backtest.sh
#
# Exit codes:
#   0 — all production gates passed
#   1 — at least one gate failed (blocks CI merge)
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "========================================================================"
echo "  RAG Enterprise — Backtesting Framework"
echo "========================================================================"
echo ""

# ── Check Python ──────────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: $PYTHON not found. Install Python 3.11+."
    exit 1
fi
echo "  Python:  $($PYTHON --version)"

# ── Install dependencies if needed ────────────────────────────────────
install_if_missing() {
    if ! $PYTHON -c "import $1" 2>/dev/null; then
        echo "  Installing $1..."
        pip install --break-system-packages -q "$2" 2>/dev/null || \
        pip install -q "$2" 2>/dev/null || true
    fi
}

install_if_missing fastapi fastapi
install_if_missing tiktoken tiktoken
install_if_missing numpy numpy
install_if_missing rank_bm25 rank-bm25

# rouge-score is optional (fallback exists)
install_if_missing rouge_score rouge-score 2>/dev/null || true

# ── Environment ───────────────────────────────────────────────────────
export RAG_USE_DUMMY_LLM="${RAG_USE_DUMMY_LLM:-true}"
export RAG_RERANKER_ENABLED="${RAG_RERANKER_ENABLED:-false}"
export RAG_LOG_LEVEL="${RAG_LOG_LEVEL:-warning}"

echo "  Dummy LLM:     $RAG_USE_DUMMY_LLM"
echo "  Reranker:      $RAG_RERANKER_ENABLED"
echo "  Log level:     $RAG_LOG_LEVEL"

# ── Gate thresholds ───────────────────────────────────────────────────
echo ""
echo "  Gate thresholds:"
echo "    MRR          >= ${GATE_MRR:-0.65}"
echo "    Hit@5        >= ${GATE_HIT5:-0.75}"
echo "    NDCG@5       >= ${GATE_NDCG5:-0.60}"
echo "    Latency p95  <= ${GATE_P95_MS:-2000} ms"
echo "    Error rate   <= ${GATE_ERROR_RATE:-0.05}"

# ── Run ───────────────────────────────────────────────────────────────
echo ""
echo "  Running evaluation..."
echo ""

# Default args: generate synthetic, single config to stay within memory
DEFAULT_ARGS="--generate-synthetic --single-config recursive_512_no_rerank"

# If user passed arguments, use those instead
if [ $# -gt 0 ]; then
    BACKTEST_ARGS="$@"
else
    BACKTEST_ARGS="$DEFAULT_ARGS"
fi

$PYTHON evaluation/backtest.py \
    --output evaluation/results.json \
    $BACKTEST_ARGS

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  RESULT: ALL GATES PASSED"
else
    echo "  RESULT: GATE FAILURE (exit code $EXIT_CODE)"
fi
echo "========================================================================"
echo ""
echo "  Artifacts:"
echo "    Results:   evaluation/results.json"
echo "    Test data: evaluation/test_data.jsonl"
echo ""

exit $EXIT_CODE
