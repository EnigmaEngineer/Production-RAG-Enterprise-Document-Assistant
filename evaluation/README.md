# Evaluation Framework (`evaluation/`)

Offline backtesting framework that measures retrieval quality, generation accuracy,
and system performance across multiple configurations. Designed to run in CI and
block merges when quality drops below production thresholds.

## Quick Start

```bash
# Default: single config, synthetic data, ~30s runtime
./scripts/backtest.sh

# Full 5-config matrix (needs ~2 min + more memory)
python evaluation/backtest.py --generate-synthetic

# Single config with load testing
python evaluation/backtest.py --generate-synthetic \
  --single-config recursive_512_no_rerank \
  --load-test --concurrency 4

# Your own test set
python evaluation/backtest.py --test-file my_tests.jsonl --top-k 5

# With LLM-as-judge (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... python evaluation/backtest.py \
  --generate-synthetic --llm-judge
```

## Files

| File | Purpose |
|------|---------|
| `backtest.py` | Main evaluation script — runs matrix, prints tables, checks gates |
| `metrics.py` | Pure-function metric library (NDCG, MRR, ROUGE-L, LLM-judge, etc.) |
| `test_data_generator.py` | Synthetic corpus + test queries with difficulty levels |
| `load_tester.py` | Concurrent throughput measurement via asyncio + ThreadPoolExecutor |

## Test Data Format (JSONL)

Each line is a JSON object with these fields:

```json
{
  "query": "How does Kubernetes autoscaling work?",
  "expected_doc_ids": ["doc_001"],
  "relevant_chunks": ["observed CPU utilization or custom metrics", "checks metrics every 15 seconds"],
  "expected_answer": "HPA scales pods based on CPU utilization or custom metrics, checking every 15 seconds.",
  "difficulty": "medium"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `query` | Yes | The question to evaluate |
| `expected_doc_ids` | Yes | Doc IDs that should appear in retrieval results. Empty `[]` for adversarial/unanswerable queries |
| `relevant_chunks` | No | Substring snippets that a correct retrieval must surface |
| `expected_answer` | No | Gold-standard answer for generation metrics |
| `difficulty` | No | `easy`, `medium`, `hard`, or `adversarial` — used for stratified reporting |

## Metrics Computed

### Retrieval Metrics

| Metric | Formula | What It Tells You |
|--------|---------|-------------------|
| **Hit Rate@K** | `1 if any(relevant in top-K) else 0`, averaged | "Can the system find the right document at all?" — the most forgiving retrieval metric. If this is low, nothing downstream works. |
| **MRR** | `mean(1 / rank_of_first_relevant)` | "How far down does the user have to look?" — penalises relevant results at rank 5 more than rank 1. The single most important retrieval metric for RAG because the LLM sees chunks in order. |
| **NDCG@K** | Normalised Discounted Cumulative Gain | "Are the best results at the top?" — the standard IR metric that handles graded relevance. With binary relevance (our default), it's similar to MRR but accounts for multiple relevant docs. |
| **Precision@K** | `relevant_in_top_K / K` | "What fraction of retrieved chunks are useful?" — matters for cost (each useless chunk wastes LLM context tokens). |
| **Recall@K** | `relevant_in_top_K / total_relevant` | "What fraction of relevant chunks did we find?" — matters for completeness when answers span multiple documents. |

### Generation Metrics

| Metric | Formula | What It Tells You |
|--------|---------|-------------------|
| **Token F1** | Unigram precision/recall F1 | Fast lexical overlap. Good for detecting catastrophic failures (F1 near 0) but insensitive to paraphrasing. Use as a lower bound. |
| **ROUGE-L** | Longest Common Subsequence F1 | Better than token F1 because it captures word order. Still lexical — two semantically identical sentences with different wording will score low. |
| **LLM-as-Judge** | GPT-4/Claude scores 1–5 on relevance, faithfulness, completeness | The gold standard for answer quality. Correlates best with human judgment but costs ~$0.01/eval and adds 1–2s latency. Use on a weekly sample, not every CI run. |

### System Metrics

| Metric | What It Tells You |
|--------|-------------------|
| **Latency p50/p95/p99** | Tail latency matters more than average. p95 is your SLO target; p99 catches edge cases (long queries, cold cache). |
| **Throughput (QPS)** | Sustained queries per second under concurrent load. Determines how many pods you need. |
| **Error rate** | Fraction of requests that raised exceptions. Must be < 5% for production. |
| **Stage breakdown** | Per-stage p50 (retrieval, rerank, generation) shows where latency is spent. |

## Production Readiness Thresholds

These are the default gate values. Override with environment variables.

| Metric | Threshold | Direction | Env Override | Rationale |
|--------|-----------|-----------|--------------|-----------|
| **MRR** | >= 0.65 | higher is better | `GATE_MRR` | Below 0.65 means the first relevant result is typically at rank 2+, which degrades answer quality because the LLM weighs earlier context more heavily. |
| **Hit Rate@5** | >= 0.75 | higher is better | `GATE_HIT5` | If 25%+ of queries can't find any relevant document in top-5, the system is not useful. Users will see "I don't have enough information" too often. |
| **NDCG@5** | >= 0.60 | higher is better | `GATE_NDCG5` | Ensures not just presence but correct ranking. A system with Hit@5=1.0 but NDCG@5=0.3 buries relevant results at rank 4–5. |
| **Latency p95** | <= 2000 ms | lower is better | `GATE_P95_MS` | 2-second p95 allows headroom for a real LLM (vLLM adds ~1–2s). Tighten to 500ms for retrieval-only evaluation. |
| **Error rate** | <= 0.05 | lower is better | `GATE_ERROR_RATE` | More than 5% errors under load indicates a stability problem (OOM, race condition, circuit breaker flapping). |

### When to Tighten Thresholds

As your system matures, tighten progressively:

| Stage | MRR | Hit@5 | NDCG@5 | p95 |
|-------|-----|-------|--------|-----|
| MVP / prototype | >= 0.50 | >= 0.60 | >= 0.40 | <= 5000 ms |
| **Production v1** (default) | **>= 0.65** | **>= 0.75** | **>= 0.60** | **<= 2000 ms** |
| Production v2 (mature) | >= 0.75 | >= 0.85 | >= 0.70 | <= 1000 ms |
| Tier 1 SLA | >= 0.85 | >= 0.95 | >= 0.80 | <= 500 ms |

### How to Set Custom Thresholds

```bash
# Strict thresholds for a release candidate
GATE_MRR=0.80 GATE_HIT5=0.90 GATE_P95_MS=500 ./scripts/backtest.sh

# Skip gate checks entirely (e.g. during development)
python evaluation/backtest.py --generate-synthetic --no-gates
```

## Configuration Matrix

The backtest runs multiple chunking × retrieval × reranking configs and compares:

| Label | Strategy | Size | Overlap | Reranker | Why |
|-------|----------|------|---------|----------|-----|
| `recursive_512_rerank` | recursive | 512 | 128 | ON | Production default |
| `recursive_512_no_rerank` | recursive | 512 | 128 | OFF | Ablation: how much does reranking help? |
| `recursive_256_rerank` | recursive | 256 | 64 | ON | Smaller chunks = more precise but more chunks |
| `fixed_512_rerank` | fixed_overlap | 512 | 128 | ON | Faster ingestion, does it hurt quality? |
| `fixed_256_rerank` | fixed_overlap | 256 | 64 | ON | Smallest chunks, fastest ingestion |

Run a single config to save time: `--single-config recursive_512_no_rerank`

## LLM-as-Judge Integration

The `llm_judge_score()` function in `metrics.py` supports any OpenAI-compatible API:

```python
from evaluation.metrics import llm_judge_score

# GPT-4o (default)
result = llm_judge_score(query, prediction, reference)

# Claude via Anthropic's OpenAI-compatible endpoint
result = llm_judge_score(
    query, prediction, reference,
    model="claude-sonnet-4-20250514",
    api_base="https://api.anthropic.com/v1/",
    api_key="sk-ant-..."
)

# Local vLLM instance
result = llm_judge_score(
    query, prediction, reference,
    model="meta-llama/Llama-3.1-8B-Instruct",
    api_base="http://localhost:8000/v1"
)
```

The judge prompt evaluates three dimensions (1–5 each):
- **Relevance**: Does the answer address the query?
- **Faithfulness**: Is everything supported by the reference?
- **Completeness**: Does it cover the key points?

**Cost estimate**: ~$0.01/query with GPT-4o. For a 100-query test set, expect ~$1 and ~3 minutes.

**Recommendation**: Use LLM-as-judge on a weekly scheduled run against a sample of production queries, not on every CI push. Token F1 and ROUGE-L are sufficient for regression detection in CI.

## Load Testing

The load tester measures throughput under realistic concurrency:

```bash
# 4 concurrent workers, 36 total queries (12 unique × 3 rounds)
python evaluation/backtest.py --generate-synthetic \
  --single-config recursive_512_no_rerank \
  --load-test --concurrency 4 --load-iterations 36
```

Output:
```
  CONCURRENT LOAD TEST
  Config:      recursive_512_no_rerank
  Concurrency: 4
  Iterations:  36
  Queries:     12 unique

  Total requests:  36
  Successful:      36
  Failed:          0
  Error rate:      0.00%
  Duration:        5.2s
  Throughput:      6.9 QPS
  Latency p50:     142.3 ms
  Latency p95:     198.7 ms
  Latency p99:     213.4 ms
```

The load tester uses `asyncio.gather` with a `ThreadPoolExecutor` to simulate
concurrent clients. This catches issues that sequential evaluation misses:
thread-safety bugs, lock contention on FAISS/BM25 indexes, and memory pressure
under concurrent embedding computations.

## Interpreting Results

### "My Hit@5 is high but MRR is low"

The system finds the right document but buries it at rank 3–5. This usually means
BM25 is promoting a lexically similar but wrong chunk above the semantically correct
one. Solutions: increase the hybrid alpha weight toward vectors, or enable reranking.

### "My ROUGE-L is low but the answers look correct"

ROUGE-L is lexical — it penalises paraphrasing. If your LLM rephrases the reference
answer, ROUGE-L will be low even though the answer is correct. Use LLM-as-judge for
a more accurate assessment, or switch to token recall (which only checks if reference
words appear in the prediction, ignoring extras).

### "My p50 is fine but p99 is 5× worse"

Tail latency spikes usually come from: (1) FAISS search on a cold CPU cache after
a period of inactivity, (2) BM25 scoring a very long query (> 50 tokens), or
(3) the embedding model processing an unusually long input. Check the per-stage
breakdown to identify which component causes the spike.

### "Adversarial queries are polluting my metrics"

The framework separates adversarial queries (those with empty `expected_doc_ids`)
from answerable queries. Retrieval metrics are computed only on answerable queries.
The per-difficulty breakdown shows performance stratified by difficulty level.

## CI Integration

Add to your CI pipeline (GitHub Actions example):

```yaml
- name: Run RAG evaluation
  env:
    RAG_USE_DUMMY_LLM: "true"
    RAG_RERANKER_ENABLED: "false"
    GATE_MRR: "0.65"
    GATE_HIT5: "0.75"
  run: |
    pip install -r requirements.txt
    python evaluation/backtest.py \
      --generate-synthetic \
      --single-config recursive_512_no_rerank \
      --output evaluation/results.json
  # exit code 1 if gates fail → blocks merge
```

## Production Metrics & Challenges Solved

### How to Measure

**Retrieval quality over time:** Run `backtest.py` on every PR that touches chunking
or retrieval code. Store `results.json` as a CI artifact. Compare MRR and NDCG@5
against the baseline committed in the repository. Alert (or block merge) if MRR
drops more than 0.05 from the baseline.

**Answer quality in production:** Sample 50 production queries weekly. Run them
through `backtest.py --llm-judge` with ground-truth answers written by domain
experts. Track the LLM-judge score trend over time. Target: average score >= 3.5/5.

**Latency under production load:** The `--load-test` flag measures sustained
throughput with configurable concurrency. Compare p95 latency under concurrency
against the gate threshold. In production, use Prometheus histograms for real-time
monitoring and set Grafana alerts.

### Challenges Solved

**1. Nondeterministic NDCG caused by tied RRF scores**

When two chunks from different documents have identical RRF scores (common with a
small corpus), their relative ordering is nondeterministic. This caused NDCG@5 to
vary by ±0.08 between runs on the same data. We fixed this by using the chunk's
vector similarity score as a tiebreaker in the RRF fusion, which is continuous-valued
and essentially never produces exact ties.

**2. ROUGE-L returning 0.0 for correct paraphrased answers**

The DummyLLMClient generates templated answers ("Based on the provided documents [1],
[2], here is the answer...") that share almost no lexical overlap with the expected
answers. This made ROUGE-L useless during development. We addressed this by:
(a) computing ROUGE-L only when `expected_answer` is non-empty, (b) reporting
token recall separately (which is more forgiving of extra text), and (c) documenting
that ROUGE-L is a regression detector, not an absolute quality measure — the
LLM-as-judge is the true quality signal.

**3. Load test oversubscribing the GIL and producing misleading throughput numbers**

Python's GIL means that ThreadPoolExecutor doesn't truly parallelise CPU-bound work
(embedding computation, BM25 scoring). Initial load test results showed QPS *decreasing*
with higher concurrency. We solved this by: (a) documenting that the load tester
measures contention-aware throughput (which is what production actually experiences),
(b) recommending `concurrency=1` for baseline measurement and `concurrency=4` for
production-realistic measurement, and (c) noting that true horizontal scaling requires
multiple pods (Kubernetes HPA), not multiple threads.
