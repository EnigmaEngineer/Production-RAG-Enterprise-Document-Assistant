# Kubernetes Deployment (`deploy/`)

Kubernetes manifests (kustomize) and Helm chart for deploying the RAG system.

## Deployment Options

### Option A: Kustomize

```bash
# Development (base)
kubectl apply -k deploy/k8s/base/

# Production (overlay with higher resources and replicas)
kubectl apply -k deploy/k8s/overlays/production/
```

### Option B: Helm

```bash
# Install with defaults
helm install rag deploy/helm/rag-assistant/

# Production with custom values
helm install rag deploy/helm/rag-assistant/ \
  --set api.replicaCount=3 \
  --set secrets.apiKey="prod-key-here" \
  --set vllm.enabled=true \
  --set qdrant.enabled=true \
  --set ingress.host=rag.company.com \
  --set ingress.tls.enabled=true \
  -f values-prod.yaml
```

### Option C: Docker Compose (local dev)

```bash
docker compose up api                    # API only
docker compose --profile gpu up          # API + vLLM
docker compose --profile qdrant up       # API + Qdrant
```

## Component Architecture on K8s

```
Namespace: rag-system
├── Deployment: rag-api (2+ replicas, HPA)
│   └── Container: rag-api (port 8000)
├── Deployment: vllm (1 replica, GPU node)
│   └── Container: vllm-openai (port 8000)
├── Deployment: qdrant (optional, 1 replica)
│   └── Container: qdrant (port 6333)
├── Service: rag-api-service (ClusterIP → 80)
├── Service: vllm-service (ClusterIP → 8000)
├── HPA: rag-api-hpa (2–10 pods, CPU/memory)
├── Ingress: rag-api-ingress (nginx)
├── ConfigMap: rag-config
└── Secret: rag-secrets
```

## Production Metrics & Challenges Solved

### How to Measure

**Latency (p50/p95/p99):**
All API pods expose `/metrics` for Prometheus scraping. The Helm chart includes
Prometheus annotations on pod templates. Use the following PromQL queries:
- End-to-end p99: `histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{handler="/v1/query"}[5m])) by (le))`
- Per-component: Custom histograms in the application code for retrieval, rerank, and
  generation stages.

**Throughput:**
`sum(rate(http_requests_total{handler="/v1/query"}[1m]))` gives cluster-wide QPS.
HPA targets 70% CPU utilization, which correlates with ~10 req/s per pod on a
2-vCPU allocation.

**Cost reduction:**
vLLM with AWQ 4-bit quantization runs Llama 3.1 8B on an L4 GPU ($0.70/hr on GKE)
instead of requiring an A100 ($2.48/hr). Continuous batching further improves
utilization: at batch_size=32, throughput is 2.4× higher than sequential serving,
yielding effective cost of ~$0.29/hr per equivalent sequential request capacity.

**Uptime:**
Pod Disruption Budgets (add `minAvailable: 1` for rag-api) ensure at least one pod
is always running during voluntary disruptions. Readiness probes prevent routing to
uninitialized pods. Liveness probes restart stuck pods within 45 seconds.

### Challenges Solved

**1. GPU cold start on vLLM pod reschedule (2–3 minute startup)**
vLLM needs to download and load the model weights into GPU memory on every pod start.
For Llama 3.1 8B, this takes 2–3 minutes. Mitigations: (a) `startupProbe` with high
`failureThreshold` so Kubernetes doesn't kill the pod during loading, (b) a
PersistentVolumeClaim for the HuggingFace cache mounted at `/root/.cache/huggingface`,
reducing subsequent starts to ~30 seconds, (c) setting `replicas: 1` with no HPA on
the vLLM deployment to avoid unnecessary reschedules.

**2. Resource contention between API and vLLM on shared GPU nodes**
Initially we ran the API (with FAISS) and vLLM on the same GPU node. The reranker's
GPU memory usage competed with vLLM's PagedAttention KV cache, causing OOM kills.
Solution: run the reranker in CPU-only mode with ONNX Runtime (4 vCPU, ~200ms per
batch of 25 pairs), and reserve the GPU exclusively for vLLM with
`gpu-memory-utilization: 0.85`.

**3. Rolling update causing brief 503 errors**
During a rolling update, old pods are terminated before new pods are ready (because
model loading takes 30+ seconds). We solved this with: (a) `maxSurge: 1` and
`maxUnavailable: 0` in the deployment strategy, (b) `terminationGracePeriodSeconds: 45`
with a `preStop` hook that sleeps 5 seconds to allow in-flight request completion,
and (c) readiness probe configuration ensuring new pods only receive traffic after
the full pipeline is initialized.
