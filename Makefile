.PHONY: help install dev test lint run docker-build docker-up backtest k8s-apply helm-install clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -r requirements.txt

dev: ## Install all dependencies (prod + dev + eval)
	pip install -r requirements.txt
	pip install pytest pytest-asyncio ruff mypy pandas tabulate

test: ## Run unit tests
	python -m pytest tests/ -v --tb=short

lint: ## Run linter
	ruff check src/ tests/ evaluation/

run: ## Run API locally with dummy LLM
	RAG_USE_DUMMY_LLM=true RAG_LOG_LEVEL=debug \
		python -m uvicorn src.api.app:app --reload --port 8000

docker-build: ## Build Docker image
	docker build -t rag-enterprise/api:latest .

docker-up: ## Start with docker compose (CPU only)
	docker compose up --build api

backtest: ## Run evaluation backtest
	./scripts/backtest.sh

k8s-apply: ## Apply Kubernetes manifests (base)
	kubectl apply -k deploy/k8s/base/

k8s-prod: ## Apply production overlay
	kubectl apply -k deploy/k8s/overlays/production/

helm-install: ## Install Helm chart
	helm install rag deploy/helm/rag-assistant/

helm-upgrade: ## Upgrade Helm release
	helm upgrade rag deploy/helm/rag-assistant/

helm-template: ## Render Helm templates (dry run)
	helm template rag deploy/helm/rag-assistant/

clean: ## Clean generated files
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf evaluation/results.json evaluation/test_data.jsonl
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
