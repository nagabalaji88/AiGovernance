# Developer entrypoints.
#
# `make dev` and `make check` are the two that matter: one runs the platform
# with no external dependencies, the other runs exactly what CI runs. Any gap
# between local and CI checks is a gap developers discover the slow way.

.DEFAULT_GOAL := help
.PHONY: help install dev demo api web test test-watch lint fmt typecheck check \
        build up down logs clean seed openapi

BACKEND := backend
FRONTEND := frontend

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install backend and frontend dependencies
	cd $(BACKEND) && python -m pip install -e ".[dev]"
	cd $(FRONTEND) && npm ci --no-audit --no-fund

dev: ## Run the API with synthetic data and no external services
	cd $(BACKEND) && SEED_DEMO_DATA=true LOG_FORMAT=console \
		uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

demo: ## Run the full stack (API + dashboards) with synthetic data
	@echo "API      http://127.0.0.1:8000/docs"
	@echo "Frontend http://127.0.0.1:5173"
	@$(MAKE) -j2 api web

api: ## Run the API alone
	cd $(BACKEND) && SEED_DEMO_DATA=true LOG_FORMAT=console \
		uvicorn app.main:app --host 127.0.0.1 --port 8000

web: ## Run the frontend dev server
	cd $(FRONTEND) && npm run dev

test: ## Run the backend test suite with coverage
	cd $(BACKEND) && pytest --cov=app --cov-report=term-missing -q

test-watch: ## Re-run tests on change
	cd $(BACKEND) && pytest -q -x --ff

lint: ## Lint backend and frontend
	cd $(BACKEND) && ruff check app tests
	cd $(FRONTEND) && npm run typecheck

fmt: ## Auto-format and auto-fix
	cd $(BACKEND) && ruff check app tests --fix && ruff format app tests

typecheck: ## Static type checks
	cd $(BACKEND) && mypy app
	cd $(FRONTEND) && npm run typecheck

check: lint test ## Everything CI runs, locally
	cd $(FRONTEND) && npm run build

openapi: ## Emit the OpenAPI schema
	cd $(BACKEND) && python -c "import json; from app.main import app; \
		print(json.dumps(app.openapi(), indent=2))" > ../openapi.json
	@echo "wrote openapi.json"

build: ## Build container images
	docker compose build

up: ## Start the full stack in containers
	docker compose up -d
	@echo "Frontend   http://localhost:5173"
	@echo "API        http://localhost:8000/docs"
	@echo "Grafana    http://localhost:3000"
	@echo "Prometheus http://localhost:9090"

down: ## Stop the stack
	docker compose down

logs: ## Tail stack logs
	docker compose logs -f --tail=100

clean: ## Remove build artefacts and caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(BACKEND)/.pytest_cache $(BACKEND)/.ruff_cache $(BACKEND)/.mypy_cache \
	       $(BACKEND)/htmlcov $(BACKEND)/coverage.xml $(BACKEND)/.coverage \
	       $(FRONTEND)/dist $(FRONTEND)/.vite
