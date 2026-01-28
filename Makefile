# RAG Orchestrator Makefile

.PHONY: help build up down logs clean test lint deploy backup restore

help: ## Show this help message
	@echo 'RAG Orchestrator Management Commands:'
	@echo ''
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build Docker images
	docker-compose build --no-cache

up: ## Start all services
	docker-compose up -d

down: ## Stop all services
	docker-compose down

restart: ## Restart all services
	docker-compose restart

logs: ## View logs from all services
	docker-compose logs -f

logs-api: ## View API logs
	docker-compose logs -f api

logs-qdrant: ## View Qdrant logs
	docker-compose logs -f qdrant

logs-redis: ## View Redis logs
	docker-compose logs -f redis

clean: ## Remove all containers, volumes, and images
	docker-compose down -v --rmi all
	docker system prune -f

test: ## Run tests
	docker-compose exec api pytest tests/ -v

lint: ## Run code linting
	docker-compose exec api black src/
	docker-compose exec api isort src/
	docker-compose exec api ruff check src/

deploy: ## Deploy to production
	./scripts/deploy.sh production

deploy-dev: ## Deploy to development
	./scripts/deploy.sh development

backup: ## Create backup
	./scripts/backup.sh

restore: ## Restore from backup (specify BACKUP_FILE)
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "Usage: make restore BACKUP_FILE=path/to/backup.tar.gz"; \
		exit 1; \
	fi
	./scripts/restore.sh $(BACKUP_FILE)

health: ## Check service health
	@echo "API:"
	@curl -s http://localhost:8000/api/v1/monitoring/health | jq .
	@echo "\nQdrant:"
	@curl -s http://localhost:6333/health | jq .
	@echo "\nRedis:"
	@docker exec rag-redis redis-cli ping

ingest-sample: ## Ingest sample documents
	docker-compose exec api python scripts/ingest_sample_data.py

metrics: ## Show system metrics
	curl -s http://localhost:8000/api/v1/monitoring/metrics | jq .

config: ## Show current configuration
	curl -s http://localhost:8000/api/v1/monitoring/config | jq .

db-shell: ## Open shell in Qdrant container
	docker exec -it rag-qdrant /bin/bash

redis-shell: ## Open Redis CLI
	docker exec -it rag-redis redis-cli

api-shell: ## Open shell in API container
	docker exec -it rag-api /bin/bash

migrate: ## Run database migrations (if needed)
	@echo "No migrations needed for Qdrant"

seed: ## Seed with sample data
	docker-compose exec api python scripts/seed_database.py

monitor: ## Open monitoring dashboard
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	@echo "Redis Commander: http://localhost:8081"

# Development shortcuts
dev: ## Start development environment
	docker-compose -f docker-compose.dev.yml up -d

dev-down: ## Stop development environment
	docker-compose -f docker-compose.dev.yml down

dev-logs: ## View development logs
	docker-compose -f docker-compose.dev.yml logs -f