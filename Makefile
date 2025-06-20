.PHONY: setup start stop restart logs shell rebuild clean django-setup quick help enhanced-start enhanced-stop enhanced-logs enhanced-status

# Enhanced ML Manager commands
.DEFAULT_GOAL := help

ifneq (,$(wildcard .env))
    include .env
    export $(shell sed 's/=.*//' .env)
endif

# ==============================================
# Enhanced ML Manager Docker Commands
# ==============================================

enhanced-setup: ## Setup Enhanced ML Manager environment
	@echo "🚀 Setting up Enhanced ML Manager..."
	@cp .env.enhanced.example .env || true
	@echo "✅ Environment file created. Please edit .env if needed."
	@docker network create enhanced-ml-network 2>/dev/null || true
	@echo "✅ Docker network created."

enhanced-start: ## Start Enhanced ML Manager with all services
	@echo "🚀 Starting Enhanced ML Manager..."
	@docker compose -f docker-compose.enhanced.yml up -d
	@echo "⏳ Waiting for services to be ready..."
	@sleep 10
	@echo "🏥 Health check:"
	@make enhanced-status
	@echo ""
	@echo "🎉 Enhanced ML Manager is ready!"
	@echo "📊 Django App: http://localhost:8000"
	@echo "📈 MLflow: http://localhost:5000"

enhanced-stop: ## Stop Enhanced ML Manager services
	@echo "🛑 Stopping Enhanced ML Manager..."
	@docker compose -f docker-compose.enhanced.yml down
	@echo "✅ All services stopped."

enhanced-restart: ## Restart Enhanced ML Manager services
	@echo "🔄 Restarting Enhanced ML Manager..."
	@make enhanced-stop
	@sleep 5
	@make enhanced-start

enhanced-logs: ## Show logs from all Enhanced ML Manager services
	@docker compose -f docker-compose.enhanced.yml logs -f

enhanced-logs-django: ## Show Django logs
	@docker compose -f docker-compose.enhanced.yml logs -f django

enhanced-status: ## Check status of Enhanced ML Manager services
	@echo "📊 Service Status:"
	@docker compose -f docker-compose.enhanced.yml ps
	@echo ""
	@echo "🔍 Health Checks:"
	@echo -n "Django: "
	@curl -s http://localhost:8000 >/dev/null && echo "✅ OK" || echo "❌ Failed"
	@echo -n "MLflow: "
	@curl -s http://localhost:5000 >/dev/null && echo "✅ OK" || echo "❌ Failed"

enhanced-shell: ## Open shell in Django container
	@docker exec -it web bash

enhanced-rebuild: ## Rebuild Enhanced ML Manager containers
	@echo "🔨 Rebuilding Enhanced ML Manager containers..."
	@docker-compose -f docker-compose.enhanced.yml build --no-cache
	@echo "✅ Containers rebuilt."

enhanced-clean: ## Clean up Enhanced ML Manager containers and volumes
	@echo "🧹 Cleaning up Enhanced ML Manager..."
	@docker-compose -f docker-compose.enhanced.yml down -v --remove-orphans
	@docker system prune -f
	@echo "✅ Cleanup completed."

enhanced-migrate: ## Run Django migrations in container
	@echo "🔄 Running Django migrations..."
	@docker exec web python core/manage.py migrate
	@echo "✅ Migrations completed."

enhanced-collectstatic: ## Collect static files in container
	@echo "📦 Collecting static files..."
	@docker exec web python core/manage.py collectstatic --noinput
	@echo "✅ Static files collected."

enhanced-createuser: ## Create Django superuser in container
	@echo "👤 Creating Django superuser..."
	@docker exec -it web python core/manage.py createsuperuser

enhanced-test: ## Run tests in container
	@echo "🧪 Running tests..."
	@docker exec web python core/manage.py test
	@echo "✅ Tests completed."

enhanced-backup: ## Backup Enhanced ML Manager data
	@echo "💾 Creating backup..."
	@mkdir -p backups
	@docker exec web tar czf - /app/data | cat > backups/enhanced-ml-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz
	@echo "✅ Backup created in backups/ directory."

enhanced-monitor: ## Monitor Enhanced ML Manager services
	@echo "📊 Monitoring Enhanced ML Manager..."
	@echo "Press Ctrl+C to stop monitoring"
	@while true; do \
		clear; \
		echo "=== Enhanced ML Manager Status ==="; \
		echo ""; \
		make enhanced-status; \
		echo ""; \
		echo "=== Resource Usage ==="; \
		docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" web mlflow; \
		sleep 5; \
	done

# ==============================================
# Legacy Commands (kept for compatibility)
# ==============================================

train: ## Run ML training (legacy)
	mkdir -p runs
	@echo $(DATASET_PATH)
	@echo $(MLFLOW_BACKEND)
	python ml/training/train.py --path '$(DATASET_PATH)' --epochs "[10]" --shuffle True --lr 0.01 --batch_size [32] --bce_weight "[0.1]" --quarterres 

metrics: ## Show training metrics with TensorBoard
	tensorboard --logdir runs
	
predict: ## Run inference (legacy)
	python ml/inference/predict.py --image_path '$(IMAGE_PATH)' --epoch 6
	
setup: ## Legacy setup
	@chmod +x scripts/development/*.sh
	@./scripts/development/setup.sh

quick: ## Legacy quick setup
	@chmod +x scripts/development/*.sh
	@./scripts/development/quick-setup.sh

start: ## Legacy start
	@./scripts/development/dev.sh start

stop: ## Legacy stop
	@./scripts/development/dev.sh stop

restart: ## Legacy restart
	@./scripts/development/dev.sh restart

logs: ## Legacy logs
	@./scripts/development/dev.sh logs

shell: ## Legacy shell
	@./scripts/development/dev.sh shell

rebuild: ## Legacy rebuild
	@./scripts/development/dev.sh rebuild

clean: ## Legacy clean
	@./scripts/development/dev.sh clean

django-setup: ## Legacy Django setup
	@./scripts/development/django-setup.sh setup

django-migrate:
	@./scripts/development/django-setup.sh migrate

django-migrate-only:
	@./scripts/development/django-setup.sh migrate-only

django-makemigrations:
	@./scripts/development/django-setup.sh makemigrations

django-user:
	@./scripts/development/django-setup.sh user

django-shell:
	@./scripts/development/django-setup.sh shell

restructure:
	@chmod +x scripts/development/restructure-full.sh
	@./scripts/development/restructure-full.sh run

restructure-apply:
	@./scripts/development/restructure-full.sh apply

restructure-rollback:
	@./scripts/development/restructure-full.sh rollback

gradual-restructure:
	@chmod +x scripts/development/gradual-restructure.sh
	@./scripts/development/gradual-restructure.sh

restructure-tests:
	@./scripts/development/gradual-restructure.sh tests

restructure-requirements:
	@./scripts/development/gradual-restructure.sh requirements

# ==============================================
# Help
# ==============================================

help: ## Show this help message
	@echo "Enhanced ML Manager - Docker Commands"
	@echo "====================================="
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make enhanced-setup    # Setup environment"
	@echo "  make enhanced-start    # Start all services"
	@echo "  make enhanced-status   # Check service status"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "🌐 Service URLs:"
	@echo "  Django App: http://localhost:8000"
	@echo "  MLflow:     http://localhost:5000"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  docs/ENHANCED_ML_MANAGER_IMPLEMENTATION.md"