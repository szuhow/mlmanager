# ================================================
# Coronary Experiments ML Manager - Makefile
# ================================================

.PHONY: help setup build-cpu build-gpu start-cpu start-gpu stop-cpu stop-gpu restart-cpu restart-gpu logs-cpu logs-gpu status-cpu status-gpu clean-cpu clean-gpu shell-cpu shell-gpu migrate-cpu migrate-gpu

# Default target
.DEFAULT_GOAL := help

# Docker compose command (use new syntax)
DOCKER_COMPOSE = docker compose

# Docker compose files
COMPOSE_CPU = infrastructure/docker-compose/docker-compose.cpu.yml
COMPOSE_GPU = infrastructure/docker-compose/docker-compose.gpu.yml

# Environment files
ENV_CPU = infrastructure/env/.env.cpu
ENV_GPU = infrastructure/env/.env.gpu

# Dataset URLs (hardcoded examples - replace with your actual URLs)
ARCADE_DATASET_URL = https://drive.google.com/file/d/1ABC123DEF456GHI789JKL/view
CORONARY_DATASET_URL = https://drive.google.com/drive/folders/1uUSasFHxscLwRrBU2Wbnc3PPgjuSasPr?usp=drive_link
CADICA_DATASET_URL = https://drive.google.com/file/d/1CADICA123456789ABCDEF/view

# ================================================
# Help
# ================================================

help: ## Show this help message
	@echo "🏥 Coronary Experiments ML Manager"
	@echo "=================================="
	@echo ""
	@echo "🚀 Quick Start (CPU):"
	@echo "  make setup-cpu      # Setup CPU environment"
	@echo "  make build-cpu      # Build CPU containers"
	@echo "  make start-cpu      # Start CPU services"
	@echo "  make status-cpu     # Check CPU status"
	@echo ""
	@echo "🚀 Quick Start (GPU):"
	@echo "  make setup-gpu      # Setup GPU environment"
	@echo "  make build-gpu      # Build GPU containers"
	@echo "  make start-gpu      # Start GPU services"
	@echo "  make status-gpu     # Check GPU status"
	@echo ""
	@echo "📥 Dataset Management:"
	@echo "  make setup-datasets         # Setup datasets directory and dependencies"
	@echo "  make download-dataset        # Download from Google Drive URL"
	@echo "  make download-arcade         # Download ARCADE dataset (hardcoded URL)"
	@echo "  make download-coronary-example # Download coronary dataset (hardcoded URL)"
	@echo "  make download-cadica         # Download CADICA dataset (hardcoded URL)"
	@echo "  make download-all-datasets   # Download all predefined datasets"
	@echo "  make list-datasets           # List all downloaded datasets"
	@echo "  Example: make download-dataset URL=https://drive.google.com/file/d/1ABC.../view"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "🌐 Service URLs:"
	@echo "  Django App: http://localhost:8000"
	@echo "  Flower:     http://localhost:5555"
	@echo "  MLflow:     http://localhost:5000"
	@echo "  Database:   localhost:5432"
	@echo "  Redis:      localhost:6379"

# ================================================
# Setup Commands
# ================================================

setup-cpu: ## Setup CPU environment
	@echo "⚙️  Setting up CPU environment..."
	@mkdir -p core/data/logs core/data/media core/data/static core/data/models core/data/mlflow
	@cp $(ENV_CPU) .env
	@echo "✅ CPU environment setup complete"
	@echo "📝 You can edit .env file to customize settings"

setup-gpu: ## Setup GPU environment  
	@echo "⚙️  Setting up GPU environment..."
	@mkdir -p core/data/logs core/data/media core/data/static core/data/models core/data/mlflow
	@cp $(ENV_GPU) .env
	@echo "✅ GPU environment setup complete"
	@echo "📝 You can edit .env file to customize settings"
	@echo "⚠️  Make sure nvidia-docker is installed for GPU support"

# ================================================
# Build Commands
# ================================================

build-cpu: ## Build CPU containers
	@echo "🔨 Building CPU containers..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_CPU) build --no-cache
	@echo "✅ CPU containers built successfully"

build-gpu: ## Build GPU containers
	@echo "🔨 Building GPU containers..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) build --no-cache
	@echo "✅ GPU containers built successfully"

# ================================================
# Start/Stop Commands
# ================================================

start-cpu: ## Start CPU services
	@echo "🚀 Starting CPU services..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_CPU) up -d
	@echo "⏳ Waiting for services to start..."
	@sleep 35
	@echo "✅ CPU services started"
	@make status-cpu

start-gpu: ## Start GPU services
	@echo "🚀 Starting GPU services..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) up -d
	@echo "⏳ Waiting for services to start..."
	@sleep 35
	@echo "✅ GPU services started"
	@make status-gpu

stop-cpu: ## Stop CPU services
	@echo "🛑 Stopping CPU services..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_CPU) down
	@echo "✅ CPU services stopped"

stop-gpu: ## Stop GPU services
	@echo "🛑 Stopping GPU services..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) down
	@echo "✅ GPU services stopped"

restart-cpu: ## Restart CPU services
	@echo "🔄 Restarting CPU services..."
	@make stop-cpu
	@sleep 5
	@make start-cpu

restart-gpu: ## Restart GPU services
	@echo "🔄 Restarting GPU services..."
	@make stop-gpu
	@sleep 5
	@make start-gpu

# ================================================
# Status and Logs
# ================================================

status-cpu: ## Check CPU services status
	@echo "📊 CPU Services Status:"
	@$(DOCKER_COMPOSE) -f $(COMPOSE_CPU) ps
	@echo ""
	@echo "🔍 Health Checks:"
	@echo -n "  Django: "
	@curl -s http://localhost:8000/health/ >/dev/null 2>&1 && echo "✅ OK" || echo "❌ Failed"
	@echo -n "  Flower: "
	@curl -s http://localhost:5555 >/dev/null 2>&1 && echo "✅ OK" || echo "❌ Failed"
	@echo -n "  MLflow: "
	@curl -s http://localhost:5000 >/dev/null 2>&1 && echo "✅ OK" || echo "❌ Failed"
	@echo -n "  Redis:  "
	@$(DOCKER_COMPOSE) -f $(COMPOSE_CPU) exec redis redis-cli ping >/dev/null 2>&1 && echo "✅ OK" || echo "❌ Failed"
	@echo -n "  DB:     "
	@$(DOCKER_COMPOSE) -f $(COMPOSE_CPU) exec db pg_isready -U postgres >/dev/null 2>&1 && echo "✅ OK" || echo "❌ Failed"

status-gpu: ## Check GPU services status
	@echo "📊 GPU Services Status:"
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) ps
	@echo ""
	@echo "🔍 Health Checks:"
	@echo -n "  Django: "
	@curl -s http://localhost:8000/health/ >/dev/null 2>&1 && echo "✅ OK" || echo "❌ Failed"
	@echo -n "  Flower: "
	@curl -s http://localhost:5555 >/dev/null 2>&1 && echo "✅ OK" || echo "❌ Failed"
	@echo -n "  MLflow: "
	@curl -s http://localhost:5000 >/dev/null 2>&1 && echo "✅ OK" || echo "❌ Failed"
	@echo -n "  Redis:  "
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) exec redis redis-cli ping >/dev/null 2>&1 && echo "✅ OK" || echo "❌ Failed"
	@echo -n "  DB:     "
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) exec db pg_isready -U postgres >/dev/null 2>&1 && echo "✅ OK" || echo "❌ Failed"
	@echo -n "  GPU:    "
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) exec django nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null && echo "✅ OK" || echo "❌ Failed"

logs-cpu: ## Show CPU services logs
	@echo "📝 CPU Services Logs:"
	@$(DOCKER_COMPOSE) -f $(COMPOSE_CPU) logs -f

logs-gpu: ## Show GPU services logs
	@echo "📝 GPU Services Logs:"
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) logs -f

logs-django-cpu: ## Show CPU Django logs only
	@$(DOCKER_COMPOSE) -f $(COMPOSE_CPU) logs -f django

logs-django-gpu: ## Show GPU Django logs only
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) logs -f django

logs-training-cpu: ## Show CPU Training Worker logs
	@$(DOCKER_COMPOSE) -f $(COMPOSE_CPU) logs -f training-worker

logs-training-gpu: ## Show GPU Training Worker logs
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) logs -f training-worker

logs-inference-cpu: ## Show CPU Inference Worker logs
	@$(DOCKER_COMPOSE) -f $(COMPOSE_CPU) logs -f inference-worker

logs-inference-gpu: ## Show GPU Inference Worker logs
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) logs -f inference-worker

logs-default-cpu: ## Show CPU Default Worker logs
	@$(DOCKER_COMPOSE) -f $(COMPOSE_CPU) logs -f default-worker

logs-default-gpu: ## Show GPU Default Worker logs
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) logs -f default-worker

# ================================================
# Database Commands
# ================================================

migrate-cpu: ## Run migrations (CPU)
	@echo "🔄 Running migrations (CPU)..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_CPU) exec django python core/manage.py migrate
	@echo "✅ Migrations completed"

migrate-gpu: ## Run migrations (GPU)
	@echo "🔄 Running migrations (GPU)..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) exec django python core/manage.py migrate
	@echo "✅ Migrations completed"

makemigrations-cpu: ## Create migrations (CPU)
	@echo "🔄 Creating migrations (CPU)..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_CPU) exec django python core/manage.py makemigrations
	@echo "✅ Migrations created"

makemigrations-gpu: ## Create migrations (GPU)
	@echo "🔄 Creating migrations (GPU)..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) exec django python core/manage.py makemigrations
	@echo "✅ Migrations created"

superuser-cpu: ## Create superuser (CPU)
	@echo "👤 Creating superuser (CPU)..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_CPU) exec django python core/manage.py createsuperuser

superuser-gpu: ## Create superuser (GPU)
	@echo "👤 Creating superuser (GPU)..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) exec django python core/manage.py createsuperuser

# ================================================
# Shell Commands
# ================================================

shell-cpu: ## Open Django shell (CPU)
	@echo "🐚 Opening Django shell (CPU)..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_CPU) exec django bash

shell-gpu: ## Open Django shell (GPU)
	@echo "🐚 Opening Django shell (GPU)..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) exec django bash

django-shell-cpu: ## Open Django Python shell (CPU)
	@echo "🐍 Opening Django Python shell (CPU)..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_CPU) exec django python core/manage.py shell

django-shell-gpu: ## Open Django Python shell (GPU)
	@echo "🐍 Opening Django Python shell (GPU)..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) exec django python core/manage.py shell

# ================================================
# Maintenance Commands
# ================================================

clean-cpu: ## Clean CPU environment
	@echo "🧹 Cleaning CPU environment..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_CPU) down -v --remove-orphans
	@docker system prune -f
	@echo "✅ CPU environment cleaned"

clean-gpu: ## Clean GPU environment
	@echo "🧹 Cleaning GPU environment..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) down -v --remove-orphans
	@docker system prune -f
	@echo "✅ GPU environment cleaned"

backup-cpu: ## Backup CPU data
	@echo "💾 Creating CPU backup..."
	@mkdir -p backups
	@$(DOCKER_COMPOSE) -f $(COMPOSE_CPU) exec django tar czf - /app/core/data | cat > backups/cpu-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz
	@echo "✅ CPU backup created in backups/ directory"

backup-gpu: ## Backup GPU data
	@echo "💾 Creating GPU backup..."
	@mkdir -p backups
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) exec django tar czf - /app/core/data | cat > backups/gpu-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz
	@echo "✅ GPU backup created in backups/ directory"

# ================================================
# Dataset Management Commands
# ================================================

download-dataset: ## Download dataset from Google Drive URL
	@echo "📥 Dataset Download Tool"
	@echo "Usage: make download-dataset URL=<google_drive_url> [NAME=<dataset_name>]"
	@echo "Example: make download-dataset URL=https://drive.google.com/file/d/1ABC123.../view NAME=arcade_dataset"
	@if [ -z "$(URL)" ]; then \
		echo "❌ Error: URL parameter is required"; \
		echo "   Example: make download-dataset URL=https://drive.google.com/file/d/1ABC123.../view"; \
		exit 1; \
	fi
	@echo "🔍 Downloading dataset from: $(URL)"
	@if [ -n "$(NAME)" ]; then \
		echo "📝 Using custom name: $(NAME)"; \
		python3 infrastructure/scripts/download_dataset.py "$(URL)" --name "$(NAME)"; \
	else \
		python3 infrastructure/scripts/download_dataset.py "$(URL)"; \
	fi
	@echo "✅ Dataset download completed"

list-datasets: ## List all downloaded datasets
	@echo "📋 Available Datasets:"
	@python3 scripts/download_dataset.py --list

install-dataset-deps: ## Install dataset download dependencies
	@echo "📦 Installing dataset download dependencies..."
	@pip3 install requests tqdm
	@echo "✅ Dependencies installed (wget/curl are used for downloading)"

# Example dataset downloads
download-arcade: ## Download ARCADE dataset (hardcoded URL)
	@echo "📥 Downloading ARCADE Challenge Dataset..."
	@echo "🔗 Using hardcoded URL: $(ARCADE_DATASET_URL)"
	@if [ "$(ARCADE_DATASET_URL)" = "https://drive.google.com/file/d/1ABC123DEF456GHI789JKL/view" ]; then \
		echo "⚠️  This is a placeholder URL - please update ARCADE_DATASET_URL in Makefile"; \
		echo "   Visit: https://arcade.grand-challenge.org/ to get the real URL"; \
		exit 1; \
	fi
	@python3 scripts/download_dataset.py "$(ARCADE_DATASET_URL)" --name "arcade"

download-coronary-example: ## Download coronary dataset (hardcoded URL)
	@echo "📥 Downloading coronary dataset..."
	@echo "🔗 Using hardcoded URL: $(CORONARY_DATASET_URL)"
	@if [ "$(CORONARY_DATASET_URL)" = "https://drive.google.com/file/d/1XYZ789ABC123DEF456GHI/view" ]; then \
		echo "⚠️  This is a placeholder URL - please update CORONARY_DATASET_URL in Makefile"; \
		exit 1; \
	fi
	@python3 scripts/download_dataset.py "$(CORONARY_DATASET_URL)" --name "coronary_example"

download-cadica: ## Download CADICA dataset (hardcoded URL)
	@echo "📥 Downloading CADICA dataset..."
	@echo "🔗 Using hardcoded URL: $(CADICA_DATASET_URL)"
	@if [ "$(CADICA_DATASET_URL)" = "https://drive.google.com/file/d/1CADICA123456789ABCDEF/view" ]; then \
		echo "⚠️  This is a placeholder URL - please update CADICA_DATASET_URL in Makefile"; \
		exit 1; \
	fi
	@python3 scripts/download_dataset.py "$(CADICA_DATASET_URL)" --name "cadica"

download-all-datasets: ## Download all hardcoded datasets
	@echo "📥 Downloading all predefined datasets..."
	@echo "🚀 This will download: ARCADE, Coronary Example, and CADICA datasets"
	@make download-arcade
	@make download-coronary-example
	@make download-cadica
	@echo "✅ All datasets downloaded successfully"

setup-datasets: ## Setup datasets directory and install dependencies
	@echo "📁 Setting up datasets directory..."
	@mkdir -p core/data/datasets
	@make install-dataset-deps
	@echo "✅ Datasets setup completed"
	@echo "📖 Usage:"
	@echo "   make download-dataset URL=<google_drive_url> [NAME=<name>]"
	@echo "   make list-datasets"

cleanup-datasets: ## Clean up downloaded datasets
	@echo "🧹 Cleaning up datasets..."
	@echo "⚠️  This will remove ALL datasets in core/data/datasets/"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		echo "🗑️  Removing datasets..."; \
		rm -rf core/data/datasets/*; \
		echo "✅ Datasets cleaned up"; \
	else \
		echo ""; \
		echo "❌ Cleanup cancelled"; \
	fi

update-dataset-urls: ## Show how to update hardcoded dataset URLs
	@echo "🔧 Updating Dataset URLs"
	@echo "========================"
	@echo ""
	@echo "To update hardcoded dataset URLs, edit the following variables in Makefile:"
	@echo ""
	@echo "ARCADE_DATASET_URL = $(ARCADE_DATASET_URL)"
	@echo "CORONARY_DATASET_URL = $(CORONARY_DATASET_URL)"
	@echo "CADICA_DATASET_URL = $(CADICA_DATASET_URL)"
	@echo ""
	@echo "📝 Example URLs format:"
	@echo "  File:   https://drive.google.com/file/d/FILE_ID/view"
	@echo "  Folder: https://drive.google.com/drive/folders/FOLDER_ID"
	@echo "  Direct: https://drive.google.com/uc?id=FILE_ID"
	@echo ""
	@echo "💡 After updating URLs, you can use:"
	@echo "  make download-arcade"
	@echo "  make download-coronary-example"
	@echo "  make download-cadica"
	@echo "  make download-all-datasets"

# ================================================
# Development Commands
# ================================================

test-cpu: ## Run tests (CPU)
	@echo "🧪 Running tests (CPU)..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_CPU) exec django python core/manage.py test
	@echo "✅ Tests completed"

test-gpu: ## Run tests (GPU)
	@echo "🧪 Running tests (GPU)..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) exec django python core/manage.py test
	@echo "✅ Tests completed"

collectstatic-cpu: ## Collect static files (CPU)
	@echo "📦 Collecting static files (CPU)..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_CPU) exec django python core/manage.py collectstatic --noinput
	@echo "✅ Static files collected"

collectstatic-gpu: ## Collect static files (GPU)
	@echo "📦 Collecting static files (GPU)..."
	@$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) exec django python core/manage.py collectstatic --noinput
	@echo "✅ Static files collected"

# ================================================
# Monitoring Commands
# ================================================

monitor-cpu: ## Monitor CPU services
	@echo "📊 Monitoring CPU services (Press Ctrl+C to stop)..."
	@while true; do \
		clear; \
		echo "=== CPU Services Status ==="; \
		make status-cpu; \
		echo ""; \
		echo "=== Resource Usage ==="; \
		docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" $$($(DOCKER_COMPOSE) -f $(COMPOSE_CPU) ps -q) 2>/dev/null || echo "No containers running"; \
		sleep 5; \
	done

monitor-gpu: ## Monitor GPU services
	@echo "📊 Monitoring GPU services (Press Ctrl+C to stop)..."
	@while true; do \
		clear; \
		echo "=== GPU Services Status ==="; \
		make status-gpu; \
		echo ""; \
		echo "=== Resource Usage ==="; \
		docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" $$($(DOCKER_COMPOSE) -f $(COMPOSE_GPU) ps -q) 2>/dev/null || echo "No containers running"; \
		echo ""; \
		echo "=== GPU Usage ==="; \
		$(DOCKER_COMPOSE) -f $(COMPOSE_GPU) exec django nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "GPU info not available"; \
		sleep 5; \
	done

# ================================================
# Quick Start Shortcuts
# ================================================

quick-cpu: ## Quick start CPU (setup + build + start)
	@echo "🚀 Quick start CPU version..."
	@make setup-cpu
	@make build-cpu
	@make start-cpu
	@echo "🎉 CPU version is ready!"

quick-gpu: ## Quick start GPU (setup + build + start)
	@echo "🚀 Quick start GPU version..."
	@make setup-gpu
	@make build-gpu
	@make start-gpu
	@echo "🎉 GPU version is ready!"

# ================================================
# Default shortcuts (use CPU as default)
# ================================================

setup: setup-cpu ## Setup (default: CPU)
build: build-cpu ## Build (default: CPU)
start: start-cpu ## Start (default: CPU)
stop: stop-cpu ## Stop (default: CPU)
restart: restart-cpu ## Restart (default: CPU)
status: status-cpu ## Status (default: CPU)
logs: logs-cpu ## Logs (default: CPU)
shell: shell-cpu ## Shell (default: CPU)
migrate: migrate-cpu ## Migrate (default: CPU)
clean: clean-cpu ## Clean (default: CPU)
quick: quick-cpu ## Quick start (default: CPU)