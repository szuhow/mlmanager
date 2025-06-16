.PHONY: setup start stop restart logs shell rebuild clean django-setup quick help


ifneq (,$(wildcard .env))
    include .env
    export $(shell sed 's/=.*//' .env)
endif

train:
	mkdir -p runs
	@echo $(DATASET_PATH)
	@echo $(MLFLOW_BACKEND)
	python ml/training/train.py --path '$(DATASET_PATH)' --epochs "[10]" --shuffle True --lr 0.01 --batch_size [32] --bce_weight "[0.1]" --quarterres 
metrics:
	tensorboard --logdir runs
	
predict:
	python ml/inference/predict.py --image_path '$(IMAGE_PATH)' --epoch 6
	
setup:
	@chmod +x scripts/development/*.sh
	@./scripts/development/setup.sh

quick:
	@chmod +x scripts/development/*.sh
	@./scripts/development/quick-setup.sh

start:
	@./scripts/development/dev.sh start

stop:
	@./scripts/development/dev.sh stop

restart:
	@./scripts/development/dev.sh restart

logs:
	@./scripts/development/dev.sh logs

shell:
	@./scripts/development/dev.sh shell

rebuild:
	@./scripts/development/dev.sh rebuild

clean:
	@./scripts/development/dev.sh clean

django-setup:
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

help:
	@echo "Available commands:"
	@echo "  quick          - Complete quick setup"
	@echo "  setup          - Initial infrastructure setup"
	@echo "  start/stop     - Start/stop services"
	@echo "  logs           - View logs"
	@echo "  shell          - Django container shell"
	@echo "  django-setup       - Setup Django (migrations + superuser)"
	@echo "  django-migrate     - Create and apply migrations + static files"
	@echo "  django-migrate-only - Apply existing migrations only"
	@echo "  django-makemigrations - Create new migrations only"
	@echo "  django-user        - Create superuser only"
	@echo "  django-shell       - Django shell"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests"
	@echo "  test-integration - Run integration tests"
	@echo "  test-e2e       - Run end-to-end tests"
	@echo "  restructure    - Reorganize project structure (preview)"
	@echo "  gradual-restructure - Gradual, safe restructuring"
	@echo "  restructure-tests - Reorganize tests only"

# Test commands
test:
	@python -m pytest tests/ -v

test-unit:
	@python -m pytest tests/unit/ -v

test-integration:
	@python -m pytest tests/integration/ -v

test-e2e:
	@python -m pytest tests/e2e/ -v

test-fixtures:
	@python -m pytest tests/fixtures/ -v

# Environment-specific commands
dev:
	docker-compose -f config/docker/docker-compose.dev.yml up

dev-build:
	docker-compose -f config/docker/docker-compose.dev.yml up --build

prod:
	docker-compose -f config/docker/docker-compose.prod.yml up

prod-build:
	docker-compose -f config/docker/docker-compose.prod.yml up --build

# Data management
clean-data:
	@echo "Cleaning temporary data..."
	rm -rf data/temp/*
	@echo "Temporary data cleaned."

backup-data:
	@echo "Creating data backup..."
	tar -czf data-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz data/
	@echo "Data backup created."

# Infrastructure management
monitoring:
	docker-compose -f infrastructure/monitoring/docker-compose.monitoring.yml up

monitoring-stop:
	docker-compose -f infrastructure/monitoring/docker-compose.monitoring.yml down

backup:
	@./infrastructure/backup/backup.sh

backup-restore:
	@echo "Available backups:"
	@ls -la data/backups/
	@echo "Use: docker exec web python core/manage.py loaddata <backup_file>"

# Network management
network-create:
	docker-compose -f infrastructure/networking/networks.yml up --no-start

network-remove:
	docker-compose -f infrastructure/networking/networks.yml down

# Security scanning
security-scan:
	@echo "Running security scan..."
	docker run --rm -v $(PWD):/app securecodewarrior/docker-security-scan

download-arcade-dataset:
	@echo "Downloading arcade dataset..."
	mkdir -p data/arcade
	curl -L https://zenodo.org/records/8386059/files/arcade_challenge_datasets.zip -o data/datasets/arcade-challange-datasets.zip
	unzip data/datasets/arcade-challange-datasets.zip -d data/datasets/
	rm data/datasets/arcade-challange-datasets.zip
	@echo "Arcade dataset downloaded and extracted."