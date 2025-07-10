#!/bin/bash
# filepath: /Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager/scripts/gradual-restructure.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[RESTRUCTURE]${NC} $1"
}

# Function to reorganize tests only
reorganize_tests() {
    print_header "Phase 1: Reorganizing tests directory"
    
    if [ ! -d "tests" ]; then
        print_warning "No tests directory found"
        return 0
    fi
    
    # Create backup
    cp -r tests tests_backup_$(date +%Y%m%d_%H%M%S)
    print_status "Created backup of tests directory"
    
    # Create new structure
    mkdir -p tests_new/{unit,integration,e2e,fixtures}
    
    # Categorize tests based on filename patterns
    for test_file in tests/*.py; do
        if [ -f "$test_file" ]; then
            filename=$(basename "$test_file")
            case "$filename" in
                test_complete_*|test_end_to_end_*|test_comprehensive_*)
                    mv "$test_file" tests_new/e2e/
                    print_status "Moved $filename to e2e/"
                    ;;
                test_*workflow*|test_*integration*|test_*docker*|test_*container*)
                    mv "$test_file" tests_new/integration/
                    print_status "Moved $filename to integration/"
                    ;;
                conftest.py|__init__.py)
                    cp "$test_file" tests_new/
                    cp "$test_file" tests_new/unit/
                    cp "$test_file" tests_new/integration/
                    cp "$test_file" tests_new/e2e/
                    print_status "Copied $filename to all test directories"
                    ;;
                *)
                    mv "$test_file" tests_new/unit/
                    print_status "Moved $filename to unit/"
                    ;;
            esac
        fi
    done
    
    # Replace old tests directory
    rm -rf tests
    mv tests_new tests
    
    print_status "Tests reorganization completed!"
    print_status "Structure:"
    find tests -type f -name "*.py" | head -10
}

# Function to create requirements structure
create_requirements_structure() {
    print_header "Phase 2: Creating requirements structure"
    
    if [ ! -f "requirements.txt" ]; then
        print_warning "No requirements.txt found"
        return 0
    fi
    
    mkdir -p requirements
    
    # Move base requirements
    cp requirements.txt requirements/base.txt
    
    # Create development requirements
    cat > requirements/dev.txt << 'EOF'
-r base.txt

# Development dependencies
pytest>=7.0.0
pytest-django>=4.5.0
pytest-cov>=4.0.0
black>=22.0.0
isort>=5.10.0
flake8>=5.0.0
mypy>=0.991
django-debug-toolbar>=3.2.0
ipython>=8.0.0
jupyter>=1.0.0
pre-commit>=2.20.0
EOF

    # Create production requirements
    cat > requirements/prod.txt << 'EOF'
-r base.txt

# Production dependencies
gunicorn>=20.1.0
psycopg2-binary>=2.9.0
whitenoise>=6.2.0
sentry-sdk>=1.9.0
redis>=4.3.0
celery>=5.2.0
uvicorn>=0.18.0
EOF

    # Create test requirements
    cat > requirements/test.txt << 'EOF'
-r base.txt
-r dev.txt

# Test dependencies
factory-boy>=3.2.0
faker>=15.0.0
responses>=0.21.0
freezegun>=1.2.0
pytest-mock>=3.8.0
pytest-xdist>=2.5.0
coverage>=6.4.0
EOF

    print_status "Requirements structure created in requirements/"
}

# Function to organize scripts
organize_scripts() {
    print_header "Phase 3: Organizing scripts"
    
    if [ ! -d "scripts" ]; then
        print_warning "No scripts directory found"
        return 0
    fi
    
    # Create subdirectories
    mkdir -p scripts/{development,deployment,maintenance}
    
    # Move existing scripts to development
    for script in scripts/*.sh; do
        if [ -f "$script" ]; then
            filename=$(basename "$script")
            case "$filename" in
                deploy*|backup*|prod*)
                    mv "$script" scripts/deployment/
                    print_status "Moved $filename to deployment/"
                    ;;
                clean*|migrate*|maintenance*)
                    mv "$script" scripts/maintenance/
                    print_status "Moved $filename to maintenance/"
                    ;;
                *)
                    mv "$script" scripts/development/
                    print_status "Moved $filename to development/"
                    ;;
            esac
        fi
    done
    
    # Move README if exists
    if [ -f "scripts/README.md" ]; then
        # Keep it in main scripts directory
        print_status "Kept README.md in scripts/"
    fi
}

# Function to create config directory
create_config_structure() {
    print_header "Phase 4: Creating config structure"
    
    mkdir -p config
    
    # Copy Docker files
    if [ -f "docker-compose.yml" ]; then
        cp docker-compose.yml config/docker-compose.dev.yml
        print_status "Created config/docker-compose.dev.yml"
    fi
    
    if [ -f "Dockerfile" ]; then
        cp Dockerfile config/Dockerfile.django
        print_status "Created config/Dockerfile.django"
    fi
    
    # Create production docker-compose
    if [ -f "docker-compose.yml" ]; then
        sed 's/build:/# build:/' docker-compose.yml > config/docker-compose.prod.yml
        print_status "Created config/docker-compose.prod.yml"
    fi
    
    # Create .env.example
    cat > .env.example << 'EOF'
# Django Settings
DEBUG=True
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1

# Database
DATABASE_URL=sqlite:///db.sqlite3

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_ARTIFACT_ROOT=./data/artifacts

# GitHub Container Registry
CR_PAT=your-github-token-here
EOF
    
    print_status "Created .env.example template"
}

# Function to create data directories
create_data_structure() {
    print_header "Phase 5: Creating data structure"
    
    mkdir -p data/{datasets,artifacts,models,temp}
    
    # Move existing model artifacts
    if [ -d "models" ]; then
        cp -r models/* data/models/ 2>/dev/null || true
        print_status "Copied models to data/models/"
    fi
    
    # Move MLflow runs
    if [ -d "mlruns" ]; then
        cp -r mlruns data/artifacts/ 2>/dev/null || true
        print_status "Copied mlruns to data/artifacts/"
    fi
    
    print_status "Data structure created"
}

# Function to update Makefile
update_makefile() {
    print_header "Phase 6: Updating Makefile"
    
    # Backup current Makefile
    cp Makefile Makefile.backup
    
    # Create new Makefile with updated paths
    cat > Makefile << 'EOF'
.PHONY: help setup dev prod test clean docs

# Load environment variables
ifneq (,$(wildcard .env))
    include .env
    export $(shell sed 's/=.*//' .env)
endif

help:
	@echo "Available commands:"
	@echo "  setup          - Initial project setup"
	@echo "  dev            - Start development environment"
	@echo "  prod           - Start production environment"  
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests"
	@echo "  test-integration - Run integration tests"
	@echo "  test-e2e       - Run end-to-end tests"
	@echo "  clean          - Clean up containers and volumes"
	@echo "  django-setup   - Setup Django (migrations + superuser)"
	@echo "  django-shell   - Django shell"

setup:
	@chmod +x scripts/development/*.sh
	@./scripts/development/setup.sh

dev:
	@docker-compose -f config/docker-compose.dev.yml up --build

prod:
	@docker-compose -f config/docker-compose.prod.yml up -d

test:
	@python -m pytest tests/ -v

test-unit:
	@python -m pytest tests/unit/ -v

test-integration:
	@python -m pytest tests/integration/ -v

test-e2e:
	@python -m pytest tests/e2e/ -v

clean:
	@docker-compose down --remove-orphans
	@docker system prune -f

django-setup:
	@./scripts/development/django-setup.sh setup

django-shell:
	@./scripts/development/django-setup.sh shell

# Legacy support for existing commands
start: dev
stop: clean
logs:
	@docker-compose logs -f
shell:
	@docker-compose exec django bash
EOF

    print_status "Makefile updated with new structure"
}

# Function to create documentation structure
create_docs_structure() {
    print_header "Phase 7: Organizing documentation"
    
    mkdir -p docs/{api,deployment,development,architecture,troubleshooting}
    
    # Move existing docs
    if [ -d "docs" ]; then
        # Move markdown files to appropriate sections
        for doc in *.md; do
            if [ -f "$doc" ] && [ "$doc" != "README.md" ]; then
                case "$doc" in
                    *MIGRATION*|*RESTRUCTURE*|*PROPOSAL*)
                        mv "$doc" docs/development/
                        print_status "Moved $doc to docs/development/"
                        ;;
                    *)
                        cp "$doc" docs/
                        print_status "Copied $doc to docs/"
                        ;;
                esac
            fi
        done
    fi
    
    # Create README for docs
    cat > docs/README.md << 'EOF'
# MLManager Documentation

## Structure

- `api/` - API documentation
- `deployment/` - Deployment guides  
- `development/` - Development setup and guides
- `architecture/` - Architecture diagrams and design docs
- `troubleshooting/` - Common issues and solutions

## Quick Links

- [Development Setup](development/)
- [API Reference](api/)
- [Deployment Guide](deployment/)
EOF

    print_status "Documentation structure created"
}

# Main function
main() {
    print_header "Starting gradual MLManager restructure..."
    
    echo ""
    print_warning "This will gradually reorganize your project structure."
    print_warning "Each phase creates backups and can be run independently."
    echo ""
    echo "Phases:"
    echo "1. Reorganize tests directory"
    echo "2. Create requirements structure"  
    echo "3. Organize scripts"
    echo "4. Create config structure"
    echo "5. Create data structure"
    echo "6. Update Makefile"
    echo "7. Organize documentation"
    echo ""
    
    echo -n "Continue with gradual restructure? (y/N): "
    read -r REPLY
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Restructure cancelled"
        exit 0
    fi
    
    # Create overall backup
    backup_dir="../mlmanager_gradual_backup_$(date +%Y%m%d_%H%M%S)"
    cp -r . "$backup_dir"
    echo "$backup_dir" > .backup_location
    print_status "Created full backup at: $backup_dir"
    
    # Execute phases
    reorganize_tests
    create_requirements_structure  
    organize_scripts
    create_config_structure
    create_data_structure
    update_makefile
    create_docs_structure
    
    print_header "Gradual restructure completed!"
    print_status "Backup location: $(cat .backup_location)"
    
    echo ""
    print_header "New structure overview:"
    echo "📁 tests/"
    echo "  ├── unit/          # Unit tests"
    echo "  ├── integration/   # Integration tests"
    echo "  └── e2e/           # End-to-end tests"
    echo ""
    echo "📁 requirements/"
    echo "  ├── base.txt       # Core dependencies"
    echo "  ├── dev.txt        # Development dependencies"
    echo "  ├── prod.txt       # Production dependencies"
    echo "  └── test.txt       # Test dependencies"
    echo ""
    echo "📁 config/"
    echo "  ├── docker-compose.dev.yml"
    echo "  └── docker-compose.prod.yml"
    echo ""
    echo "📁 scripts/"
    echo "  ├── development/   # Dev scripts"
    echo "  ├── deployment/    # Deploy scripts"
    echo "  └── maintenance/   # Maintenance scripts"
    echo ""
    
    print_header "Next steps:"
    echo "1. Test the new structure: make test"
    echo "2. Try development setup: make dev"
    echo "3. Review and adjust as needed"
    echo "4. Commit changes when satisfied"
}

# Handle command line arguments
case ${1:-main} in
    "tests")
        reorganize_tests
        ;;
    "requirements")
        create_requirements_structure
        ;;
    "scripts")
        organize_scripts
        ;;
    "config")
        create_config_structure
        ;;
    "data")
        create_data_structure
        ;;
    "makefile")
        update_makefile
        ;;
    "docs")
        create_docs_structure
        ;;
    "main"|*)
        main
        ;;
esac
