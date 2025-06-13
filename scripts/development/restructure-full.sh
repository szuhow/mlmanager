#!/bin/bash
# filepath: /Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager/scripts/restructure.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if we're in the right directory
check_environment() {
    if [ ! -f "manage.py" ] && [ ! -f "core/manage.py" ] && [ ! -d "ml_manager" ] && [ ! -d "core/ml_manager" ]; then
        print_error "Not in MLManager project root directory"
        print_error "Expected to find manage.py or ml_manager directory"
        exit 1
    fi
    print_status "Environment check passed"
}

# Create backup
create_backup() {
    local backup_dir="../mlmanager_backup_$(date +%Y%m%d_%H%M%S)"
    print_status "Creating backup at: $backup_dir"
    cp -r . "$backup_dir"
    echo "$backup_dir" > .backup_location
    print_status "Backup created successfully"
}

# Create new directory structure
create_structure() {
    print_header "Creating new directory structure..."
    
    # Main directories
    mkdir -p config
    mkdir -p core/{requirements,static}
    mkdir -p ml/{training/{pipelines,configs,scripts},inference}
    mkdir -p data/{datasets,artifacts,models,temp}
    mkdir -p tests_new/{unit,integration,e2e,fixtures}
    mkdir -p docs_new/{api,deployment,development,architecture,troubleshooting}
    mkdir -p scripts_new/{deployment,development,maintenance}
    mkdir -p infrastructure/{kubernetes,terraform,monitoring,ci-cd}
    
    print_status "Directory structure created"
}

# Phase 1: Move configuration files
phase1_config() {
    print_header "Phase 1: Moving configuration files..."
    
    # Docker configs
    if [ -f "docker-compose.yml" ]; then
        cp docker-compose.yml config/docker-compose.prod.yml
        print_status "Copied docker-compose.yml -> config/docker-compose.prod.yml"
    fi
    
    if [ -f "Dockerfile" ]; then
        cp Dockerfile config/Dockerfile.django
        print_status "Copied Dockerfile -> config/Dockerfile.django"
    fi
    
    print_status "Phase 1 completed"
}

# Phase 2: Move Django core
phase2_django() {
    print_header "Phase 2: Moving Django application..."
    
    # Copy Django project
    if [ -d "coronary_experiments" ]; then
        cp -r coronary_experiments core/
        print_status "Copied coronary_experiments -> core/"
    fi
    
    if [ -d "ml_manager" ]; then
        cp -r ml_manager core/
        print_status "Copied ml_manager -> core/"
    fi
    
    if [ -f "manage.py" ]; then
        cp manage.py core/
        print_status "Copied manage.py -> core/"
    fi
    
    # Handle requirements
    if [ -f "requirements.txt" ]; then
        cp requirements.txt core/requirements/base.txt
        print_status "Copied requirements.txt -> core/requirements/base.txt"
    fi
    
    # Copy static files
    if [ -d "static" ]; then
        cp -r static core/
        print_status "Copied static -> core/"
    fi
    
    if [ -d "staticfiles" ]; then
        cp -r staticfiles core/
        print_status "Copied staticfiles -> core/"
    fi
    
    if [ -d "media" ]; then
        cp -r media core/
        print_status "Copied media -> core/"
    fi
    
    print_status "Phase 2 completed"
}

# Phase 3: Move ML components
phase3_ml() {
    print_header "Phase 3: Moving ML components..."
    
    # Copy shared ML code
    if [ -d "shared" ]; then
        cp -r shared ml/
        print_status "Copied shared -> ml/"
    fi
    
    # Copy models to data
    if [ -d "models" ]; then
        cp -r models data/
        print_status "Copied models -> data/"
    fi
    
    # Copy MLflow artifacts
    if [ -d "mlruns" ]; then
        cp -r mlruns data/artifacts/
        print_status "Copied mlruns -> data/artifacts/"
    fi
    
    print_status "Phase 3 completed"
}

# Phase 4: Reorganize tests
phase4_tests() {
    print_header "Phase 4: Reorganizing tests..."
    
    if [ -d "tests" ]; then
        # Categorize tests (basic categorization)
        for test_file in tests/*.py; do
            if [ -f "$test_file" ]; then
                filename=$(basename "$test_file")
                case "$filename" in
                    test_complete_*|test_end_to_end_*|test_comprehensive_*)
                        cp "$test_file" tests_new/e2e/
                        ;;
                    test_*workflow*|test_*integration*)
                        cp "$test_file" tests_new/integration/
                        ;;
                    test_*model*|test_*view*|test_*util*)
                        cp "$test_file" tests_new/unit/
                        ;;
                    *)
                        cp "$test_file" tests_new/unit/  # Default to unit tests
                        ;;
                esac
            fi
        done
        
        print_status "Tests reorganized and categorized"
    fi
    
    print_status "Phase 4 completed"
}

# Phase 5: Organize documentation
phase5_docs() {
    print_header "Phase 5: Organizing documentation..."
    
    # Copy existing docs
    if [ -d "docs" ]; then
        cp -r docs/* docs_new/troubleshooting/ 2>/dev/null || true
    fi
    
    # Copy documentation files from root
    for doc in *.md; do
        if [ "$doc" != "README.md" ] && [ -f "$doc" ]; then
            case "$doc" in
                *MIGRATION*|*RESTRUCTURE*|*PROPOSAL*)
                    cp "$doc" docs_new/development/
                    ;;
                *)
                    cp "$doc" docs_new/
                    ;;
            esac
        fi
    done
    
    print_status "Phase 5 completed"
}

# Phase 6: Organize scripts
phase6_scripts() {
    print_header "Phase 6: Organizing scripts..."
    
    if [ -d "scripts" ]; then
        # Copy development scripts to subdirectory
        cp -r scripts/* scripts_new/development/ 2>/dev/null || true
        
        # Copy this restructure script to maintenance
        cp scripts/restructure.sh scripts_new/maintenance/ 2>/dev/null || true
    fi
    
    print_status "Phase 6 completed"
}

# Update configuration files
update_configs() {
    print_header "Updating configuration files..."
    
    # Create development requirements
    cat > core/requirements/dev.txt << EOF
-r base.txt

# Development dependencies
pytest
pytest-django
pytest-cov
black
isort
flake8
mypy
django-debug-toolbar
ipython
jupyter
EOF

    # Create production requirements
    cat > core/requirements/prod.txt << EOF
-r base.txt

# Production dependencies
gunicorn
psycopg2-binary
whitenoise
sentry-sdk
redis
celery
EOF

    # Create test requirements
    cat > core/requirements/test.txt << EOF
-r base.txt
-r dev.txt

# Test dependencies
factory-boy
faker
responses
freezegun
EOF

    # Create .env.example
    cat > .env.example << EOF
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

    print_status "Configuration files updated"
}

# Preview changes
preview_changes() {
    print_header "Preview of new structure:"
    echo ""
    echo "📁 New directory structure:"
    find . -type d -name "*_new" -o -name "config" -o -name "core" -o -name "ml" -o -name "data" -o -name "infrastructure" | sort
    echo ""
    echo "📄 New files created:"
    find core/requirements/ -name "*.txt" 2>/dev/null || echo "  (requirements files will be created)"
    echo ""
}

# Main execution
main() {
    print_header "Starting MLManager project restructure..."
    
    # Safety checks
    check_environment
    
    print_warning "This will create a new structure alongside your current project."
    print_warning "Original files will be preserved until you confirm the changes."
    
    echo ""
    echo -n "Do you want to continue? (y/N): "
    read -r REPLY
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Restructure cancelled by user"
        exit 0
    fi
    
    # Execute phases
    create_backup
    create_structure
    phase1_config
    phase2_django
    phase3_ml
    phase4_tests
    phase5_docs
    phase6_scripts
    update_configs
    
    preview_changes
    
    print_header "Restructure preview completed!"
    print_status "Backup location: $(cat .backup_location)"
    print_status "New structure created successfully"
    print_warning "Review the new structure before applying changes"
    
    # Show what to do next
    echo ""
    print_header "Next steps:"
    echo "1. Review the new structure in *_new directories"
    echo "2. Test the new structure"
    echo "3. Run: $0 apply  (to apply changes)"
    echo "4. Run: $0 rollback  (to rollback)"
}

# Apply changes
apply_changes() {
    print_header "Applying restructure changes..."
    
    print_warning "This will replace your current structure!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Apply cancelled"
        exit 0
    fi
    
    # Replace old structure with new
    [ -d "tests_new" ] && rm -rf tests && mv tests_new tests
    [ -d "docs_new" ] && rm -rf docs && mv docs_new docs  
    [ -d "scripts_new" ] && rm -rf scripts && mv scripts_new scripts
    
    print_status "Changes applied successfully!"
}

# Rollback changes
rollback_changes() {
    if [ -f ".backup_location" ]; then
        local backup_dir=$(cat .backup_location)
        print_header "Rolling back from: $backup_dir"
        
        print_warning "This will restore your original structure!"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Remove new structure
            rm -rf config core ml data tests_new docs_new scripts_new infrastructure
            
            # Restore from backup (keeping current location)
            cp -r "$backup_dir"/* .
            
            print_status "Rollback completed!"
        fi
    else
        print_error "No backup location found"
    fi
}

# Handle script arguments
case ${1:-help} in
    "run")
        main
        ;;
    "apply")
        apply_changes
        ;;
    "rollback")
        rollback_changes
        ;;
    "help"|*)
        echo "MLManager Project Restructure Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  run      - Preview the restructure (safe)"
        echo "  apply    - Apply the restructure changes"
        echo "  rollback - Rollback to original structure"
        echo "  help     - Show this help"
        echo ""
        echo "Workflow:"
        echo "  1. $0 run      # Preview changes"
        echo "  2. $0 apply    # Apply if satisfied"
        echo "  3. $0 rollback # Undo if needed"
        ;;
esac
