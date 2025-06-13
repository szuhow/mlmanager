#!/bin/bash
# filepath: /Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager/django-setup.sh

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
    echo -e "${BLUE}[DJANGO]${NC} $1"
}

# Check if containers are running
check_containers() {
    if ! docker-compose ps | grep -q "Up"; then
        print_error "Containers are not running. Start them first with: ./dev.sh start"
        exit 1
    fi
}

# Run Django management commands
run_django_command() {
    print_status "Running: python core/manage.py $1"
    docker-compose exec django python core/manage.py $@
}

# Main setup function
setup_django() {
    print_header "Setting up Django application..."
    
    check_containers
    
    # Make migrations
    print_status "Creating migrations..."
    run_django_command makemigrations
    
    # Apply migrations
    print_status "Applying migrations..."
    run_django_command migrate
    
    # Collect static files
    print_status "Collecting static files..."
    run_django_command collectstatic --noinput
    
    # Create superuser
    print_header "Creating Django superuser..."
    
    read -p "Enter superuser username: " DJANGO_USER
    read -p "Enter superuser email: " DJANGO_EMAIL
    
    if [ -z "$DJANGO_USER" ] || [ -z "$DJANGO_EMAIL" ]; then
        print_error "Username and email are required"
        exit 1
    fi
    
    # Create superuser using Django shell
    docker-compose exec django python core/manage.py shell << EOF
from django.contrib.auth import get_user_model
User = get_user_model()

if not User.objects.filter(username='$DJANGO_USER').exists():
    user = User.objects.create_superuser('$DJANGO_USER', '$DJANGO_EMAIL', 'admin123')
    print(f"Superuser '{user.username}' created successfully!")
    print("Default password: admin123")
    print("Please change it after first login!")
else:
    print(f"User '$DJANGO_USER' already exists")
EOF
    
    print_status "Django setup completed!"
    print_status "Access Django admin at: http://localhost:8000/admin/"
    print_status "Username: $DJANGO_USER"
    print_status "Password: admin123 (change after first login)"
}

# Database operations
reset_database() {
    print_warning "This will delete all data in the database!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Resetting database..."
        
        # Remove migration files (keep __init__.py)
        run_django_command shell << 'EOF'
import os
import glob

# Find all migration files except __init__.py
for root, dirs, files in os.walk('.'):
    if 'migrations' in root:
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                file_path = os.path.join(root, file)
                print(f"Removing: {file_path}")
                os.remove(file_path)
EOF
        
        # Remove database file
        docker-compose exec django rm -f db.sqlite3
        
        # Recreate migrations and database
        run_django_command makemigrations
        run_django_command migrate
        
        print_status "Database reset completed!"
    else
        print_status "Database reset cancelled"
    fi
}

# Show help
show_help() {
    echo "Django Setup Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  setup     - Full Django setup (migrations + superuser)"
    echo "  migrate   - Run migrations only"
    echo "  user      - Create superuser only"
    echo "  reset     - Reset database (dangerous!)"
    echo "  shell     - Open Django shell"
    echo "  logs      - Show Django logs"
    echo "  test      - Run Django tests"
    echo ""
    echo "Examples:"
    echo "  $0 setup"
    echo "  $0 migrate"
    echo "  $0 user"
}

# Main script logic
case $1 in
    "setup")
        setup_django
        ;;
    "migrate")
        check_containers
        print_status "Creating and applying migrations..."
        run_django_command makemigrations
        run_django_command migrate
        ;;
    "user")
        check_containers
        print_header "Creating Django superuser..."
        run_django_command createsuperuser
        ;;
    "reset")
        check_containers
        reset_database
        ;;
    "shell")
        check_containers
        print_status "Opening Django shell..."
        run_django_command shell
        ;;
    "logs")
        docker-compose logs -f django
        ;;
    "test")
        check_containers
        print_status "Running Django tests..."
        run_django_command test
        ;;
    "dbshell")
        check_containers
        print_status "Opening database shell..."
        run_django_command dbshell
        ;;
    *)
        show_help
        ;;
esac