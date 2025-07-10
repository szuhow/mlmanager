#!/bin/bash
# filepath: /Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager/scripts/quick-setup.sh

# Quick setup script for development environment

echo "🚀 Quick MLManager Setup"
echo "======================="

# Make scripts executable
chmod +x scripts/setup.sh scripts/dev.sh scripts/django-setup.sh

# Start infrastructure
echo "1. Starting infrastructure..."
./scripts/dev.sh start

# Wait for containers
echo "2. Waiting for containers to start..."
sleep 15

# Setup Django
echo "3. Setting up Django..."
./scripts/django-setup.sh setup

echo ""
echo "✅ Setup completed!"
echo ""
echo "📍 Services:"
echo "  - Django:  http://localhost:8000"
echo "  - MLflow:  http://localhost:5001"
echo "  - Admin:   http://localhost:8000/admin"
echo ""
echo "🛠️  Useful commands:"
echo "  ./scripts/dev.sh logs           # View logs"
echo "  ./scripts/django-setup.sh shell # Django shell"
echo "  ./scripts/dev.sh stop           # Stop services"