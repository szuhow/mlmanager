#!/bin/bash
# filepath: /Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager/setup.sh

set -e  # Exit on any error

echo "🚀 Starting MLManager Infrastructure Setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env exists
if [ ! -f .env ]; then
    print_warning ".env file not found. Creating template..."
    cat > .env << EOF
# GitHub Container Registry Token
CR_PAT=your_github_token_here

# Django Settings
DEBUG=True
SECRET_KEY=your_secret_key_here

# MLflow Settings
MLFLOW_TRACKING_URI=http://mlflow:5000
EOF
    print_warning "Please edit .env file with your tokens and settings"
    exit 1
fi

# Load environment variables
source .env

# Check if CR_PAT is set
if [ "$CR_PAT" = "your_github_token_here" ] || [ -z "$CR_PAT" ]; then
    print_error "Please set your GitHub token in .env file (CR_PAT variable)"
    exit 1
fi

# Get GitHub username
read -p "Enter your GitHub username: " GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    print_error "GitHub username is required"
    exit 1
fi

print_status "Logging into GitHub Container Registry..."
echo $CR_PAT | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin

if [ $? -ne 0 ]; then
    print_error "Failed to login to GitHub Container Registry"
    exit 1
fi

print_status "Creating required directories..."
mkdir -p mlruns
mkdir -p data/models/artifacts

print_status "Stopping existing containers..."
docker-compose down

print_status "Pulling MLflow image..."
docker pull ghcr.io/mlflow/mlflow:v2.12.1

if [ $? -ne 0 ]; then
    print_warning "Failed to pull MLflow image. Creating custom MLflow image..."
    
    # Create Dockerfile for MLflow if pull fails
    cat > Dockerfile.mlflow << EOF
FROM python:3.11-slim

WORKDIR /mlflow

# Install MLflow and dependencies
RUN pip install mlflow==2.12.1 psycopg2-binary

# Create directories
RUN mkdir -p /mlflow/data /mlflow/artifacts

# Expose port
EXPOSE 5000

# Default command
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
EOF
    
    # Update docker-compose to use custom build
    sed -i.bak 's/image: ghcr.io\/mlflow\/mlflow:v2.12.1/build:\n      context: .\n      dockerfile: Dockerfile.mlflow/' docker-compose.yml
fi

print_status "Building and starting containers..."
docker-compose up --build -d

print_status "Waiting for services to start..."
sleep 10

# Health checks
print_status "Checking service health..."

# Check MLflow
if curl -f http://localhost:5000 >/dev/null 2>&1; then
    print_status "✅ MLflow is running at http://localhost:5000"
else
    print_warning "⚠️  MLflow health check failed"
fi

# Check Django
if curl -f http://localhost:8000 >/dev/null 2>&1; then
    print_status "✅ Django is running at http://localhost:8000"
else
    print_warning "⚠️  Django health check failed"
fi

print_status "Infrastructure setup complete!"
print_status "Services:"
print_status "  - MLflow UI: http://localhost:5000"
print_status "  - Django App: http://localhost:8000"

echo ""
print_status "Useful commands:"
echo "  View logs:     docker-compose logs -f"
echo "  Stop services: docker-compose down"
echo "  Restart:       docker-compose restart"
echo "  Shell access:  docker-compose exec django bash"