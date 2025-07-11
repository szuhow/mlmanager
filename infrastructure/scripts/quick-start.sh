#!/bin/bash

# Coronary Experiments - Quick Start Script
# This script helps you get started with the ML Manager application

set -e

echo "🚀 Coronary Experiments ML Manager - Quick Start"
echo "==============================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose v2 is available
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose v2 is not available. Please install Docker Compose v2."
    exit 1
fi

# Function to check if GPU is available
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            echo "✅ NVIDIA GPU detected"
            return 0
        else
            echo "❌ NVIDIA GPU not accessible"
            return 1
        fi
    else
        echo "❌ NVIDIA drivers not found"
        return 1
    fi
}

# Function to check if nvidia-docker is available
check_nvidia_docker() {
    if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA Docker runtime available"
        return 0
    else
        echo "❌ NVIDIA Docker runtime not available"
        return 1
    fi
}

# Ask user for version preference
echo ""
echo "Please select the version to run:"
echo "1) CPU version (recommended for development)"
echo "2) GPU version (requires NVIDIA GPU and Docker runtime)"
echo ""
read -p "Enter your choice (1 or 2): " version_choice

case $version_choice in
    1)
        echo "📱 Selected: CPU version"
        VERSION="cpu"
        ;;
    2)
        echo "🎮 Selected: GPU version"
        if check_gpu && check_nvidia_docker; then
            VERSION="gpu"
        else
            echo "❌ GPU requirements not met. Falling back to CPU version."
            VERSION="cpu"
        fi
        ;;
    *)
        echo "❌ Invalid choice. Using CPU version."
        VERSION="cpu"
        ;;
esac

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created. You can edit it later if needed."
fi

# Build and start services
echo ""
echo "🔨 Building containers..."
make build-${VERSION}

echo ""
echo "🚀 Starting services..."
make start-${VERSION}

echo ""
echo "⏳ Waiting for services to start..."
sleep 15

echo ""
echo "🔄 Running database migrations..."
make migrate-${VERSION}

echo ""
echo "📊 Checking service status..."
make status-${VERSION}

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🌐 Your application is now running at:"
echo "  • Django App: http://localhost:8000"
echo "  • Flower (Celery monitoring): http://localhost:5555"
echo ""
echo "📚 Useful commands:"
echo "  • View logs: make logs-${VERSION}"
echo "  • Stop services: make stop-${VERSION}"
echo "  • Restart services: make restart-${VERSION}"
echo "  • Open shell: make shell-${VERSION}"
echo "  • Clean up: make clean-${VERSION}"
echo ""
echo "📖 For more information, check the README.md file"
echo ""
echo "🎯 To create a Django superuser, run:"
echo "  docker-compose -f docker-compose.${VERSION}.yml exec django python manage.py createsuperuser"
echo ""
echo "Happy coding! 🚀"
