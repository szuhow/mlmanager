#!/bin/bash
# filepath: /Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager/dev.sh

# Development helper script

case $1 in
    "start")
        echo "🚀 Starting development environment..."
        docker-compose up -d
        ;;
    "stop")
        echo "🛑 Stopping development environment..."
        docker-compose down
        ;;
    "restart")
        echo "🔄 Restarting development environment..."
        docker-compose restart
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "shell")
        docker-compose exec django bash
        ;;
    "rebuild")
        echo "🏗️  Rebuilding containers..."
        docker-compose down
        docker-compose up --build -d
        ;;
    "clean")
        echo "🧹 Cleaning up..."
        docker-compose down
        docker system prune -f
        docker volume prune -f
        ;;
    "rebuild")
        echo "🏗️  Rebuilding containers..."
        docker-compose down
        docker-compose build --no-cache
        docker-compose up -d
        ;;
    "rebuild-fast")
        echo "🏗️  Fast rebuilding containers..."
        docker-compose down
        docker-compose up --build -d
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|shell|rebuild|clean}"
        echo ""
        echo "Commands:"
        echo "  start   - Start all services"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        echo "  logs    - Show and follow logs"
        echo "  shell   - Open shell in Django container"
        echo "  rebuild - Rebuild and restart containers"
        echo "  clean   - Clean up containers and volumes"
        ;;
esac