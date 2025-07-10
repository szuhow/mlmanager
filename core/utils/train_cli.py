#!/usr/bin/env python
"""
CLI Training Script - Train models without GUI
Usage:
    python train_cli.py --model-name "My Model" --dataset-id 1 --model-type classification
    python train_cli.py --config config.json
"""

import os
import sys
import json
import argparse
import requests
from getpass import getpass

# Django setup
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')

import django
django.setup()

from django.contrib.auth import authenticate
from core.apps.ml_manager.models import MLModel, TrainingTemplate
from core.apps.dataset_manager.models import Dataset

class TrainingCLI:
    def __init__(self):
        self.api_base = "http://localhost:8000/api/ml"
        self.session = requests.Session()
        self.token = None
    
    def authenticate(self, username=None, password=None):
        """Authenticate user"""
        if not username:
            username = input("Username: ")
        if not password:
            password = getpass("Password: ")
        
        # Use Django authentication
        user = authenticate(username=username, password=password)
        if user:
            print(f"✅ Authenticated as {username}")
            return user
        else:
            print("❌ Authentication failed")
            return None
    
    def list_datasets(self):
        """List available datasets"""
        try:
            datasets = Dataset.objects.filter(status='ready').order_by('-created_at')
            
            print("\n📊 Available Datasets:")
            print("-" * 80)
            print(f"{'ID':<5} {'Name':<30} {'Type':<15} {'Samples':<10} {'Created':<20}")
            print("-" * 80)
            
            for dataset in datasets:
                print(f"{dataset.id:<5} {dataset.name:<30} {dataset.get_format_type_display():<15} {dataset.total_samples:<10} {dataset.created_at.strftime('%Y-%m-%d %H:%M'):<20}")
            
            if not datasets.exists():
                print("No datasets available")
            
        except Exception as e:
            print(f"❌ Error listing datasets: {e}")
    
    def list_models(self):
        """List available models"""
        try:
            models = MLModel.objects.all().order_by('-created_at')
            
            print("\n🤖 Available Models:")
            print("-" * 100)
            print(f"{'ID':<5} {'Name':<30} {'Type':<15} {'Version':<10} {'Status':<10} {'Created':<20}")
            print("-" * 100)
            
            for model in models:
                print(f"{model.id:<5} {model.name:<30} {model.model_type:<15} {model.version:<10} {model.status:<10} {model.created_at.strftime('%Y-%m-%d %H:%M'):<20}")
            
            if not models.exists():
                print("No models available")
                
        except Exception as e:
            print(f"❌ Error listing models: {e}")
    
    def start_training(self, config):
        """Start training with given configuration"""
        try:
            # Validate dataset exists
            try:
                dataset = Dataset.objects.get(id=config['dataset_id'])
                if dataset.status != 'ready':
                    print(f"❌ Dataset '{dataset.name}' is not ready (status: {dataset.status})")
                    return False
            except Dataset.DoesNotExist:
                print(f"❌ Dataset with ID {config['dataset_id']} not found")
                return False
            
            # Start training using existing ml.training module
            from ml.training.train import start_training
            
            training_config = {
                'model_name': config['model_name'],
                'model_type': config['model_type'],
                'architecture': config.get('architecture', 'resnet50'),
                'dataset_path': dataset.file_path,
                'hyperparameters': config.get('hyperparameters', {}),
                'output_dir': f'/app/data/models/{config["model_name"].replace(" ", "_")}',
                'description': config.get('description', 'CLI Training')
            }
            
            print(f"✅ Starting training with configuration:")
            print(f"   Model Name: {training_config['model_name']}")
            print(f"   Dataset: {dataset.name}")
            print(f"   Model Type: {training_config['model_type']}")
            print(f"   Architecture: {training_config['architecture']}")
            
            # Start training
            result = start_training(**training_config)
            
            if result:
                print(f"🚀 Training started successfully!")
                return True
            else:
                print(f"❌ Training failed to start")
                return False
            
        except ImportError:
            print(f"❌ Training module not available. Starting basic training...")
            # Create MLModel entry for tracking
            model = MLModel.objects.create(
                name=config['model_name'],
                model_type=config['model_type'],
                version="1.0.0",
                status='training',
                description=config.get('description', 'CLI Training'),
                training_data_info={
                    'dataset_id': config['dataset_id'],
                    'dataset_name': dataset.name
                },
                model_architecture_info={
                    'architecture': config.get('architecture', 'resnet50'),
                    'hyperparameters': config.get('hyperparameters', {})
                }
            )
            
            print(f"✅ Model created with ID: {model.id}")
            print(f"⚠️  Note: Full training pipeline not available. Model entry created for tracking.")
            return True
            
        except Exception as e:
            print(f"❌ Error starting training: {e}")
            return False
    
    def get_training_status(self, model_id):
        """Get training status"""
        try:
            model = MLModel.objects.get(id=model_id)
            
            print(f"\n📊 Model Status:")
            print(f"   Model ID: {model.id}")
            print(f"   Name: {model.name}")
            print(f"   Type: {model.model_type}")
            print(f"   Version: {model.version}")
            print(f"   Status: {model.status}")
            print(f"   Created: {model.created_at}")
            print(f"   Updated: {model.updated_at}")
            
            if model.training_data_info:
                print(f"   Training Data: {model.training_data_info}")
            
            if model.performance_metrics:
                print(f"   Performance: {model.performance_metrics}")
            
            return True
            
        except MLModel.DoesNotExist:
            print(f"❌ Model with ID {model_id} not found")
            return False
        except Exception as e:
            print(f"❌ Error getting model status: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='CLI Training for Coronary Experiments')
    
    # Commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List datasets
    list_datasets_parser = subparsers.add_parser('list-datasets', help='List available datasets')
    
    # List models
    list_models_parser = subparsers.add_parser('list-models', help='List available models')
    
    # Start training
    train_parser = subparsers.add_parser('train', help='Start training')
    train_parser.add_argument('--model-name', required=True, help='Name for the new model')
    train_parser.add_argument('--dataset-id', type=int, required=True, help='Dataset ID to use for training')
    train_parser.add_argument('--model-type', required=True, choices=['classification', 'segmentation', 'detection'], help='Model type')
    train_parser.add_argument('--architecture', default='resnet50', help='Model architecture (default: resnet50)')
    train_parser.add_argument('--description', default='CLI Training', help='Model description')
    train_parser.add_argument('--config', help='JSON config file')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    
    # Get status
    status_parser = subparsers.add_parser('status', help='Get model status')
    status_parser.add_argument('--model-id', type=int, required=True, help='Model ID')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = TrainingCLI()
    
    # Handle commands
    if args.command == 'list-datasets':
        cli.list_datasets()
    
    elif args.command == 'list-models':
        cli.list_models()
    
    elif args.command == 'train':
        if args.config:
            # Load from config file
            try:
                with open(args.config, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                print(f"❌ Error loading config file: {e}")
                return
        else:
            # Use command line arguments
            config = {
                'model_name': args.model_name,
                'dataset_id': args.dataset_id,
                'model_type': args.model_type,
                'architecture': args.architecture,
                'description': args.description,
                'hyperparameters': {
                    'learning_rate': args.learning_rate,
                    'batch_size': args.batch_size,
                    'epochs': args.epochs
                }
            }
        
        cli.start_training(config)
    
    elif args.command == 'status':
        cli.get_training_status(args.model_id)

if __name__ == "__main__":
    main()
