from django.core.management.base import BaseCommand
from django.conf import settings
from ml_manager.models import MLModel, Prediction, TrainingTemplate
import mlflow
import logging
import os
import shutil

class Command(BaseCommand):
    help = 'Clean up orphaned MLflow references and optionally reset all data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--reset-all',
            action='store_true',
            help='Reset all MLflow data and model records',
        )
        parser.add_argument(
            '--fix-orphaned',
            action='store_true',
            help='Fix models with missing MLflow runs',
        )
        parser.add_argument(
            '--clean-mlflow',
            action='store_true',
            help='Clean MLflow database and runs directory',
        )

    def handle(self, *args, **options):
        if options['reset_all']:
            self.reset_all_data()
        elif options['fix_orphaned']:
            self.fix_orphaned_models()
        elif options['clean_mlflow']:
            self.clean_mlflow_data()
        else:
            self.stdout.write(
                self.style.WARNING(
                    'Please specify an action: --reset-all, --fix-orphaned, or --clean-mlflow'
                )
            )
            self.stdout.write('Use --help for more information')

    def reset_all_data(self):
        """Reset all data - models, predictions, templates, and MLflow"""
        self.stdout.write(self.style.WARNING('Resetting ALL data...'))
        
        # Delete all model records
        model_count = MLModel.objects.count()
        prediction_count = Prediction.objects.count()
        template_count = TrainingTemplate.objects.count()
        
        MLModel.objects.all().delete()
        Prediction.objects.all().delete()
        TrainingTemplate.objects.all().delete()
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Deleted {model_count} models, {prediction_count} predictions, '
                f'and {template_count} templates from database'
            )
        )
        
        # Clean MLflow data
        self.clean_mlflow_data()
        
        self.stdout.write(self.style.SUCCESS('All data has been reset!'))

    def fix_orphaned_models(self):
        """Fix models that reference non-existent MLflow runs"""
        self.stdout.write('Checking for orphaned MLflow references...')
        
        try:
            client = mlflow.tracking.MlflowClient()
            fixed_count = 0
            
            for model in MLModel.objects.all():
                if model.mlflow_run_id:
                    try:
                        # Try to get the run
                        client.get_run(model.mlflow_run_id)
                        self.stdout.write(f'Model {model.id} ({model.name}): MLflow run OK')
                    except Exception as e:
                        self.stdout.write(
                            self.style.WARNING(
                                f'Model {model.id} ({model.name}): Fixing orphaned run {model.mlflow_run_id}'
                            )
                        )
                        # Clear the orphaned MLflow run ID
                        model.mlflow_run_id = None
                        model.save()
                        fixed_count += 1
                else:
                    self.stdout.write(f'Model {model.id} ({model.name}): No MLflow run ID')
            
            self.stdout.write(
                self.style.SUCCESS(f'Fixed {fixed_count} orphaned MLflow references')
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error accessing MLflow: {e}')
            )

    def clean_mlflow_data(self):
        """Clean MLflow database and runs directory"""
        self.stdout.write('Cleaning MLflow data...')
        
        # Remove MLflow database
        mlflow_db_path = 'mlflow.db'
        if os.path.exists(mlflow_db_path):
            os.remove(mlflow_db_path)
            self.stdout.write(f'Removed {mlflow_db_path}')
        
        # Remove mlruns directory
        mlruns_path = 'mlruns'
        if os.path.exists(mlruns_path):
            shutil.rmtree(mlruns_path)
            self.stdout.write(f'Removed {mlruns_path} directory')
        
        # Remove any training logs
        training_logs_path = 'training_logs'
        if os.path.exists(training_logs_path):
            shutil.rmtree(training_logs_path)
            self.stdout.write(f'Removed {training_logs_path} directory')
        
        # Remove model artifacts
        models_path = 'models'
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
            self.stdout.write(f'Removed {models_path} directory')
        
        # Remove organized artifacts directory
        artifacts_path = 'artifacts'
        if os.path.exists(artifacts_path):
            shutil.rmtree(artifacts_path)
            self.stdout.write(f'Removed {artifacts_path} directory')
        
        # Remove any standalone artifact files in project root
        artifact_patterns = [
            'training_*.log',
            'model_summary*.txt',
            'predictions_epoch_*.png',
            'training_curves_epoch_*.png',
            'training_config*.json',
            'enhanced_training_curves_*.png',
            'performance_radar_*.png',
            'detailed_metrics_*.json'
        ]
        
        import glob
        for pattern in artifact_patterns:
            for file_path in glob.glob(pattern):
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.stdout.write(f'Removed {file_path}')
        
        self.stdout.write(self.style.SUCCESS('MLflow data cleaned!'))
