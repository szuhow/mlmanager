from django.core.management.base import BaseCommand
import logging
import mlflow

from core.apps.ml_manager.models import MLModel

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Synchronize metrics from MLflow to the Django database'

    def add_arguments(self, parser):
        parser.add_argument(
            '--model-id',
            type=str,
            help='The ID or name of a specific model to synchronize metrics for',
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Synchronize metrics for all models with MLflow run IDs',
        )

    def handle(self, *args, **options):
        model_id = options.get('model_id')
        sync_all = options.get('all')
        
        if not model_id and not sync_all:
            self.stdout.write(self.style.ERROR('Please specify either --model-id or --all option'))
            return
            
        try:
            # Set up MLflow connection
            mlflow.set_tracking_uri('http://mlflow:5000')
            
            if model_id:
                # Try to find model by ID first
                try:
                    model_id_int = int(model_id)
                    model = MLModel.objects.filter(id=model_id_int).first()
                except ValueError:
                    # Not an integer, try by name
                    model = MLModel.objects.filter(name=model_id).first()
                
                if not model:
                    self.stdout.write(self.style.ERROR(f"Model with ID or name '{model_id}' not found"))
                    return
                    
                self.stdout.write(f"Synchronizing metrics for model {model.id} ({model.name})...")
                success = self._sync_model_metrics(model)
                
                if success:
                    self.stdout.write(self.style.SUCCESS(f"Successfully synchronized metrics for model {model.id} ({model.name})"))
                else:
                    self.stdout.write(self.style.ERROR(f"Failed to synchronize metrics for model {model.id} ({model.name})"))
                    
            elif sync_all:
                models = MLModel.objects.filter(
                    mlflow_run_id__isnull=False
                ).exclude(mlflow_run_id='')
                
                self.stdout.write(f"Synchronizing metrics for {models.count()} models...")
                
                success_count = 0
                error_count = 0
                
                for model in models:
                    self.stdout.write(f"  Syncing model {model.id} ({model.name})...")
                    success = self._sync_model_metrics(model)
                    
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                
                self.stdout.write(self.style.SUCCESS(
                    f"Finished syncing metrics: {success_count} models updated, {error_count} errors"
                ))
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error synchronizing metrics: {e}"))
            logger.exception("Error in sync_metrics command")
    
    def _sync_model_metrics(self, model):
        """Synchronize metrics from MLflow to Django for a single model"""
        try:
            if not model.mlflow_run_id:
                self.stdout.write(self.style.WARNING(f"Model {model.id} has no MLflow run ID"))
                return False
            
            # Get MLflow run data
            run = mlflow.get_run(model.mlflow_run_id)
            metrics = run.data.metrics
            
            if not metrics:
                self.stdout.write(self.style.WARNING(f"No metrics found in MLflow for model {model.id}"))
                return False
            
            # Store current metrics for comparison
            old_metrics = {
                'best_val_dice': model.best_val_dice,
                'best_val_iou': model.best_val_iou,
                'val_dice': model.val_dice,
                'val_iou': model.val_iou,
                'train_dice': model.train_dice,
                'train_iou': model.train_iou,
                'train_loss': model.train_loss,
                'val_loss': model.val_loss
            }
            
            # Update Django model with MLflow metrics
            if 'best_val_dice' in metrics:
                model.best_val_dice = metrics['best_val_dice']
            
            if 'best_val_iou' in metrics:
                model.best_val_iou = metrics['best_val_iou']
            
            # Update current epoch metrics
            if 'val_dice' in metrics:
                model.val_dice = metrics['val_dice']
            if 'val_iou' in metrics:
                model.val_iou = metrics['val_iou']
            if 'train_dice' in metrics:
                model.train_dice = metrics['train_dice']
            if 'train_iou' in metrics:
                model.train_iou = metrics['train_iou']
            if 'train_loss' in metrics:
                model.train_loss = metrics['train_loss']
            if 'val_loss' in metrics:
                model.val_loss = metrics['val_loss']
            
            model.save()
            
            # Show what metrics were updated
            updated_metrics = []
            for key in old_metrics:
                mlflow_value = metrics.get(key, None)
                if mlflow_value is not None and old_metrics[key] != getattr(model, key):
                    updated_metrics.append(f"{key}: {old_metrics[key]} → {getattr(model, key)}")
            
            if updated_metrics:
                self.stdout.write(self.style.SUCCESS("  Updated metrics:"))
                for update in updated_metrics:
                    self.stdout.write(f"    - {update}")
            else:
                self.stdout.write(self.style.WARNING("  No metrics were updated (already in sync)"))
            
            return True
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  Error syncing model {model.id}: {e}"))
            logger.exception(f"Error syncing metrics for model {model.id}")
            return False
