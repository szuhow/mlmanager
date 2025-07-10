"""
Management command to synchronize MLflow data with database.
"""

from django.core.management.base import BaseCommand
from django.utils import timezone
import logging
import mlflow
from core.apps.ml_manager.models import MLModel
from core.apps.ml_manager.utils.mlflow_utils import get_mlflow_tracking_uri

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Synchronize MLflow data with database'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be updated without making changes',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force update even if data seems current',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        force = options['force']
        
        self.stdout.write(
            self.style.SUCCESS(f'Starting MLflow data sync (dry_run={dry_run})')
        )
        
        try:
            # Set up MLflow connection
            tracking_uri = get_mlflow_tracking_uri()
            mlflow.set_tracking_uri(tracking_uri)
            
            # Get all models with MLflow run IDs
            models_with_runs = MLModel.objects.exclude(mlflow_run_id__isnull=True).exclude(mlflow_run_id='')
            
            updated_count = 0
            error_count = 0
            
            for model in models_with_runs:
                try:
                    # Get MLflow run data
                    run = mlflow.get_run(model.mlflow_run_id)
                    
                    # Check if we need to update
                    needs_update = force or self._needs_update(model, run)
                    
                    if needs_update:
                        if not dry_run:
                            self._update_model_from_run(model, run)
                        updated_count += 1
                        
                        status_msg = f"{'[DRY RUN] ' if dry_run else ''}Updated model {model.id} ({model.name})"
                        self.stdout.write(self.style.SUCCESS(status_msg))
                        
                except mlflow.exceptions.MlflowException as e:
                    if "RESOURCE_DOES_NOT_EXIST" in str(e):
                        # MLflow run doesn't exist anymore
                        if not dry_run:
                            model.mlflow_run_id = None
                            model.save(update_fields=['mlflow_run_id'])
                        error_msg = f"{'[DRY RUN] ' if dry_run else ''}Cleared missing MLflow run for model {model.id}"
                        self.stdout.write(self.style.WARNING(error_msg))
                        error_count += 1
                    else:
                        self.stdout.write(
                            self.style.ERROR(f'MLflow error for model {model.id}: {e}')
                        )
                        error_count += 1
                        
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f'Error processing model {model.id}: {e}')
                    )
                    error_count += 1
            
            # Clear experiments cache to refresh it
            try:
                from django.core.cache import cache
                cache.delete('mlflow_experiments')
                self.stdout.write(self.style.SUCCESS('Cleared experiments cache'))
            except ImportError:
                pass
            
            summary = f'Sync complete: {updated_count} updated, {error_count} errors'
            self.stdout.write(self.style.SUCCESS(summary))
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Failed to sync MLflow data: {e}')
            )
            
    def _needs_update(self, model, run):
        """Check if model needs update based on MLflow run data"""
        # Simple check - update if run status changed or if data is older than 1 hour
        run_status = run.info.status
        
        # Map MLflow status to our status
        if run_status == 'FINISHED':
            expected_status = 'completed'
        elif run_status == 'FAILED':
            expected_status = 'failed'
        elif run_status == 'RUNNING':
            expected_status = 'training'
        else:
            expected_status = model.status
            
        return model.status != expected_status
        
    def _update_model_from_run(self, model, run):
        """Update model fields from MLflow run data"""
        run_status = run.info.status
        run_metrics = run.data.metrics
        
        # Map MLflow status to our status
        if run_status == 'FINISHED':
            model.status = 'completed'
        elif run_status == 'FAILED':
            model.status = 'failed'
        elif run_status == 'RUNNING':
            model.status = 'training'
            
        # Update metrics if available
        if 'val_dice' in run_metrics:
            model.val_dice = run_metrics['val_dice']
        if 'train_dice' in run_metrics:
            model.train_dice = run_metrics['train_dice']
        if 'val_loss' in run_metrics:
            model.val_loss = run_metrics['val_loss']
        if 'train_loss' in run_metrics:
            model.train_loss = run_metrics['train_loss']
        if 'best_val_dice' in run_metrics:
            model.best_val_dice = run_metrics['best_val_dice']
            
        # Update epoch info if available
        if 'epoch' in run_metrics:
            model.current_epoch = int(run_metrics['epoch'])
            
        model.save()
        logger.info(f"Updated model {model.id} from MLflow run {run.info.run_id}")
