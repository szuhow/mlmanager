from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
import logging
from core.apps.ml_manager.models import MLModel

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Synchronize model statuses with MLflow and cleanup orphaned runs'

    def add_arguments(self, parser):
        parser.add_argument(
            '--full',
            action='store_true',
            help='Perform full synchronization including cleanup',
        )
        parser.add_argument(
            '--cleanup-only',
            action='store_true',
            help='Only cleanup orphaned runs, skip status sync',
        )
        parser.add_argument(
            '--auto',
            action='store_true',
            help='Run automatically on startup (less verbose)',
        )
        parser.add_argument(
            '--max-age-hours',
            type=int,
            default=24,
            help='Maximum age in hours for considering runs as potentially orphaned (default: 24)',
        )
        parser.add_argument(
            '--force-end-run',
            type=str,
            help='Force end a specific MLflow run by ID',
        )

    def handle(self, *args, **options):
        if options['auto']:
            self.stdout.write('Running MLflow auto-synchronization on startup...')
        else:
            self.stdout.write(self.style.SUCCESS('Starting MLflow synchronization...'))

        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            client = MlflowClient()
            
            # Handle force-end-run first (single action)
            if options['force_end_run']:
                result = self._force_end_run(client, options['force_end_run'])
                if result['success']:
                    self.stdout.write(self.style.SUCCESS(f"Successfully ended run {options['force_end_run']}"))
                else:
                    self.stdout.write(self.style.ERROR(f"Failed to end run {options['force_end_run']}: {result['error']}"))
                return
            
            if not options['cleanup_only']:
                # Sync statuses
                sync_results = self._sync_statuses(client, options['auto'])
                if not options['auto']:
                    self._print_sync_results(sync_results)
            
            if options['full'] or options['cleanup_only']:
                # Cleanup orphaned runs
                cleanup_results = self._cleanup_orphaned_runs(client, options['max_age_hours'], options['auto'])
                if not options['auto']:
                    self._print_cleanup_results(cleanup_results)

            if options['auto']:
                self.stdout.write('MLflow auto-synchronization completed')
            else:
                self.stdout.write(self.style.SUCCESS('MLflow synchronization completed successfully'))

        except ImportError:
            self.stdout.write(self.style.ERROR('MLflow not available'))
        except Exception as e:
            logger.error(f"Error in MLflow synchronization: {e}")
            self.stdout.write(self.style.ERROR(f'Error: {e}'))

    def _sync_statuses(self, client, auto_mode=False):
        """Synchronize model statuses with MLflow"""
        updated_models = []
        errors = []
        
        models_with_runs = MLModel.objects.filter(
            mlflow_run_id__isnull=False
        ).exclude(mlflow_run_id='')
        
        if not auto_mode:
            self.stdout.write(f'Checking {models_with_runs.count()} models with MLflow runs...')
        
        for model in models_with_runs:
            try:
                run = client.get_run(model.mlflow_run_id)
                mlflow_status = run.info.status
                
                # Map MLflow status to our model status
                status_mapping = {
                    'RUNNING': 'training',
                    'FINISHED': 'completed',
                    'FAILED': 'failed',
                    'KILLED': 'stopped'
                }
                
                new_status = status_mapping.get(mlflow_status, model.status)
                
                if new_status != model.status:
                    old_status = model.status
                    model.status = new_status
                    model.save()
                    
                    updated_models.append({
                        'model_id': model.id,
                        'model_name': model.name,
                        'old_status': old_status,
                        'new_status': new_status,
                        'mlflow_status': mlflow_status
                    })
                    
                    if not auto_mode:
                        self.stdout.write(
                            f'  Updated model {model.id} ({model.name}): {old_status} → {new_status}'
                        )
                
            except Exception as e:
                error_msg = f"Model {model.id} ({model.name}): {str(e)}"
                errors.append(error_msg)
                if not auto_mode:
                    self.stdout.write(self.style.WARNING(f'  Error: {error_msg}'))
        
        return {
            'updated_models': updated_models,
            'errors': errors,
            'total_checked': models_with_runs.count()
        }

    def _cleanup_orphaned_runs(self, client, max_age_hours, auto_mode=False):
        """Clean up orphaned MLflow runs"""
        cleaned_models = []
        ended_runs = []
        errors = []
        
        # Get potentially stale models
        cutoff_time = timezone.now() - timedelta(hours=max_age_hours)
        stale_models = MLModel.objects.filter(
            status__in=['training', 'pending'],
            created_at__lt=cutoff_time,
            mlflow_run_id__isnull=False
        ).exclude(mlflow_run_id='')
        
        if not auto_mode:
            self.stdout.write(f'Checking {stale_models.count()} potentially stale models...')
        
        for model in stale_models:
            try:
                run = client.get_run(model.mlflow_run_id)
                mlflow_status = run.info.status
                
                if mlflow_status == 'RUNNING':
                    # Check if the run is actually active
                    last_logged = run.info.end_time or run.info.start_time
                    
                    if last_logged:
                        # Convert MLflow timestamp (milliseconds) to datetime
                        last_log_time = timezone.datetime.fromtimestamp(last_logged / 1000, tz=timezone.utc)
                        if timezone.now() - last_log_time > timedelta(hours=2):
                            # Run seems abandoned, end it
                            try:
                                with mlflow.start_run(run_id=model.mlflow_run_id):
                                    mlflow.end_run(status='KILLED')
                                
                                ended_runs.append({
                                    'run_id': model.mlflow_run_id,
                                    'model_id': model.id,
                                    'reason': 'No activity for >2 hours'
                                })
                                
                                # Update model status
                                model.status = 'failed'
                                model.save()
                                
                                if not auto_mode:
                                    self.stdout.write(f'  Ended orphaned run for model {model.id}')
                                
                            except Exception as end_error:
                                errors.append(f"Failed to end run {model.mlflow_run_id}: {end_error}")
                
                elif mlflow_status in ['FINISHED', 'FAILED', 'KILLED']:
                    # Update model status to match MLflow
                    status_mapping = {
                        'FINISHED': 'completed',
                        'FAILED': 'failed',
                        'KILLED': 'stopped'
                    }
                    
                    old_status = model.status
                    model.status = status_mapping[mlflow_status]
                    model.save()
                    
                    cleaned_models.append({
                        'model_id': model.id,
                        'old_status': old_status,
                        'new_status': model.status,
                        'mlflow_status': mlflow_status
                    })
                    
                    if not auto_mode:
                        self.stdout.write(f'  Updated stale model {model.id}: {old_status} → {model.status}')
                
            except mlflow.exceptions.MlflowException as e:
                if "RESOURCE_DOES_NOT_EXIST" in str(e):
                    # MLflow run doesn't exist, clear the reference
                    old_status = model.status
                    model.mlflow_run_id = None
                    model.status = 'failed'
                    model.save()
                    
                    cleaned_models.append({
                        'model_id': model.id,
                        'old_status': old_status,
                        'new_status': 'failed',
                        'reason': 'MLflow run not found'
                    })
                    
                    if not auto_mode:
                        self.stdout.write(f'  Cleared orphaned reference for model {model.id}')
                else:
                    errors.append(f"Model {model.id}: {str(e)}")
            
            except Exception as e:
                errors.append(f"Model {model.id}: {str(e)}")
        
        return {
            'cleaned_models': cleaned_models,
            'ended_runs': ended_runs,
            'errors': errors,
            'total_checked': stale_models.count()
        }

    def _print_sync_results(self, results):
        """Print sync results"""
        self.stdout.write(f"\nStatus Synchronization Results:")
        self.stdout.write(f"  Total models checked: {results['total_checked']}")
        self.stdout.write(f"  Models updated: {len(results['updated_models'])}")
        self.stdout.write(f"  Errors: {len(results['errors'])}")
        
        if results['errors']:
            self.stdout.write(self.style.WARNING("\nErrors encountered:"))
            for error in results['errors']:
                self.stdout.write(f"  - {error}")

    def _print_cleanup_results(self, results):
        """Print cleanup results"""
        self.stdout.write(f"\nCleanup Results:")
        self.stdout.write(f"  Total models checked: {results['total_checked']}")
        self.stdout.write(f"  Models cleaned: {len(results['cleaned_models'])}")
        self.stdout.write(f"  Runs ended: {len(results['ended_runs'])}")
        self.stdout.write(f"  Errors: {len(results['errors'])}")
        
        if results['errors']:
            self.stdout.write(self.style.WARNING("\nCleanup errors:"))
            for error in results['errors']:
                self.stdout.write(f"  - {error}")

    def _force_end_run(self, client, run_id):
        """Force end a specific MLflow run"""
        try:
            import mlflow
            
            # Get run info first
            run = client.get_run(run_id)
            if run.info.status in ['FINISHED', 'FAILED', 'KILLED']:
                return {'success': False, 'error': f'Run {run_id} is already in terminal state: {run.info.status}'}
            
            # Force end the run
            with mlflow.start_run(run_id=run_id):
                mlflow.end_run(status='KILLED')
            
            return {'success': True, 'error': None}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
