import os
import logging
import psutil
from django.apps import AppConfig


class MlManagerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core.apps.ml_manager'
    
    def ready(self):
        """Called when the app is fully loaded - perfect place for startup checks"""
        # Import here to avoid Django setup issues
        from .models import MLModel
        from django.db import connection
        
        logger = logging.getLogger(__name__)
        
        # Check environment and process type to avoid duplicate logging
        run_main = os.environ.get('RUN_MAIN')
        django_settings = os.environ.get('DJANGO_SETTINGS_MODULE', '')
        process_type = 'worker' if 'worker' in django_settings else 'django'
        
        # Only log startup in main Django process or when explicitly requested
        if run_main or (django_settings and not 'worker' in django_settings):
            logger.info("🚀 ML Manager app ready() called")
            
            # Initialize MLflow connection
            try:
                from .utils.mlflow_utils import initialize_mlflow_connection
                success = initialize_mlflow_connection()
                if success:
                    logger.info("[ML_MANAGER] MLflow connection initialized successfully")
                else:
                    logger.warning("[ML_MANAGER] MLflow connection initialization skipped or failed")
            except Exception as e:
                logger.error(f"[ML_MANAGER] Error initializing MLflow: {e}")
            
            # Check environment
            logger.info(f"🔍 Environment check: RUN_MAIN={run_main}, DJANGO_SETTINGS_MODULE={django_settings}")
            
            # Run validation in development or if RUN_MAIN is set (main Django process)
            if run_main or django_settings.endswith('development') or django_settings.endswith('production'):
                logger.info("✅ Conditions met for running startup validation")
                try:
                    # Check if database is ready
                    with connection.cursor() as cursor:
                        cursor.execute("SELECT 1")
                    logger.info("✅ Database connection verified")
                    
                    # Perform startup training status validation
                    self.validate_training_statuses()
                    
                    # Perform MLflow synchronization
                    self.sync_mlflow_on_startup()
                    
                except Exception as e:
                    # Log the error but don't crash the application
                    logger.warning(f"❌ Failed to validate training statuses on startup: {e}")
            else:
                logger.info("⏭️  Skipping startup validation (not main process)")
        else:
            # Silent initialization for workers - no duplicate logging
            try:
                from .utils.mlflow_utils import initialize_mlflow_connection
                initialize_mlflow_connection()
            except Exception:
                pass  # Silent failure for workers
    
    def validate_training_statuses(self):
        """Check and update orphaned training statuses after container restart"""
        from .models import MLModel
        from django.utils import timezone
        from datetime import timedelta
        
        logger = logging.getLogger(__name__)
        logger.info("🔍 Validating training statuses after potential container restart...")
        
        # Find models with potentially orphaned training status
        orphaned_models = MLModel.objects.filter(status__in=['training', 'loading'])
        
        if not orphaned_models.exists():
            logger.info("✅ No potentially orphaned training models found")
            return
        
        logger.info(f"🔍 Found {orphaned_models.count()} models with training/loading status - validating...")
        
        # Grace period for recently created models to avoid race conditions
        grace_period = timedelta(minutes=5)
        now = timezone.now()
        
        corrected_count = 0
        for model in orphaned_models:
            # Check if model was created recently - give it time to start training
            model_age = now - model.created_at
            if model_age < grace_period:
                logger.info(f"⏰ Model {model.id} ({model.name}) created {model_age.total_seconds():.1f}s ago - skipping validation (grace period)")
                continue
                
            if self.is_training_process_active(model):
                logger.info(f"✅ Model {model.id} ({model.name}) - training process is active")
            else:
                # Training process is not active - update status
                old_status = model.status
                model.status = 'failed'
                model.save()
                corrected_count += 1
                logger.warning(f"🔧 Model {model.id} ({model.name}) - corrected orphaned status '{old_status}' → 'failed' (age: {model_age.total_seconds():.1f}s)")
        
        if corrected_count > 0:
            logger.info(f"🎯 Corrected {corrected_count} orphaned training statuses")
        else:
            logger.info("✅ All training statuses are valid")
    
    def is_training_process_active(self, model):
        """Check if a training process is actually running for this model using direct training manager"""
        
        logger = logging.getLogger(__name__)
        
        try:
            # Import direct training manager
            from .utils.direct_training import training_manager
            
            # Check if this model is currently training
            if training_manager.is_training_active():
                active_model_id = training_manager.get_active_training_model_id()
                if active_model_id == model.id:
                    logger.info(f"✅ Found active direct training for model {model.id}")
                    return True
                else:
                    logger.info(f"⚠️  Model {model.id} marked as training but active training is for model {active_model_id}")
                    return False
            else:
                logger.info(f"💤 No active training for model {model.id}")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️  Could not check direct training for model {model.id}: {e}")
            # Fallback to legacy process checking
            return self._check_process_active(model, logger)
    
    def _check_process_active(self, model, logger):
        """Check if there are local Python processes running for this model (legacy method)"""
        try:
            # Check if there are any Python processes running train.py with this model ID
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        cmdline = proc.info['cmdline']
                        if (cmdline and 
                            any('train.py' in str(arg) for arg in cmdline) and
                            any(f'--model-id={model.id}' in str(arg) for arg in cmdline)):
                            logger.info(f"✅ Found active process for model {model.id}: PID {proc.info['pid']}")
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return False
            
        except Exception as e:
            logger.warning(f"⚠️  Could not check processes for model {model.id}: {e}")
            return False

    def sync_mlflow_on_startup(self):
        """Synchronize MLflow status on application startup"""
        logger = logging.getLogger(__name__)
        logger.info("🔄 Starting MLflow synchronization on startup...")
        
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            client = MlflowClient()
            updated_count = 0
            error_count = 0
            
            # Get models with MLflow run IDs that might need syncing
            models_with_runs = self._get_models_needing_sync()
            
            if not models_with_runs:
                logger.info("✅ No models need MLflow synchronization")
                return
            
            logger.info(f"🔍 Synchronizing {len(models_with_runs)} models with MLflow...")
            
            for model in models_with_runs:
                try:
                    old_status = model.status
                    updated = self._sync_model_with_mlflow(model, client)
                    
                    if updated:
                        updated_count += 1
                        logger.info(f"🔧 Model {model.id} ({model.name}): {old_status} → {model.status}")
                        
                except Exception as e:
                    error_count += 1
                    logger.warning(f"⚠️  Error syncing model {model.id}: {e}")
            
            if updated_count > 0:
                logger.info(f"✅ MLflow sync completed: {updated_count} models updated, {error_count} errors")
            else:
                logger.info("✅ MLflow sync completed: all models already in sync")
                
        except ImportError:
            logger.info("⏭️  MLflow not available, skipping synchronization")
        except Exception as e:
            logger.warning(f"⚠️  MLflow synchronization failed: {e}")
    
    def _get_models_needing_sync(self):
        """Get models that might need MLflow synchronization"""
        try:
            from .models import MLModel
            from django.utils import timezone
            from datetime import timedelta
            
            # Focus on models that:
            # 1. Have MLflow run IDs
            # 2. Are in training/pending status (most likely to be stale)
            # 3. Are older than 1 hour (avoid interfering with fresh training)
            
            one_hour_ago = timezone.now() - timedelta(hours=1)
            
            models = MLModel.objects.filter(
                mlflow_run_id__isnull=False,
                status__in=['training', 'pending'],
                created_at__lt=one_hour_ago
            ).exclude(mlflow_run_id='')
            
            return list(models)
            
        except Exception as e:
            # Handle database schema issues (e.g., during migrations)
            logger = logging.getLogger(__name__)
            logger.debug(f"Cannot query models for MLflow sync: {e}")
            return []
    
    def _sync_model_with_mlflow(self, model, client):
        """Sync a single model with its MLflow run"""
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
                model.status = new_status
                model.save()
                return True
                
            return False
            
        except Exception as e:
            # If MLflow run doesn't exist, clear the reference
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                model.mlflow_run_id = None
                model.status = 'failed'
                model.save()
                return True
            else:
                raise
