import os
import psutil
import logging
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
        run_main = os.environ.get('RUN_MAIN')
        django_settings = os.environ.get('DJANGO_SETTINGS_MODULE', '')
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
                
            except Exception as e:
                # Log the error but don't crash the application
                logger.warning(f"❌ Failed to validate training statuses on startup: {e}")
        else:
            logger.info("⏭️  Skipping startup validation (not main process)")
    
    def validate_training_statuses(self):
        """Check and update orphaned training statuses after container restart"""
        from .models import MLModel
        
        logger = logging.getLogger(__name__)
        logger.info("🔍 Validating training statuses after potential container restart...")
        
        # Find models with potentially orphaned training status
        orphaned_models = MLModel.objects.filter(status__in=['training', 'loading'])
        
        if not orphaned_models.exists():
            logger.info("✅ No potentially orphaned training models found")
            return
        
        logger.info(f"🔍 Found {orphaned_models.count()} models with training/loading status - validating...")
        
        corrected_count = 0
        for model in orphaned_models:
            if self.is_training_process_active(model):
                logger.info(f"✅ Model {model.id} ({model.name}) - training process is active")
            else:
                # Training process is not active - update status
                old_status = model.status
                model.status = 'failed'
                model.save()
                corrected_count += 1
                logger.warning(f"🔧 Model {model.id} ({model.name}) - corrected orphaned status '{old_status}' → 'failed'")
        
        if corrected_count > 0:
            logger.info(f"🎯 Corrected {corrected_count} orphaned training statuses")
        else:
            logger.info("✅ All training statuses are valid")
    
    def is_training_process_active(self, model):
        """Check if a training process is actually running for this model"""
        try:
            # Method 1: Check if there are any Python processes running train.py with this model ID
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        cmdline = proc.info['cmdline']
                        if (cmdline and 
                            any('train.py' in str(arg) for arg in cmdline) and
                            any(f'--model-id={model.id}' in str(arg) for arg in cmdline)):
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Method 2: Check MLflow run status if available
            if model.mlflow_run_id:
                try:
                    import mlflow
                    run_info = mlflow.get_run(model.mlflow_run_id)
                    # If MLflow run is still RUNNING, the process might be active
                    if run_info.info.status == 'RUNNING':
                        # Double-check with process verification
                        return False  # We already checked processes above, so this is likely orphaned
                except Exception:
                    pass
            
            return False
            
        except Exception as e:
            # If we can't determine, assume it's active to be safe
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not verify training process for model {model.id}: {e}")
            return True  # Err on the side of caution
