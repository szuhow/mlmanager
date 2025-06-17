# Dataset Manager App Configuration

from django.apps import AppConfig

class DatasetManagerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core.apps.dataset_manager'
    verbose_name = 'Dataset Manager'
    
    def ready(self):
        # Import signals
        import core.apps.dataset_manager.signals
