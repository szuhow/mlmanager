from django.core.management.base import BaseCommand
from ml_manager.models import MLModel
from shared.utils.training_callback import TrainingCallback

class Command(BaseCommand):
    help = 'Test the callback system'

    def handle(self, *args, **options):
        self.stdout.write("Testing callback system in Docker...")
        
        # Create a test model
        test_model = MLModel.objects.create(
            name="Callback Test Model Docker",
            description="Test callback functionality in Docker",
            status="pending"
        )
        
        self.stdout.write(f"Created test model with ID: {test_model.id}")
        self.stdout.write(f"Initial status: {test_model.status}")
        
        # Create callback
        callback = TrainingCallback(test_model.id, "test-run-callback-docker")
        
        # Test on_training_start
        self.stdout.write("Calling on_training_start()...")
        callback.on_training_start()
        
        # Refresh and check status
        test_model.refresh_from_db()
        self.stdout.write(f"Status after on_training_start(): {test_model.status}")
        
        # Test on_dataset_loaded
        self.stdout.write("Calling on_dataset_loaded()...")
        callback.on_dataset_loaded()
        
        # Refresh and check status
        test_model.refresh_from_db()
        self.stdout.write(f"Status after on_dataset_loaded(): {test_model.status}")
        
        # Test some epoch progress
        self.stdout.write("Testing epoch progress...")
        callback.on_epoch_end(0, 0.8, 0.75)
        
        # Refresh and check
        test_model.refresh_from_db()
        self.stdout.write(f"Status after epoch update: {test_model.status}")
        self.stdout.write(f"Training progress: {test_model.training_progress}%")
        
        # Cleanup
        test_model.delete()
        self.stdout.write("Test completed and cleaned up")
