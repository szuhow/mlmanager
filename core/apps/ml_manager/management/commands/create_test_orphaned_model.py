from django.core.management.base import BaseCommand
from ...models import MLModel


class Command(BaseCommand):
    help = 'Create a test model with orphaned training status'

    def handle(self, *args, **options):
        import uuid
        
        # Create a test model with orphaned training status
        test_model = MLModel.objects.create(
            name='Test Orphaned Model',
            description='Test model to simulate orphaned training status after container restart',
            status='training',
            current_epoch=5,
            total_epochs=10,
            mlflow_run_id=f'test-orphaned-run-{uuid.uuid4().hex[:8]}',
            model_family='UNet-Test',
            model_type='unet'
        )
        
        self.stdout.write(f"✅ Created test model {test_model.id} with orphaned training status")
        self.stdout.write(f"   Name: {test_model.name}")
        self.stdout.write(f"   Status: {test_model.status}")
        self.stdout.write(f"   MLflow Run: {test_model.mlflow_run_id}")
        self.stdout.write(f"   Progress: {test_model.current_epoch}/{test_model.total_epochs}")
        
        self.stdout.write("\n🔧 Now run: python core/manage.py validate_training_statuses --dry-run --verbose")
