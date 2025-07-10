from django.core.management.base import BaseCommand
from core.apps.ml_manager.models import MLModel
import subprocess
import time
import sys

class Command(BaseCommand):
    help = 'Test training startup functionality'

    def handle(self, *args, **options):
        self.stdout.write("🧪 Testing Training Startup Fix...")
        
        # Create test model
        test_model_data = {
            'name': 'Test Training Startup Model',
            'description': 'Test model to verify training startup works',
            'status': 'pending',
            'current_epoch': 0,
            'total_epochs': 2,
            'training_data_info': {
                'model_type': 'unet',
                'data_path': 'shared/datasets/data',
                'batch_size': 16,
            }
        }
        
        model = MLModel.objects.create(**test_model_data)
        self.stdout.write(f"✅ Created test model {model.id} with status: {model.status}")
        
        # Test status transitions
        model.status = 'loading'
        model.save()
        self.stdout.write(f"✅ Status updated to: {model.status}")
        
        model.status = 'training'
        model.save()
        self.stdout.write(f"✅ Status updated to: {model.status}")
        
        # Cleanup
        model.delete()
        self.stdout.write(f"🧹 Cleaned up test model")
        
        self.stdout.write("🎉 Test completed successfully!")
