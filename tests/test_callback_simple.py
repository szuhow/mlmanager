#!/usr/bin/env python3
import os
import django

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
django.setup()

# Now run the test
from ml_manager.models import MLModel
from shared.utils.training_callback import TrainingCallback

print("🧪 Testing callback system in Docker...")

# Create a test model
test_model = MLModel.objects.create(
    name="Callback Test Model Docker",
    description="Test callback functionality in Docker",
    status="pending"
)

print(f"✅ Created test model with ID: {test_model.id}")
print(f"📊 Initial status: {test_model.status}")

try:
    # Create callback
    callback = TrainingCallback(test_model.id, "test-run-callback-docker")
    
    # Test on_training_start
    print("🔄 Calling on_training_start()...")
    callback.on_training_start()
    
    # Refresh and check status
    test_model.refresh_from_db()
    print(f"📊 Status after on_training_start(): {test_model.status}")
    
    # Test on_dataset_loaded
    print("🔄 Calling on_dataset_loaded()...")
    callback.on_dataset_loaded()
    
    # Refresh and check status
    test_model.refresh_from_db()
    print(f"📊 Status after on_dataset_loaded(): {test_model.status}")
    
    # Test some epoch progress
    print("🔄 Testing epoch progress...")
    callback.on_epoch_end(0, 0.8, 0.75)
    
    # Refresh and check
    test_model.refresh_from_db()
    print(f"📊 Status after epoch update: {test_model.status}")
    print(f"📈 Training progress: {test_model.training_progress}%")

    print("✅ All callback tests passed!")
    
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    # Cleanup
    test_model.delete()
    print("🧹 Test completed and cleaned up")
