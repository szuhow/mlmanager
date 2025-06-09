#!/usr/bin/env python3
"""
Test training with real dataset after fixing the NameError
"""

import os
import sys
import django
import subprocess
import time

# Setup Django for Docker environment
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
django.setup()

from ml_manager.models import MLModel
import mlflow

def test_training_after_fix():
    """Test training with real dataset after fixing NameError"""
    print("🧪 Testing training after fixing NameError...")
    
    # Use the real dataset path
    data_path = "/app/shared/datasets/data"
    
    # Check if data exists
    images_path = os.path.join(data_path, "imgs")
    masks_path = os.path.join(data_path, "masks")
    
    if not os.path.exists(images_path):
        print(f"❌ Images directory not found at {images_path}")
        return False
        
    if not os.path.exists(masks_path):
        print(f"❌ Masks directory not found at {masks_path}")
        return False
        
    image_count = len([f for f in os.listdir(images_path) if f.endswith('.png')])
    mask_count = len([f for f in os.listdir(masks_path) if f.endswith('.png')])
    
    print(f"✅ Found {image_count} images and {mask_count} masks")
    
    # Create MLflow run
    mlflow.set_tracking_uri("http://mlflow:5000")
    experiment_name = "test-after-nameError-fix"
    
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"✅ Created MLflow run: {run_id}")
        
        # Create test model
        test_model = MLModel.objects.create(
            name="Test After NameError Fix",
            description="Test with real dataset after fixing NameError",
            status="pending"
        )
        
        print(f"✅ Created test model with ID: {test_model.id}")
        
        # Build training command with smaller parameters for testing
        command = [
            "python", "/app/shared/train.py",
            "--mode", "train",
            "--model-type", "unet",
            "--batch-size", "2",      # Small batch
            "--epochs", "1",          # Just 1 epoch
            "--learning-rate", "0.001",
            "--validation-split", "0.2", # Use less data for validation
            "--data-path", data_path,
            "--crop-size", "128",     # Standard crop size
            "--model-id", str(test_model.id),
            "--mlflow-run-id", run_id,
            "--num-workers", "1"      # Single worker to avoid multiprocessing issues
        ]
        
        print(f"🚀 Training command: {' '.join(command)}")
        
        # Start training process
        print("▶️  Starting training process...")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/app"
        )
        
        print(f"✅ Process started with PID: {process.pid}")
        
        # Monitor for 60 seconds
        start_time = time.time()
        timeout = 60
        last_status = "pending"
        
        while time.time() - start_time < timeout:
            # Check if process is still running
            poll = process.poll()
            if poll is not None:
                # Process finished
                stdout, stderr = process.communicate()
                print(f"\n🏁 Process finished with exit code: {poll}")
                print(f"📄 STDOUT (last 1000 chars):\n{stdout[-1000:]}")
                if stderr:
                    print(f"❌ STDERR (last 1000 chars):\n{stderr[-1000:]}")
                break
                
            # Check model status
            test_model.refresh_from_db()
            current_time = time.time() - start_time
            
            if test_model.status != last_status:
                print(f"🔄 Status changed: {last_status} → {test_model.status} (after {current_time:.1f}s)")
                last_status = test_model.status
            
            time.sleep(3)
        else:
            # Timeout reached
            print("⏰ Timeout reached, terminating process...")
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            print(f"📄 STDOUT (last 1000 chars):\n{stdout[-1000:] if stdout else 'No stdout'}")
            if stderr:
                print(f"❌ STDERR (last 1000 chars):\n{stderr[-1000:]}")
        
        # Final status check
        test_model.refresh_from_db()
        print(f"\n🎯 Final model status: {test_model.status}")
        
        # Show success/failure
        if test_model.status == 'training':
            print("✅ SUCCESS: Training is running!")
            success = True
        elif test_model.status == 'completed':
            print("🎉 EXCELLENT: Training completed!")
            success = True
        elif test_model.status == 'failed':
            print("❌ FAILED: Training failed")
            success = False
        else:
            print(f"⚠️  UNKNOWN: Status is '{test_model.status}'")
            success = False
        
        # Cleanup
        test_model.delete()
        print("🧹 Cleaned up test model")
        
        return success

if __name__ == "__main__":
    success = test_training_after_fix()
    if success:
        print("\n🎉 TRAINING EXECUTION IS WORKING!")
    else:
        print("\n❌ Training execution still has issues")
