#!/usr/bin/env python3
"""
Test training execution with existing test data
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

def test_training_with_existing_data():
    """Test training with the existing test_coronary_data"""
    print("Testing training with existing test data...")
    
    # Check if test data exists
    data_path = "/app/test_coronary_data"
    if not os.path.exists(data_path):
        print(f"❌ Test data not found at {data_path}")
        return False
        
    images_path = os.path.join(data_path, "images")
    masks_path = os.path.join(data_path, "masks")
    
    if not os.path.exists(images_path) or not os.path.exists(masks_path):
        print(f"❌ Images or masks directory not found")
        return False
        
    image_count = len([f for f in os.listdir(images_path) if f.endswith('.png')])
    mask_count = len([f for f in os.listdir(masks_path) if f.endswith('.png')])
    
    print(f"✅ Found {image_count} images and {mask_count} masks")
    
    if image_count == 0:
        print("❌ No training images found")
        return False
    
    # Create MLflow run
    mlflow.set_tracking_uri("http://mlflow:5000")
    experiment_name = "test-training-execution"
    
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
            name="Test Training Execution",
            description="Test with existing data",
            status="pending"
        )
        
        print(f"✅ Created test model with ID: {test_model.id}")
        
        # Build training command
        command = [
            "python", "/app/shared/train.py",
            "--mode", "train",
            "--model-type", "unet",
            "--batch-size", "1",  # Small batch size for test
            "--epochs", "1",      # Just 1 epoch for test
            "--learning-rate", "0.001",
            "--validation-split", "0.5",
            "--data-path", data_path,
            "--crop-size", "32",  # Small crop size that should work
            "--model-id", str(test_model.id),
            "--mlflow-run-id", run_id,
            "--num-workers", "0"  # No multiprocessing to avoid issues
        ]
        
        print(f"✅ Training command: {' '.join(command)}")
        
        # Start training process
        print("🚀 Starting training process...")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/app"
        )
        
        print(f"✅ Process started with PID: {process.pid}")
        
        # Monitor for a short time
        start_time = time.time()
        timeout = 30  # 30 seconds timeout
        
        while time.time() - start_time < timeout:
            # Check if process is still running
            poll = process.poll()
            if poll is not None:
                # Process finished
                stdout, stderr = process.communicate()
                print(f"\n📊 Process finished with exit code: {poll}")
                print(f"📄 STDOUT:\n{stdout}")
                if stderr:
                    print(f"❌ STDERR:\n{stderr}")
                break
                
            # Check model status
            test_model.refresh_from_db()
            current_time = time.time() - start_time
            print(f"⏱️  After {current_time:.1f}s: Status = '{test_model.status}'")
            
            time.sleep(2)
        else:
            # Timeout reached
            print("⏰ Timeout reached, terminating process...")
            process.terminate()
            process.wait()
        
        # Final status check
        test_model.refresh_from_db()
        print(f"\n🏁 Final model status: {test_model.status}")
        
        # Cleanup
        test_model.delete()
        print("🧹 Cleaned up test model")
        
        return test_model.status in ['training', 'completed']

if __name__ == "__main__":
    success = test_training_with_existing_data()
    if success:
        print("\n✅ SUCCESS: Training execution is working!")
    else:
        print("\n❌ FAILED: Training execution has issues")
