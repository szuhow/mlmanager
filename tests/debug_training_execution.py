#!/usr/bin/env python3
"""
Debug training execution to see exactly what happens when training starts
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
from ml_manager.mlflow_utils import setup_mlflow, create_new_run

def debug_training_execution():
    """Debug the exact training execution flow"""
    print("=== DEBUGGING TRAINING EXECUTION ===")
    
    # 1. Create a test model
    print("1. Creating test model...")
    test_model = MLModel.objects.create(
        name="Debug Training Model",
        description="Test training execution debug",
        status="pending",
        total_epochs=2,
        training_data_info={
            'model_type': 'unet',
            'data_path': '/tmp/test_data',
            'batch_size': 2,
            'learning_rate': 0.001,
            'validation_split': 0.5,
            'crop_size': 32,
            'num_workers': 2,
        }
    )
    print(f"   Created model ID: {test_model.id}")
    print(f"   Initial status: {test_model.status}")
    
    # 2. Setup MLflow
    print("2. Setting up MLflow...")
    try:
        setup_mlflow()
        
        mlflow_params = {
            'model_type': 'unet',
            'batch_size': 2,
            'epochs': 2,
            'learning_rate': 0.001,
            'data_path': '/tmp/test_data',
            'validation_split': 0.5,
            'crop_size': 32,
            'num_workers': 2
        }
        
        run_id = create_new_run(params=mlflow_params)
        test_model.mlflow_run_id = run_id
        test_model.save()
        print(f"   MLflow run ID: {run_id}")
        
    except Exception as e:
        print(f"   MLflow setup failed: {e}")
        test_model.delete()
        return
    
    # 3. Build training command
    print("3. Building training command...")
    train_args = [
        sys.executable, 
        '/app/shared/train.py',
        '--mode', 'train',
        '--model-type', 'unet',
        '--batch-size', '2',
        '--epochs', '2',
        '--learning-rate', '0.001',
        '--data-path', '/tmp/test_data',
        '--validation-split', '0.5',
        '--crop-size', '32',
        '--num-workers', '2',
        '--mlflow-run-id', run_id,
        '--model-id', str(test_model.id)
    ]
    
    print(f"   Command: {' '.join(train_args)}")
    
    # 4. Check if training script exists
    print("4. Checking training script...")
    if os.path.exists('/app/shared/train.py'):
        print("   ✓ Training script exists")
    else:
        print("   ✗ Training script NOT found")
        test_model.delete()
        return
    
    # 5. Try to start training process
    print("5. Starting training process...")
    try:
        process = subprocess.Popen(
            train_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd='/app'
        )
        
        print(f"   ✓ Process started with PID: {process.pid}")
        
        # 6. Monitor process for a few seconds
        print("6. Monitoring process...")
        for i in range(10):
            poll_result = process.poll()
            if poll_result is not None:
                print(f"   Process exited with code: {poll_result}")
                
                # Get output
                stdout, stderr = process.communicate()
                print("   STDOUT:")
                print(f"   {stdout}")
                print("   STDERR:")
                print(f"   {stderr}")
                break
            else:
                # Check model status
                test_model.refresh_from_db()
                print(f"   {i+1}s: Process running, model status: {test_model.status}")
                time.sleep(1)
        else:
            print("   Process still running after 10 seconds")
            process.terminate()
            
    except Exception as e:
        print(f"   ✗ Failed to start process: {e}")
    
    # 7. Final model status
    test_model.refresh_from_db()
    print(f"7. Final model status: {test_model.status}")
    
    # Cleanup
    test_model.delete()
    print("8. Cleaned up test model")

if __name__ == "__main__":
    debug_training_execution()