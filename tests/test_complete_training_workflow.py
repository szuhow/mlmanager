#!/usr/bin/env python3
"""
Test the actual training startup workflow end-to-end
"""

import os
import sys
import django
import time
import subprocess
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
django.setup()

from ml_manager.models import MLModel
from ml_manager.mlflow_utils import setup_mlflow, create_new_run

def test_actual_training_startup():
    """Test the complete training startup workflow"""
    
    print("🧪 Testing Complete Training Startup Workflow...")
    
    try:
        # Setup MLflow
        setup_mlflow()
        
        # Create test model
        model = MLModel.objects.create(
            name='End-to-End Training Test',
            description='Testing complete training workflow',
            status='pending',
            total_epochs=1,  # Minimal for testing
            training_data_info={
                'model_type': 'unet',
                'data_path': 'shared/datasets/data',
                'batch_size': 2,
                'learning_rate': 0.001,
                'validation_split': 0.2,
                'use_random_flip': True,
                'use_random_rotate': True,
                'use_random_scale': True,
                'use_random_intensity': True,
                'crop_size': 64,
                'num_workers': 0,
            }
        )
        
        print(f"✅ Created model {model.id} with status: {model.status}")
        
        # Create MLflow run
        mlflow_params = model.training_data_info.copy()
        mlflow_params['epochs'] = model.total_epochs
        run_id = create_new_run(params=mlflow_params)
        
        model.mlflow_run_id = run_id
        model.save()
        print(f"✅ Created MLflow run: {run_id}")
        
        # Build training command (similar to the StartTrainingView)
        training_args = [
            sys.executable, 'shared/train.py',
            '--mode', 'train',
            '--model-type', 'unet',
            '--batch-size', '2',
            '--epochs', '1',
            '--learning-rate', '0.001', 
            '--data-path', 'shared/datasets/data',
            '--validation-split', '0.2',
            '--crop-size', '64',
            '--num-workers', '0',
            '--mlflow-run-id', run_id,
            '--model-id', str(model.id),
            '--random-flip',
            '--random-rotate',
            '--random-scale', 
            '--random-intensity'
        ]
        
        print(f"🚀 Starting training process...")
        print(f"Command: {' '.join(training_args)}")
        
        # Start training process
        process = subprocess.Popen(
            training_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"✅ Training process started with PID: {process.pid}")
        
        # Check status transitions
        status_checks = [
            (2, 'loading', 'Model should be loading dataset'),
            (5, 'training', 'Model should be training'),
        ]
        
        for wait_time, expected_status, description in status_checks:
            time.sleep(wait_time)
            model.refresh_from_db()
            print(f"📊 After {wait_time}s: Status = '{model.status}' ({description})")
            
            if model.status == expected_status:
                print(f"✅ SUCCESS: {description}")
                if expected_status == 'training':
                    # If we reached training state, we can consider this a success
                    process.terminate()
                    process.wait()
                    model.delete()
                    return True
            else:
                print(f"⚠️  Expected '{expected_status}', got '{model.status}'")
        
        # Check if process is still running
        if process.poll() is None:
            print("🔍 Training process still running, waiting a bit more...")
            time.sleep(5)
            model.refresh_from_db()
            print(f"📊 Final status check: {model.status}")
            
            # Terminate process
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=5)
                if stderr:
                    print(f"🔍 Process stderr: {stderr[:500]}")
                if stdout:
                    print(f"🔍 Process stdout: {stdout[:500]}")
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        
        # Cleanup
        model.delete()
        
        return model.status in ['loading', 'training']
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_actual_training_startup()
    
    print("\n" + "="*60)
    if success:
        print("🎉 TRAINING STARTUP END-TO-END TEST PASSED!")
        print("✅ Training process can start and update model status")
    else:
        print("❌ TRAINING STARTUP END-TO-END TEST FAILED!")
        print("🔧 Check the training process startup and status updates")
    print("="*60)
    
    sys.exit(0 if success else 1)
