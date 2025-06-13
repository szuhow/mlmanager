#!/usr/bin/env python3
"""
Test script to verify that the training startup fix works correctly.
This script simulates creating a model via Django form and starting training.
"""

import os
import sys
import django
import tempfile
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.testing')
django.setup()

def test_training_startup():
    """Test that training can be started and status transitions work correctly"""
    
    from core.apps.ml_manager.models import MLModel
    from ml_manager.mlflow_utils import setup_mlflow, create_new_run
    import subprocess
    
    print("🧪 Testing Training Startup Fix...")
    
    # Create test data
    test_model_data = {
        'name': 'Test Training Startup Model',
        'description': 'Test model to verify training startup works',
        'status': 'pending',
        'current_epoch': 0,
        'total_epochs': 2,  # Use minimal epochs for testing
        'train_loss': 0.0,
        'val_loss': 0.0,
        'train_dice': 0.0,
        'val_dice': 0.0,
        'best_val_dice': 0.0,
        'training_data_info': {
            'model_type': 'unet',
            'data_path': 'shared/datasets/data',
            'batch_size': 16,
            'learning_rate': 0.0001,
            'validation_split': 0.2,
            'use_random_flip': True,
            'use_random_rotate': True,
            'use_random_scale': True,
            'use_random_intensity': True,
            'crop_size': 128,
            'num_workers': 1,
        }
    }
    
    # Create model
    model = MLModel.objects.create(**test_model_data)
    print(f"✅ Created test model {model.id} with status: {model.status}")
    
    # Setup MLflow and create run
    try:
        setup_mlflow()
        
        mlflow_params = test_model_data['training_data_info'].copy()
        mlflow_params['epochs'] = test_model_data['total_epochs']
        
        run_id = create_new_run(params=mlflow_params)
        model.mlflow_run_id = run_id
        model.save()
        
        print(f"✅ Created MLflow run: {run_id}")
        
        # Build training command (similar to what the view does)
        train_args = [
            sys.executable,
            str(project_root / 'shared' / 'train.py'),
            '--mode', 'train',
            '--model-type', 'unet',
            '--batch-size', '16',
            '--epochs', '2',
            '--learning-rate', '0.0001',
            '--data-path', 'shared/datasets/data',
            '--validation-split', '0.2',
            '--crop-size', '128',
            '--num-workers', '1',
            '--mlflow-run-id', run_id,
            '--model-id', str(model.id),
            '--random-flip',
            '--random-rotate', 
            '--random-scale',
            '--random-intensity'
        ]
        
        print(f"🚀 Starting training process...")
        print(f"Command: {' '.join(train_args)}")
        
        # Start training process
        process = subprocess.Popen(
            train_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(project_root)
        )
        
        print(f"✅ Training process started with PID: {process.pid}")
        
        # Wait a moment for the process to initialize and update status
        time.sleep(3)
        
        # Check if status was updated to 'training'
        model.refresh_from_db()
        print(f"📊 Model status after 3 seconds: {model.status}")
        
        if model.status == 'training':
            print("✅ SUCCESS: Model status correctly updated to 'training'!")
            
            # Let it run for a bit more to see if it's actually training
            time.sleep(10)
            model.refresh_from_db()
            print(f"📈 Model status after 13 seconds: {model.status}")
            print(f"📈 Current epoch: {model.current_epoch}/{model.total_epochs}")
            
            # Terminate the process for testing purposes
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            print("🛑 Training process terminated for testing")
            
        else:
            print(f"❌ FAILED: Model status is still '{model.status}', expected 'training'")
            
            # Check process output for errors
            try:
                stdout, stderr = process.communicate(timeout=5)
                if stderr:
                    print(f"🔍 Training process stderr: {stderr}")
                if stdout:
                    print(f"🔍 Training process stdout: {stdout[:500]}...")
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                print(f"🔍 Process killed, stderr: {stderr}")
            
        # Cleanup
        model.delete()
        print(f"🧹 Cleaned up test model {model.id}")
        
        return model.status == 'training'
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        # Cleanup on error
        try:
            model.delete()
        except:
            pass
        return False

if __name__ == "__main__":
    success = test_training_startup()
    
    print("\n" + "="*50)
    if success:
        print("🎉 TRAINING STARTUP FIX TEST PASSED!")
        print("✅ Models can now transition from 'pending' to 'training' status")
    else:
        print("❌ TRAINING STARTUP FIX TEST FAILED!")
        print("🔧 The training process is not starting correctly")
    print("="*50)
    
    sys.exit(0 if success else 1)
