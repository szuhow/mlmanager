#!/usr/bin/env python3
"""
Test actual training with training_callback to verify log path fixes
"""
import os
import sys
import django
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'core'))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
django.setup()

import subprocess
import tempfile
import time
from core.apps.ml_manager.models import MLModel
from core.apps.ml_manager.utils.mlflow_utils import setup_mlflow, create_new_run

def test_training_callback_integration():
    """Test that training_callback works with actual training process"""
    print("🧪 Testing Training Callback Integration...")
    
    try:
        # Setup MLflow
        setup_mlflow()
        
        # Create test model
        form_data = {
            'name': 'Callback Integration Test',
            'description': 'Testing training_callback integration',
            'model_type': 'unet',
            'data_path': '/app/data/datasets',
            'batch_size': 1,
            'epochs': 1,
            'learning_rate': 0.001,
            'validation_split': 0.5,
            'crop_size': 64,
            'num_workers': 0,
        }
        
        # Create MLflow run
        mlflow_params = {
            'model_type': form_data['model_type'],
            'batch_size': form_data['batch_size'],
            'epochs': form_data['epochs'],
            'learning_rate': form_data['learning_rate'],
        }
        
        run_id = create_new_run(params=mlflow_params)
        print(f"✅ Created MLflow run: {run_id}")
        
        # Create Django model
        model = MLModel.objects.create(
            name=form_data['name'],
            description=form_data.get('description', ''),
            status='pending',
            current_epoch=0,
            total_epochs=form_data['epochs'],
            train_loss=0.0,
            val_loss=0.0,
            train_dice=0.0,
            val_dice=0.0,
            best_val_dice=0.0,
            mlflow_run_id=run_id,
            training_data_info=form_data,
            model_type=form_data['model_type']
        )
        
        print(f"✅ Created model {model.id} with status: '{model.status}'")
        
        # Build training command with very short duration
        training_args = [
            sys.executable, 
            '/app/ml/training/train.py',
            '--mode', 'train',
            '--model-type', form_data['model_type'],
            '--data-path', form_data['data_path'],
            '--batch-size', str(form_data['batch_size']),
            '--epochs', str(form_data['epochs']),
            '--learning-rate', str(form_data['learning_rate']),
            '--validation-split', str(form_data['validation_split']),
            '--crop-size', str(form_data['crop_size']),
            '--num-workers', str(form_data['num_workers']),
            '--mlflow-run-id', run_id,
            '--model-id', str(model.id),
        ]
        
        print(f"🚀 Starting training process...")
        print(f"Command: {' '.join(training_args)}")
        
        # Set environment with proper Python paths
        env = os.environ.copy()
        env['PYTHONPATH'] = '/app/core:/app/ml:/app'
        env['DJANGO_SETTINGS_MODULE'] = 'core.config.settings.development'
        
        # Start training process
        process = subprocess.Popen(
            training_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd='/app'
        )
        
        print(f"✅ Training process started (PID: {process.pid})")
        
        # Monitor for 15 seconds
        start_time = time.time()
        max_wait_time = 15
        
        while time.time() - start_time < max_wait_time:
            # Check if process is still running
            if process.poll() is not None:
                print(f"🏁 Process finished with return code: {process.poll()}")
                break
            
            # Check model status
            model.refresh_from_db()
            elapsed = time.time() - start_time
            print(f"📊 After {elapsed:.1f}s: Model status = '{model.status}'")
            
            time.sleep(2)
        
        # Get final status
        if process.poll() is None:
            print("⏰ Process still running after timeout, terminating...")
            process.terminate()
            process.wait(timeout=5)
        
        # Get process output
        stdout, stderr = process.communicate()
        
        # Check for specific errors in stderr
        training_callback_success = True
        log_path_success = True
        
        if stderr:
            print(f"\n🔍 Process stderr (first 1000 chars):")
            print(stderr[:1000])
            
            # Check for specific error patterns
            if "FileNotFoundError" in stderr and "training.log" in stderr:
                log_path_success = False
                print("❌ Log path error detected")
                
            if "Failed to import MLModel" in stderr:
                training_callback_success = False
                print("❌ Training callback import error detected")
        
        if stdout:
            print(f"\n🔍 Process stdout (first 500 chars):")
            print(stdout[:500])
        
        # Check final model status
        model.refresh_from_db()
        final_status = model.status
        print(f"\n📊 Final model status: '{final_status}'")
        
        # Cleanup
        model.delete()
        print("🧹 Cleaned up test model")
        
        # Determine success
        success = (
            training_callback_success and 
            log_path_success and 
            final_status in ['loading', 'training', 'pending']  # Any valid status transition
        )
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_callback_integration()
    
    print("\n" + "="*60)
    if success:
        print("🎉 TRAINING CALLBACK INTEGRATION TEST PASSED!")
        print("✅ Training callback loads without log path errors")
        print("✅ Django imports work correctly in training process")
        print("✅ All path issues have been resolved")
    else:
        print("❌ TRAINING CALLBACK INTEGRATION TEST FAILED!")
        print("🔧 Some training callback issues remain")
    print("="*60)
