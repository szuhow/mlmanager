#!/usr/bin/env python3
"""
Direct test of the StartTrainingView logic to verify training startup fix
"""

import os
import sys
import django
import subprocess
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
django.setup()

from ml_manager.models import MLModel
from ml_manager.mlflow_utils import setup_mlflow, create_new_run

def test_start_training_view_logic():
    """Test the StartTrainingView logic directly"""
    
    print("🧪 Testing StartTrainingView Logic...")
    
    try:
        # Setup MLflow
        setup_mlflow()
        
        # Simulate form data from StartTrainingView
        form_data = {
            'name': 'Direct Test Model',
            'description': 'Testing StartTrainingView logic directly',
            'model_type': 'unet',
            'data_path': 'shared/datasets/data',
            'batch_size': 2,
            'epochs': 1,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'use_random_flip': True,
            'use_random_rotate': True,
            'use_random_scale': True,
            'use_random_intensity': True,
            'crop_size': 64,
            'num_workers': 0,
        }
        
        print("✅ Prepared form data")
        
        # Step 1: Create MLflow run (as done in the view)
        mlflow_params = {
            'model_type': form_data['model_type'],
            'batch_size': form_data['batch_size'],
            'epochs': form_data['epochs'],
            'learning_rate': form_data['learning_rate'],
            'data_path': form_data['data_path'],
            'validation_split': form_data['validation_split'],
        }
        
        run_id = create_new_run(params=mlflow_params)
        print(f"✅ Created MLflow run: {run_id}")
        
        # Step 2: Create model instance (as done in the view)
        model = MLModel.objects.create(
            name=form_data['name'],
            description=form_data.get('description', ''),
            status='pending',  # This should be 'pending' initially
            current_epoch=0,
            total_epochs=form_data['epochs'],
            train_loss=0.0,
            val_loss=0.0,
            train_dice=0.0,
            val_dice=0.0,
            best_val_dice=0.0,
            mlflow_run_id=run_id,
            training_data_info={
                'model_type': form_data['model_type'],
                'data_path': form_data['data_path'],
                'batch_size': form_data['batch_size'],
                'learning_rate': form_data['learning_rate'],
                'validation_split': form_data['validation_split'],
                'use_random_flip': form_data['use_random_flip'],
                'use_random_rotate': form_data['use_random_rotate'],
                'use_random_scale': form_data['use_random_scale'],
                'use_random_intensity': form_data['use_random_intensity'],
                'crop_size': form_data['crop_size'],
                'num_workers': form_data['num_workers'],
            }
        )
        
        print(f"✅ Created model {model.id} with status: '{model.status}'")
        
        # Step 3: Start training process (the NEW logic we added)
        training_args = [
            sys.executable, 'shared/train.py',
            '--mode', 'train',
            '--model-type', form_data['model_type'],
            '--batch-size', str(form_data['batch_size']),
            '--epochs', str(form_data['epochs']),
            '--learning-rate', str(form_data['learning_rate']),
            '--data-path', form_data['data_path'],
            '--validation-split', str(form_data['validation_split']),
            '--crop-size', str(form_data['crop_size']),
            '--num-workers', str(form_data['num_workers']),
            '--mlflow-run-id', run_id,
            '--model-id', str(model.id),
        ]
        
        # Add augmentation flags
        if form_data['use_random_flip']:
            training_args.append('--random-flip')
        if form_data['use_random_rotate']:
            training_args.append('--random-rotate')
        if form_data['use_random_scale']:
            training_args.append('--random-scale')
        if form_data['use_random_intensity']:
            training_args.append('--random-intensity')
        
        print(f"🚀 Starting training process...")
        print(f"Command: {' '.join(training_args)}")
        
        # Start training process in background (as in the view)
        process = subprocess.Popen(
            training_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Store process ID (this is what the view should do)
        model.training_process_id = process.pid
        model.save()
        
        print(f"✅ Started training process with PID: {process.pid}")
        print(f"✅ Model {model.id} now has training_process_id: {model.training_process_id}")
        
        # Check status transitions over time
        import time
        
        # Expected transitions: pending → loading → training
        status_checks = [
            (3, 'loading', 'Should be loading dataset'),
            (10, 'training', 'Should be training after dataset loaded'),
            (20, 'training', 'Should still be training'),
            (30, 'training', 'Should definitely be training by now')
        ]
        
        success = False
        loading_seen = False
        training_seen = False
        
        for wait_time, expected_status, description in status_checks:
            time.sleep(wait_time if wait_time == 2 else wait_time - (status_checks[status_checks.index((wait_time, expected_status, description)) - 1][0] if status_checks.index((wait_time, expected_status, description)) > 0 else 0))
            model.refresh_from_db()
            print(f"📊 After {wait_time}s: Status = '{model.status}' ({description})")
            
            if model.status == 'loading':
                loading_seen = True
                print("  ✅ Loading status detected")
            elif model.status == 'training':
                training_seen = True
                print("  ✅ Training status detected")
            
            # Check if process is still running
            if process.poll() is not None:
                print(f"  ⚠️  Process exited with code: {process.poll()}")
                stdout, stderr = process.communicate()
                if stderr:
                    print(f"  🔍 Process stderr: {stderr[:500]}")
                if stdout:
                    print(f"  🔍 Process stdout: {stdout[:500]}")
                break
        
        # Success if we saw the progression from pending to loading to training
        success = loading_seen and training_seen
        
        if success:
            print("🎉 SUCCESS! Training startup fix is working:")
            print(f"  ✅ Process started (PID: {process.pid})")
            print(f"  ✅ Status transitioned: pending → loading → training")
            print(f"  ✅ Final status: '{model.status}'")
        else:
            print("❌ FAILED! Issues detected:")
            if process.poll() is not None:
                print(f"  ❌ Process exited early (return code: {process.poll()})")
                stdout, stderr = process.communicate()
                if stderr:
                    print(f"  🔍 Process stderr: {stderr[:300]}")
            if not loading_seen:
                print(f"  ❌ Never saw 'loading' status transition")
            if not training_seen:
                print(f"  ❌ Never saw 'training' status transition")
            print(f"  📊 Final status: '{model.status}'")
        
        # Cleanup
        if process.poll() is None:
            process.terminate()
            process.wait()
        
        model.delete()
        print("🧹 Cleaned up test model")
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_start_training_view_logic()
    
    print("\n" + "="*60)
    if success:
        print("🎉 START TRAINING VIEW LOGIC TEST PASSED!")
        print("✅ The training startup fix is working correctly")
    else:
        print("❌ START TRAINING VIEW LOGIC TEST FAILED!")
        print("🔧 The training startup needs further investigation")
    print("="*60)
