#!/usr/bin/env python3
"""
Debug the training process to see what's happening with status updates
"""

import os
import sys
import django
import subprocess
import time
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.testing')
django.setup()

from core.apps.ml_manager.models import MLModel
from ml_manager.mlflow_utils import setup_mlflow, create_new_run

def debug_training_process():
    """Debug why the training process isn't updating status"""
    
    print("🔍 Debugging Training Process Status Updates...")
    
    try:
        # Setup MLflow
        setup_mlflow()
        
        # Create test model with minimal configuration
        form_data = {
            'name': 'Debug Training Process',
            'description': 'Debugging status transitions',
            'model_type': 'unet',
            'data_path': 'shared/datasets/data',
            'batch_size': 1,  # Minimal batch size
            'epochs': 1,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'use_random_flip': False,  # Disable augmentations for simplicity
            'use_random_rotate': False,
            'use_random_scale': False,
            'use_random_intensity': False,
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
        
        # Create model
        model = MLModel.objects.create(
            name=form_data['name'],
            description=form_data['description'],
            status='pending',
            current_epoch=0,
            total_epochs=form_data['epochs'],
            mlflow_run_id=run_id,
            training_data_info=form_data
        )
        
        print(f"✅ Created model {model.id} with status: '{model.status}'")
        
        # Start training with verbose output
        training_args = [
            sys.executable, 'shared/train.py',
            '--mode', 'train',
            '--model-type', 'unet',
            '--batch-size', '1',
            '--epochs', '1',
            '--learning-rate', '0.001',
            '--data-path', 'shared/datasets/data',
            '--validation-split', '0.2',
            '--crop-size', '64',
            '--num-workers', '0',
            '--mlflow-run-id', run_id,
            '--model-id', str(model.id),
        ]
        
        print(f"🚀 Starting training process with command:")
        print(f"   {' '.join(training_args)}")
        
        # Start process and capture output in real-time
        process = subprocess.Popen(
            training_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stdout and stderr
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        print(f"✅ Process started with PID: {process.pid}")
        
        # Monitor for 15 seconds and track status changes
        start_time = time.time()
        last_status = model.status
        status_changes = []
        
        while time.time() - start_time < 15:
            # Check for process output
            if process.poll() is not None:
                # Process finished
                remaining_output = process.stdout.read()
                if remaining_output:
                    print(f"📝 Final output: {remaining_output}")
                print(f"🏁 Process finished with return code: {process.poll()}")
                break
            
            # Check model status
            model.refresh_from_db()
            if model.status != last_status:
                timestamp = time.time() - start_time
                print(f"📊 Status change at {timestamp:.1f}s: '{last_status}' → '{model.status}'")
                status_changes.append((timestamp, last_status, model.status))
                last_status = model.status
            
            # Read any available output
            try:
                # Non-blocking read
                import select
                if select.select([process.stdout], [], [], 0.1)[0]:
                    line = process.stdout.readline()
                    if line:
                        print(f"📝 Process output: {line.strip()}")
            except:
                pass  # Skip if select not available
            
            time.sleep(0.5)
        
        # Final status check
        model.refresh_from_db()
        print(f"🔚 Final model status: '{model.status}'")
        print(f"🔚 Current epoch: {model.current_epoch}/{model.total_epochs}")
        
        # Show all status changes
        print(f"\n📈 Status changes detected: {len(status_changes)}")
        for timestamp, old_status, new_status in status_changes:
            print(f"   {timestamp:.1f}s: {old_status} → {new_status}")
        
        # Cleanup
        if process.poll() is None:
            print("🛑 Terminating process...")
            process.terminate()
            process.wait()
        
        model.delete()
        print("🧹 Cleaned up test model")
        
        # Success if we saw any status changes
        return len(status_changes) > 0
        
    except Exception as e:
        print(f"❌ Debug failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_training_process()
    
    print("\n" + "="*60)
    if success:
        print("🎉 DEBUG SUCCESS: Status transitions are working!")
    else:
        print("❌ DEBUG FAILED: No status transitions detected")
        print("🔧 Need to investigate training callback system")
    print("="*60)
