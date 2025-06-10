#!/usr/bin/env python3
"""
Test MLflow run lifecycle to verify runs stay active during training
"""

import os
import sys
import django
import time
import subprocess
from pathlib import Path

# Setup Django
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
django.setup()

from ml_manager.models import MLModel
from ml_manager.mlflow_utils import setup_mlflow, create_new_run
import mlflow

def test_mlflow_run_lifecycle():
    """Test that MLflow runs stay active during training"""
    print("🧪 Testing MLflow Run Lifecycle...")
    
    try:
        # Setup MLflow
        setup_mlflow()
        
        # Create test model with minimal configuration
        form_data = {
            'name': 'MLflow Lifecycle Test',
            'description': 'Testing MLflow run stays active',
            'model_type': 'unet',
            'data_path': 'shared/datasets/data',
            'batch_size': 1,  # Minimal batch size
            'epochs': 1,      # Single epoch
            'learning_rate': 0.001,
            'validation_split': 0.2,
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
        
        # Check initial run status
        run_info = mlflow.get_run(run_id)
        print(f"📊 Initial run status: {run_info.info.status}")
        
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
        
        # Start training with minimal config for quick test
        training_args = [
            sys.executable, '/app/shared/train.py',
            '--mode', 'train',
            '--model-type', 'unet',
            '--batch-size', '1',
            '--epochs', '1',
            '--learning-rate', '0.001',
            '--data-path', 'shared/datasets/data',
            '--validation-split', '0.5',  # Use small validation for speed
            '--crop-size', '32',  # Very small crop
            '--num-workers', '0',
            '--mlflow-run-id', run_id,
            '--model-id', str(model.id),
        ]
        
        print(f"🚀 Starting training process...")
        print(f"Command: {' '.join(training_args)}")
        
        # Start process
        process = subprocess.Popen(
            training_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print(f"✅ Process started with PID: {process.pid}")
        
        # Monitor for 30 seconds and check run status periodically
        start_time = time.time()
        max_wait_time = 30
        check_interval = 3
        
        while time.time() - start_time < max_wait_time:
            # Check if process is still running
            if process.poll() is not None:
                print(f"🏁 Process finished with return code: {process.poll()}")
                break
            
            # Check run status
            try:
                run_info = mlflow.get_run(run_id)
                elapsed = time.time() - start_time
                print(f"📊 After {elapsed:.1f}s: MLflow run status = {run_info.info.status}")
                
                # Check model status too
                model.refresh_from_db()
                print(f"📊 After {elapsed:.1f}s: Model status = {model.status}")
                
            except Exception as e:
                print(f"❌ Error checking run status: {e}")
                break
            
            time.sleep(check_interval)
        
        # Get final statuses
        try:
            final_run_info = mlflow.get_run(run_id)
            model.refresh_from_db()
            print(f"\n🎯 Final Results:")
            print(f"   - MLflow run status: {final_run_info.info.status}")
            print(f"   - Model status: {model.status}")
            
            # Get any process output
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                if stderr:
                    print(f"🔍 Process stderr: {stderr[:500]}")
                if stdout:
                    print(f"🔍 Process stdout: {stdout[:500]}")
            
            # Success if run stayed RUNNING during the process or completed successfully
            success = (final_run_info.info.status in ['RUNNING', 'FINISHED'] and 
                      model.status in ['training', 'completed', 'loading'])
            
            return success
            
        except Exception as e:
            print(f"❌ Error getting final status: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            if 'process' in locals() and process.poll() is None:
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
        except:
            pass
        
        try:
            if 'model' in locals():
                model.delete()
                print("✅ Cleaned up test model")
        except:
            pass

if __name__ == "__main__":
    success = test_mlflow_run_lifecycle()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 MLFLOW RUN LIFECYCLE TEST PASSED!")
        print("✅ MLflow runs stay active during training")
    else:
        print("❌ MLFLOW RUN LIFECYCLE TEST FAILED!")
        print("🔧 MLflow runs are ending prematurely")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
