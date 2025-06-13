#!/usr/bin/env python3
"""
Debug why MLflow runs are finishing early during training
"""

import os
import sys
import django
import time
import subprocess
from pathlib import Path

# Setup Django
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.testing')
django.setup()

from core.apps.ml_manager.models import MLModel
from ml_manager.mlflow_utils import setup_mlflow, create_new_run
import mlflow

def debug_mlflow_early_finish():
    """Debug why MLflow runs finish early"""
    print("🔍 Debugging MLflow Early Finish...")
    
    try:
        # Setup MLflow
        setup_mlflow()
        
        # Create minimal test run
        run_id = create_new_run(params={'test': 'debug_early_finish'})
        print(f"✅ Created MLflow run: {run_id}")
        
        # Check initial status
        run_info = mlflow.get_run(run_id)
        print(f"📊 Initial run status: {run_info.info.status}")
        
        # Create test model
        model = MLModel.objects.create(
            name='Debug Early Finish',
            description='Debug MLflow early finish',
            status='pending',
            total_epochs=1,
            mlflow_run_id=run_id,
            training_data_info={
                'model_type': 'unet',
                'batch_size': 1,
                'epochs': 1,
                'learning_rate': 0.001,
            }
        )
        
        print(f"✅ Created model {model.id}")
        
        # Start minimal training with detailed output capture
        training_args = [
            sys.executable, '/app/shared/train.py',
            '--mode', 'train',
            '--model-type', 'unet', 
            '--batch-size', '1',
            '--epochs', '1',
            '--learning-rate', '0.001',
            '--data-path', 'shared/datasets/data',
            '--validation-split', '0.5',
            '--crop-size', '32',
            '--num-workers', '0',
            '--mlflow-run-id', run_id,
            '--model-id', str(model.id),
        ]
        
        print(f"🚀 Starting training with detailed monitoring...")
        
        # Start process with real-time output capture
        process = subprocess.Popen(
            training_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stdout and stderr
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print(f"✅ Process started with PID: {process.pid}")
        
        # Monitor both run status and process output
        start_time = time.time()
        last_status = None
        output_lines = []
        
        while time.time() - start_time < 60:  # Monitor for up to 1 minute
            # Check if process finished
            if process.poll() is not None:
                # Get remaining output
                remaining = process.stdout.read()
                if remaining:
                    output_lines.extend(remaining.strip().split('\n'))
                print(f"🏁 Process finished with return code: {process.poll()}")
                break
            
            # Check run status
            try:
                run_info = mlflow.get_run(run_id)
                current_status = run_info.info.status
                elapsed = time.time() - start_time
                
                if current_status != last_status:
                    print(f"📊 Status change at {elapsed:.1f}s: {last_status} → {current_status}")
                    last_status = current_status
                    
                    # If run finished unexpectedly, capture current output
                    if current_status == 'FINISHED' and elapsed < 30:
                        print(f"⚠️  Run finished early at {elapsed:.1f}s! Capturing output...")
                        
                        # Try to get current process output
                        try:
                            import select
                            ready, _, _ = select.select([process.stdout], [], [], 0.1)
                            if ready:
                                line = process.stdout.readline()
                                if line:
                                    output_lines.append(line.strip())
                                    print(f"📝 Process output: {line.strip()}")
                        except:
                            pass
                
            except Exception as e:
                print(f"❌ Error checking run status: {e}")
                break
            
            # Try to read process output
            try:
                import select
                ready, _, _ = select.select([process.stdout], [], [], 0.1)
                if ready:
                    line = process.stdout.readline()
                    if line:
                        output_lines.append(line.strip())
                        # Print important lines
                        if any(keyword in line.lower() for keyword in ['error', 'exception', 'mlflow', 'end_run', 'failed']):
                            print(f"📝 Important: {line.strip()}")
            except:
                pass
            
            time.sleep(1)
        
        # Print summary
        print(f"\n🎯 Final Analysis:")
        try:
            final_run_info = mlflow.get_run(run_id)
            model.refresh_from_db()
            print(f"   - Final MLflow run status: {final_run_info.info.status}")
            print(f"   - Final model status: {model.status}")
            print(f"   - Process return code: {process.poll()}")
        except Exception as e:
            print(f"   - Error getting final status: {e}")
        
        # Show key output lines
        print(f"\n📋 Key Process Output Lines:")
        for i, line in enumerate(output_lines[-10:]):  # Last 10 lines
            print(f"   {i+1}: {line}")
        
        # Clean up
        try:
            if process.poll() is None:
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
        except:
            pass
        
        model.delete()
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_mlflow_early_finish()
