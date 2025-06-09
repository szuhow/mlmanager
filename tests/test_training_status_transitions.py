#!/usr/bin/env python3
"""
Test script to verify training status transitions work properly
This test handles the longer dataset loading times
"""

import os
import sys
import django
import time
import subprocess
import logging
import mlflow
from pathlib import Path

# Setup Django
sys.path.append('/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
django.setup()

from ml_manager.models import MLModel
from shared.utils.training_callback import TrainingCallback

def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_test_dataset():
    """Create a minimal test dataset for faster loading"""
    test_data_dir = Path("/tmp/test_coronary_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Create minimal directory structure
    imgs_dir = test_data_dir / "imgs"
    masks_dir = test_data_dir / "masks"
    imgs_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    
    # Create real test images (minimal size) instead of empty files
    import numpy as np
    from PIL import Image
    
    for i in range(3):
        # Create a small 32x32 grayscale image
        img_data = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        mask_data = np.random.randint(0, 2, (32, 32), dtype=np.uint8) * 255  # Binary mask
        
        # Save as PNG files
        img = Image.fromarray(img_data, mode='L')
        mask = Image.fromarray(mask_data, mode='L')
        
        img.save(imgs_dir / f"test_img_{i:03d}.png")
        mask.save(masks_dir / f"test_mask_{i:03d}.png")
    
    return str(test_data_dir)

def test_training_status_transitions():
    """Test the complete training status transition flow"""
    logger = setup_logging()
    logger.info("Starting training status transition test")
    
    try:
        # Create test dataset
        test_data_path = create_test_dataset()
        logger.info(f"Created test dataset at: {test_data_path}")
        
        # Setup MLflow and create a real run
        from ml_manager.mlflow_utils import setup_mlflow, create_new_run
        setup_mlflow()
        
        # Create MLflow run with minimal parameters
        mlflow_params = {
            'model_type': 'unet',
            'batch_size': 2,
            'epochs': 2,
            'learning_rate': 0.001,
            'data_path': test_data_path,
            'validation_split': 0.5
        }
        
        run_id = create_new_run(params=mlflow_params)
        logger.info(f"Created MLflow run: {run_id}")
        
        # Create a test model (using correct field names)
        test_model = MLModel.objects.create(
            name="Status Transition Test Model",
            description="Test model for status transitions",
            status="pending",
            total_epochs=2,  # Minimal epochs for testing
            mlflow_run_id=run_id,  # Use the real MLflow run ID
            training_data_info={
                'model_type': 'unet',
                'batch_size': 2,  # Small batch size for faster processing
                'epochs': 2,
                'learning_rate': 0.001,
                'data_path': test_data_path,
                'validation_split': 0.5,
                'crop_size': 32  # Match our test image size
            }
        )
        
        logger.info(f"Created test model with ID: {test_model.id}")
        logger.info(f"Initial status: {test_model.status}")
        
        # Simulate the training process startup
        training_info = test_model.training_data_info
        cmd = [
            sys.executable, 
            "/app/shared/train.py",  # Use container path
            "--mode", "train",
            "--model-type", training_info['model_type'],
            "--batch-size", str(training_info['batch_size']),
            "--epochs", str(training_info['epochs']),
            "--learning-rate", str(training_info['learning_rate']),  
            "--validation-split", str(training_info['validation_split']),
            "--data-path", training_info['data_path'],
            "--crop-size", str(training_info['crop_size']),  # Add crop size parameter
            "--model-id", str(test_model.id),
            "--mlflow-run-id", run_id
        ]
        
        logger.info(f"Starting training process with command: {' '.join(cmd)}")
        
        # Start the training process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Log the process ID (we don't store it in the model since that field doesn't exist)
        process_pid = process.pid
        logger.info(f"Training process started with PID: {process_pid}")
        
        # Wait for status transitions with proper timing
        status_history = []
        max_wait_time = 60  # Wait up to 60 seconds total
        check_interval = 2  # Check every 2 seconds
        
        start_time = time.time()
        expected_transitions = ["pending", "loading", "training", "completed"]
        current_transition_index = 0
        
        while time.time() - start_time < max_wait_time:
            # Refresh the model from database
            test_model.refresh_from_db()
            current_status = test_model.status
            
            # Log status if it changed
            if not status_history or current_status != status_history[-1][1]:
                elapsed = time.time() - start_time
                status_history.append((elapsed, current_status))
                logger.info(f"Status after {elapsed:.1f}s: {current_status}")
                
                # Check if we've progressed to the next expected transition
                if (current_transition_index < len(expected_transitions) and 
                    current_status == expected_transitions[current_transition_index]):
                    current_transition_index += 1
                    logger.info(f"✓ Successfully transitioned to: {current_status}")
                    
                    # If we've reached 'training' status, we can consider the test successful
                    if current_status == "training":
                        logger.info("SUCCESS: Reached 'training' status!")
                        # Continue waiting to see if it completes
                    elif current_status == "completed":
                        logger.info("COMPLETE SUCCESS: Training completed!")
                        break
            
            time.sleep(check_interval)
        
        # Check if the process is still running
        poll_result = process.poll()
        if poll_result is None:
            logger.info(f"Training process is still running (PID: {process.pid})")
        else:
            logger.warning(f"Training process exited with code: {poll_result}")
            # Get any error output
            stdout, stderr = process.communicate()
            if stderr:
                logger.error(f"Training process stderr: {stderr}")
            if stdout:
                logger.info(f"Training process stdout: {stdout}")
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info("TRAINING STATUS TRANSITION TEST RESULTS")
        logger.info("="*50)
        logger.info(f"Model ID: {test_model.id}")
        logger.info(f"Process PID: {process_pid}")
        logger.info(f"Final Status: {test_model.status}")
        logger.info("\nStatus History:")
        for elapsed, status in status_history:
            logger.info(f"  {elapsed:6.1f}s: {status}")
        
        # Determine test result
        if test_model.status in ["training", "completed"]:
            logger.info(f"\n✓ TEST PASSED: Successfully transitioned to '{test_model.status}' status")
            return True
        elif test_model.status == "loading":
            logger.info("\n⚠ TEST PARTIAL: Reached 'loading' but not 'training' (dataset loading may take longer)")
            return False
        else:
            logger.info(f"\n✗ TEST FAILED: Stuck in '{test_model.status}' status")
            return False
            
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        # Cleanup: terminate the training process if still running
        try:
            if 'process' in locals() and process.poll() is None:
                logger.info(f"Terminating training process {process.pid}")
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
        except:
            pass
        
        # Cleanup test model
        try:
            if 'test_model' in locals():
                test_model.delete()
                logger.info("Cleaned up test model")
        except:
            pass

if __name__ == "__main__":
    success = test_training_status_transitions()
    sys.exit(0 if success else 1)
