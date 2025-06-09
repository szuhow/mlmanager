#!/usr/bin/env python3
"""
Test training execution with real coronary dataset from shared/datasets/data/
"""

import os
import sys
import django
import subprocess
import time
import logging

# Setup Django for Docker environment
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
django.setup()

from ml_manager.models import MLModel
import mlflow

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_training_with_real_dataset():
    """Test training with the real coronary dataset"""
    print("🚀 Testing training with REAL coronary dataset...")
    
    # Check if real dataset exists
    data_path = "/app/shared/datasets/data"
    
    if not os.path.exists(data_path):
        print(f"❌ Real dataset not found at {data_path}")
        return False
        
    images_path = os.path.join(data_path, "imgs")
    masks_path = os.path.join(data_path, "masks")
    
    if not os.path.exists(images_path) or not os.path.exists(masks_path):
        print(f"❌ Images or masks directory not found")
        return False
        
    # Count images and masks
    image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))]
    mask_files = [f for f in os.listdir(masks_path) if f.endswith(('.jpg', '.png'))]
    
    print(f"✅ Found {len(image_files)} images and {len(mask_files)} masks")
    
    if len(image_files) == 0:
        print("❌ No training images found")
        return False
    
    # Create MLflow run
    mlflow.set_tracking_uri("http://mlflow:5000")
    experiment_name = "real-dataset-training-test"
    
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created new experiment: {experiment_name}")
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"✅ Created MLflow run: {run_id}")
        logger.info(f"MLflow run created: {run_id}")
        
        # Create test model
        test_model = MLModel.objects.create(
            name="Real Dataset Training Test",
            description=f"Test with {len(image_files)} real images",
            status="pending"
        )
        
        print(f"✅ Created test model with ID: {test_model.id}")
        logger.info(f"Created model {test_model.id}")
        
        # Build training command with appropriate parameters for real data
        command = [
            "python", "/app/shared/train.py",
            "--mode", "train",
            "--model-type", "unet",
            "--batch-size", "4",     # Reasonable batch size for real training
            "--epochs", "2",         # Limited epochs for testing
            "--learning-rate", "0.001",
            "--validation-split", "0.2",  # 20% validation split
            "--data-path", data_path,
            "--crop-size", "256",    # Appropriate crop size for real images
            "--model-id", str(test_model.id),
            "--mlflow-run-id", run_id,
            "--num-workers", "2"     # Some parallel processing
        ]
        
        command_str = ' '.join(command)
        print(f"🔥 Training command: {command_str}")
        logger.info(f"Executing command: {command_str}")
        
        # Start training process
        print("🚀 Starting training process...")
        start_time = time.time()
        
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd="/app"
            )
            
            print(f"✅ Process started with PID: {process.pid}")
            logger.info(f"Training process started with PID: {process.pid}")
            
            # Monitor the process
            timeout = 120  # 2 minutes timeout for real data
            status_history = []
            
            while time.time() - start_time < timeout:
                # Check if process is still running
                poll = process.poll()
                if poll is not None:
                    # Process finished
                    stdout, stderr = process.communicate()
                    elapsed = time.time() - start_time
                    print(f"\n📊 Process finished after {elapsed:.1f}s with exit code: {poll}")
                    
                    if stdout:
                        print(f"📄 STDOUT (last 20 lines):")
                        lines = stdout.strip().split('\n')
                        for line in lines[-20:]:
                            print(f"    {line}")
                    
                    if stderr:
                        print(f"❌ STDERR:")
                        print(f"    {stderr}")
                    
                    break
                    
                # Check model status
                test_model.refresh_from_db()
                current_time = time.time() - start_time
                current_status = test_model.status
                
                # Log status changes
                if not status_history or status_history[-1][1] != current_status:
                    status_change = (current_time, current_status)
                    status_history.append(status_change)
                    print(f"⏱️  After {current_time:.1f}s: Status = '{current_status}'")
                    logger.info(f"Status change after {current_time:.1f}s: {current_status}")
                
                time.sleep(3)  # Check every 3 seconds
            else:
                # Timeout reached
                elapsed = time.time() - start_time
                print(f"⏰ Timeout reached after {elapsed:.1f}s, terminating process...")
                process.terminate()
                process.wait()
                stdout, stderr = process.communicate()
                
                if stdout:
                    print(f"📄 STDOUT (last 10 lines):")
                    lines = stdout.strip().split('\n')
                    for line in lines[-10:]:
                        print(f"    {line}")
        
        except Exception as e:
            print(f"❌ Error starting process: {e}")
            logger.error(f"Process error: {e}")
            return False
        
        # Final status check
        test_model.refresh_from_db()
        final_status = test_model.status
        elapsed_total = time.time() - start_time
        
        print(f"\n🏁 Final Results:")
        print(f"   Total time: {elapsed_total:.1f}s")
        print(f"   Final status: {final_status}")
        print(f"   Model ID: {test_model.id}")
        
        # Print status history
        if status_history:
            print(f"\n📈 Status History:")
            for timestamp, status in status_history:
                print(f"     {timestamp:6.1f}s: {status}")
        
        # Determine success
        success = final_status in ['training', 'completed']
        
        if success:
            print(f"\n✅ SUCCESS: Training with real data working!")
            if final_status == 'training':
                print("   Status reached 'training' - process is working correctly")
            elif final_status == 'completed':
                print("   Training completed successfully!")
        else:
            print(f"\n❌ ISSUE: Final status is '{final_status}'")
            if final_status == 'pending':
                print("   Training never started - check StartTrainingView")
            elif final_status == 'failed':
                print("   Training failed - check error logs")
        
        # Cleanup
        test_model.delete()
        print("🧹 Cleaned up test model")
        logger.info("Test completed and model cleaned up")
        
        return success

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 REAL DATASET TRAINING TEST")
    print("=" * 60)
    
    success = test_training_with_real_dataset()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 RESULT: TRAINING SYSTEM IS WORKING WITH REAL DATA!")
    else:
        print("🔧 RESULT: TRAINING SYSTEM NEEDS DEBUGGING")
    print("=" * 60)
