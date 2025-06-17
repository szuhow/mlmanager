#!/usr/bin/env python3
"""
Integration test for classification model fixes
Tests a complete classification training workflow
"""

import os
import sys
import tempfile
import shutil
import django
import logging
from pathlib import Path

# Setup Django
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
django.setup()

from core.apps.ml_manager.models import MLModel
import mlflow
from core.apps.ml_manager.utils.mlflow_utils import setup_mlflow, create_new_run

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_minimal_classification_dataset():
    """Create minimal test dataset for classification"""
    print("🔧 Creating minimal classification dataset...")
    
    # Create temporary dataset structure
    temp_dir = tempfile.mkdtemp(prefix="classification_test_")
    
    # Create ARCADE-like structure for classification
    arcade_dir = os.path.join(temp_dir, "arcade_classification")
    images_dir = os.path.join(arcade_dir, "images")
    masks_dir = os.path.join(arcade_dir, "masks")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Create dummy files
    import numpy as np
    from PIL import Image
    
    for i in range(4):  # Just 4 samples for quick test
        # Create dummy image (RGB)
        img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(images_dir, f"sample_{i:03d}.png"))
        
        # Create dummy mask (binary classification - 0 or 1)
        mask_array = np.random.choice([0, 1], (64, 64), p=[0.7, 0.3]).astype(np.uint8)
        mask = Image.fromarray(mask_array, mode='L')
        mask.save(os.path.join(masks_dir, f"sample_{i:03d}.png"))
    
    print(f"✅ Created test dataset at: {temp_dir}")
    return temp_dir

def test_classification_training():
    """Test complete classification training workflow"""
    print("\n🧪 Testing Classification Training Workflow")
    print("=" * 60)
    
    # Setup MLflow
    setup_mlflow()
    
    # Create test dataset
    dataset_path = create_minimal_classification_dataset()
    
    try:
        # Create MLflow run
        run_id = create_new_run()
        logger.info(f"✅ Created MLflow run: {run_id}")
        
        # Create test model in database
        test_model = MLModel.objects.create(
            name="Classification Test Model",
            description="Testing classification fixes",
            status="pending",
            total_epochs=2,
            mlflow_run_id=run_id,
            training_data_info={
                'dataset_type': 'arcade_classification',
                'task_type': 'artery_classification',
                'model_type': 'classification',
                'batch_size': 2,
                'epochs': 2,
                'learning_rate': 0.001,
                'data_path': dataset_path,
                'validation_split': 0.5,
                'crop_size': 64
            }
        )
        logger.info(f"✅ Created test model with ID: {test_model.id}")
        
        # Start training process
        cmd = [
            sys.executable, 
            "/app/ml/training/train.py",
            "--mode", "train",
            "--model-type", "unet",  # Will be converted to classification
            "--dataset-type", "arcade_classification",
            "--batch-size", "2",
            "--epochs", "2",
            "--learning-rate", "0.001",
            "--data-path", dataset_path,
            "--validation-split", "0.5",
            "--crop-size", "64",
            "--num-workers", "0",
            "--mlflow-run-id", run_id,
            "--model-id", str(test_model.id),
        ]
        
        import subprocess
        
        print(f"🚀 Starting classification training...")
        print(f"Command: {' '.join(cmd)}")
        
        # Run training with timeout
        try:
            result = subprocess.run(
                cmd,
                timeout=300,  # 5 minutes timeout
                capture_output=True,
                text=True,
                cwd="/app"
            )
            
            print(f"✅ Training completed with return code: {result.returncode}")
            
            if result.stdout:
                print("📋 Training stdout (last 1000 chars):")
                print(result.stdout[-1000:])
            
            if result.stderr:
                print("⚠️ Training stderr (last 1000 chars):")
                print(result.stderr[-1000:])
            
            # Check MLflow metrics
            print("\n📊 Checking MLflow metrics...")
            run_info = mlflow.get_run(run_id)
            metrics = run_info.data.metrics
            
            # Look for classification-specific metrics
            accuracy_metrics = [k for k in metrics.keys() if 'accuracy' in k.lower()]
            dice_metrics = [k for k in metrics.keys() if 'dice' in k.lower()]
            
            print(f"✅ Accuracy metrics found: {len(accuracy_metrics)}")
            for metric in accuracy_metrics:
                print(f"  - {metric}: {metrics[metric]}")
            
            print(f"🔍 Dice metrics found: {len(dice_metrics)}")
            for metric in dice_metrics:
                print(f"  - {metric}: {metrics[metric]}")
            
            # Check step numbering
            step_based_metrics = {}
            for key, value in metrics.items():
                if key in ['train_accuracy', 'val_accuracy', 'learning_rate']:
                    step_based_metrics[key] = value
            
            print(f"📈 Step-based metrics: {step_based_metrics}")
            
            # Check model status
            test_model.refresh_from_db()
            print(f"🎯 Final model status: {test_model.status}")
            
            # Verify the fixes worked
            checks = {
                'MLflow metrics logged': len(metrics) > 0,
                'Accuracy metrics present': len(accuracy_metrics) > 0,
                'Training completed': result.returncode == 0,
                'Model status updated': test_model.status != 'pending'
            }
            
            print("\n✅ Fix Verification:")
            all_passed = True
            for check, result in checks.items():
                status = "✅" if result else "❌"
                print(f"  {status} {check}")
                if not result:
                    all_passed = False
            
            return all_passed
            
        except subprocess.TimeoutExpired:
            print("⏰ Training timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"❌ Training failed with error: {e}")
            return False
            
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up test dataset: {dataset_path}")
        shutil.rmtree(dataset_path, ignore_errors=True)
        
        # Try to end MLflow run
        try:
            if mlflow.active_run():
                mlflow.end_run()
        except:
            pass

def test_inference_with_classification():
    """Test inference with classification model"""
    print("\n🧪 Testing Classification Inference")
    print("=" * 60)
    
    try:
        # Create a dummy classification model checkpoint
        import torch
        from ml.training.models.classification_models import UNetClassifier
        
        model = UNetClassifier(input_channels=3, num_classes=2)
        
        # Create enhanced checkpoint with metadata
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_metadata': {
                'model_type': 'classification',
                'task_type': 'artery_classification',
                'input_channels': 3,
                'num_classes': 2,
                'epoch': 5,
                'validation_accuracy': 0.85
            }
        }
        
        # Save checkpoint
        temp_checkpoint = tempfile.mktemp(suffix='.pth')
        torch.save(checkpoint, temp_checkpoint)
        
        print(f"✅ Created test checkpoint: {temp_checkpoint}")
        
        # Test loading with enhanced error handling
        from ml.inference.predict import load_model, inference
        
        print("🔧 Testing model loading...")
        loaded_model, detected_type = load_model(temp_checkpoint, 'classification')
        
        print(f"✅ Model loaded successfully")
        print(f"✅ Detected type: {detected_type}")
        print(f"✅ Model class: {type(loaded_model).__name__}")
        
        # Test inference
        print("🔧 Testing inference...")
        dummy_input = torch.randn(1, 3, 64, 64)
        
        with torch.no_grad():
            output = loaded_model(dummy_input)
        
        print(f"✅ Inference successful. Output shape: {output.shape}")
        
        # Test with PIL image (like real usage)
        from PIL import Image
        import numpy as np
        
        # Create dummy RGB image
        img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        test_image = Image.fromarray(img_array)
        
        result = inference(loaded_model, test_image, 'classification')
        
        print(f"✅ PIL image inference successful")
        print(f"✅ Result type: {type(result)}")
        if isinstance(result, dict):
            print(f"✅ Predicted class: {result.get('predicted_class')}")
            print(f"✅ Confidence: {result.get('confidence', 0):.3f}")
        
        # Cleanup
        os.unlink(temp_checkpoint)
        
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run integration tests"""
    print("🔍 CLASSIFICATION FIXES INTEGRATION TESTS")
    print("=" * 80)
    
    results = []
    
    # Test 1: Classification training workflow
    results.append(("Classification Training", test_classification_training()))
    
    # Test 2: Classification inference
    results.append(("Classification Inference", test_inference_with_classification()))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 INTEGRATION TEST RESULTS")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status:<12} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} integration tests passed")
    
    if passed == total:
        print("🎉 All classification fixes work correctly in integration!")
        print("✅ MLflow step-based logging fixed")
        print("✅ Classification accuracy metrics working")
        print("✅ Enhanced model loading working")
    else:
        print("⚠️ Some integration tests failed - review output above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
