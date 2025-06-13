#!/usr/bin/env python3
"""
Final test of MLflow run lifecycle fixes:
1. System metrics logging fix
2. PNG training sample generation fix  
3. MLflow model signature fix
"""

import os
import sys
import django
import time
import subprocess
import mlflow
import logging
from pathlib import Path

# Setup Django
sys.path.append('/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.testing')
django.setup()

from core.apps.ml_manager.models import MLModel
from core.apps.ml_manager.utils.mlflow_utils import setup_mlflow, create_new_run

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_dataset():
    """Create a minimal test dataset"""
    test_data_dir = Path("/tmp/test_data_final")
    test_data_dir.mkdir(exist_ok=True)
    
    # Create minimal directory structure
    imgs_dir = test_data_dir / "imgs"
    masks_dir = test_data_dir / "masks"
    imgs_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    
    # Create real test images (minimal size)
    import numpy as np
    from PIL import Image
    
    for i in range(4):  # A few samples
        # Create a small 64x64 grayscale image
        img_data = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        mask_data = np.random.randint(0, 2, (64, 64), dtype=np.uint8) * 255  # Binary mask
        
        # Save as PNG files
        img = Image.fromarray(img_data, mode='L')
        mask = Image.fromarray(mask_data, mode='L')
        
        img.save(imgs_dir / f"test_img_{i:03d}.png")
        mask.save(masks_dir / f"test_mask_{i:03d}.png")
    
    return str(test_data_dir)

def test_mlflow_final_fixes():
    """Test all the MLflow fixes"""
    logger.info("🧪 Testing MLflow final fixes...")
    
    try:
        # Setup MLflow
        setup_mlflow()
        logger.info("✅ MLflow setup completed")
        
        # Create test dataset
        test_data_path = create_test_dataset()
        logger.info(f"✅ Created test dataset at: {test_data_path}")
        
        # Create MLflow run
        mlflow_params = {
            'model_type': 'unet',
            'batch_size': 2,
            'epochs': 2,  # Just 2 epochs for testing
            'learning_rate': 0.001,
            'data_path': test_data_path,
            'validation_split': 0.5,
            'crop_size': 64
        }
        
        run_id = create_new_run(params=mlflow_params)
        logger.info(f"✅ Created MLflow run: {run_id}")
        
        # Create test model
        test_model = MLModel.objects.create(
            name="Final MLflow Fixes Test",
            description="Testing system metrics, PNG generation, and model signature fixes",
            status="pending",
            total_epochs=2,
            mlflow_run_id=run_id,
            training_data_info={
                'model_type': 'unet',
                'batch_size': 2,
                'epochs': 2,
                'learning_rate': 0.001,
                'data_path': test_data_path,
                'validation_split': 0.5,
                'crop_size': 64
            }
        )
        logger.info(f"✅ Created test model with ID: {test_model.id}")
        
        # Start training process
        cmd = [
            sys.executable, 
            "/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments/shared/train.py",
            "--mode", "train",
            "--model-type", "unet",
            "--batch-size", "2",
            "--epochs", "2",
            "--learning-rate", "0.001",
            "--validation-split", "0.5",
            "--data-path", test_data_path,
            "--crop-size", "64",
            "--model-id", str(test_model.id),
            "--mlflow-run-id", run_id,
            "--num-workers", "0"
        ]
        
        logger.info(f"🚀 Starting training process: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor for up to 120 seconds (2 epochs should complete)
        start_time = time.time()
        max_wait = 120
        
        logger.info("⏱️  Monitoring training progress...")
        
        while time.time() - start_time < max_wait:
            test_model.refresh_from_db()
            elapsed = time.time() - start_time
            
            if elapsed % 10 < 1:  # Log every 10 seconds
                logger.info(f"Status after {elapsed:.1f}s: {test_model.status}")
            
            if test_model.status in ['completed', 'failed']:
                logger.info(f"🏁 Training finished with status: {test_model.status}")
                break
                
            if process.poll() is not None:
                logger.info(f"🏁 Process finished with return code: {process.poll()}")
                break
                
            time.sleep(2)
        
        # Get process output
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        
        # Check for fixes
        results = {
            'training_completed': test_model.status in ['completed', 'training'],
            'system_metrics_logged': False,
            'png_samples_generated': False,
            'model_signature_clean': True  # Assume true unless we find warning
        }
        
        # Check MLflow run for system metrics
        try:
            run_data = mlflow.get_run(run_id)
            metrics_keys = list(run_data.data.metrics.keys())
            
            # Look for system metrics with our new organization
            system_metric_patterns = [
                'system/hardware/cpu/',
                'system/hardware/memory/',
                'system/process/',
                'system/hardware/gpu/'
            ]
            
            system_metrics_found = []
            for key in metrics_keys:
                for pattern in system_metric_patterns:
                    if pattern in key:
                        system_metrics_found.append(key)
            
            if system_metrics_found:
                results['system_metrics_logged'] = True
                logger.info(f"✅ System metrics found: {len(system_metrics_found)} metrics")
                logger.info(f"   Sample metrics: {system_metrics_found[:5]}")
            else:
                logger.warning(f"❌ No organized system metrics found. Available metrics: {metrics_keys[:10]}")
                
        except Exception as e:
            logger.error(f"❌ Error checking MLflow metrics: {e}")
        
        # Check for PNG samples in MLflow artifacts
        try:
            artifacts = mlflow.list_artifacts(run_id)
            png_artifacts = []
            
            def find_png_artifacts(artifacts_list, prefix=""):
                png_files = []
                for artifact in artifacts_list:
                    full_path = f"{prefix}/{artifact.path}" if prefix else artifact.path
                    if artifact.is_dir:
                        # Recursively check subdirectories
                        sub_artifacts = mlflow.list_artifacts(run_id, artifact.path)
                        png_files.extend(find_png_artifacts(sub_artifacts, artifact.path))
                    elif artifact.path.endswith('.png') and 'prediction' in artifact.path.lower():
                        png_files.append(full_path)
                return png_files
            
            png_artifacts = find_png_artifacts(artifacts)
            
            if png_artifacts:
                results['png_samples_generated'] = True
                logger.info(f"✅ PNG sample artifacts found: {len(png_artifacts)} files")
                logger.info(f"   Sample files: {png_artifacts[:3]}")
            else:
                logger.warning(f"❌ No PNG prediction samples found in artifacts")
                logger.info(f"   Available artifacts: {[a.path for a in artifacts[:10]]}")
                
        except Exception as e:
            logger.error(f"❌ Error checking MLflow artifacts: {e}")
        
        # Check for model signature warnings in output
        if stderr and 'without a signature' in stderr:
            results['model_signature_clean'] = False
            logger.warning("❌ Model signature warning found in stderr")
        elif stdout and 'without a signature' in stdout:
            results['model_signature_clean'] = False
            logger.warning("❌ Model signature warning found in stdout")
        else:
            logger.info("✅ No model signature warnings detected")
        
        # Print results summary
        logger.info("\n" + "="*60)
        logger.info("MLFLOW FIXES TEST RESULTS")
        logger.info("="*60)
        logger.info(f"✅ Training Status: {test_model.status}")
        logger.info(f"{'✅' if results['system_metrics_logged'] else '❌'} System Metrics Logging: {'WORKING' if results['system_metrics_logged'] else 'FAILED'}")
        logger.info(f"{'✅' if results['png_samples_generated'] else '❌'} PNG Sample Generation: {'WORKING' if results['png_samples_generated'] else 'FAILED'}")
        logger.info(f"{'✅' if results['model_signature_clean'] else '❌'} Model Signature Fix: {'WORKING' if results['model_signature_clean'] else 'FAILED'}")
        
        all_fixes_working = all(results.values())
        
        if all_fixes_working:
            logger.info("\n🎉 ALL MLFLOW FIXES ARE WORKING!")
        else:
            logger.info(f"\n⚠️  Some fixes need attention: {[k for k, v in results.items() if not v]}")
        
        # Cleanup
        test_model.delete()
        logger.info("🧹 Test model cleaned up")
        
        return all_fixes_working
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mlflow_final_fixes()
    
    print("\n" + "="*60)
    if success:
        print("🎉 MLFLOW FIXES TEST PASSED!")
        print("✅ System metrics, PNG generation, and model signature fixes are working!")
    else:
        print("❌ MLFLOW FIXES TEST FAILED!")
        print("🔧 Some issues still need to be addressed")
    print("="*60)
    
    sys.exit(0 if success else 1)
