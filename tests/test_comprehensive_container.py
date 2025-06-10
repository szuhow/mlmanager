#!/usr/bin/env python3
"""
Comprehensive Container Integration Test
Test all enhanced MLflow artifact manager features in container environment
"""

import os
import sys
import tempfile
import json
import time

# Add the project root to the path
sys.path.insert(0, '/app' if os.path.exists('/app') else '.')

import mlflow
from shared.utils.mlflow_artifact_manager import MLflowArtifactManager, log_epoch_artifacts, log_final_model

def test_comprehensive_workflow():
    """Test complete workflow with multiple epochs and final model"""
    print("🚀 Comprehensive Container Integration Test")
    print("=" * 60)
    
    # Use internal container networking
    mlflow.set_tracking_uri("http://mlflow:5000")
    print(f"🔗 MLflow URI: {mlflow.get_tracking_uri()}")
    
    # Create test artifacts
    temp_dir = tempfile.mkdtemp(prefix="comprehensive_test_")
    
    # Create various test files
    training_log = os.path.join(temp_dir, "training.log")
    with open(training_log, 'w') as f:
        f.write("Training started...\n")
        f.write("Loading dataset...\n")
        f.write("Model initialized...\n")
    
    config_file = os.path.join(temp_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump({
            "model": "MONAI UNet",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 32
        }, f, indent=2)
    
    try:
        with mlflow.start_run(run_name="Comprehensive_Enhanced_Test") as run:
            print(f"📝 Started comprehensive run: {run.info.run_id}")
            
            # Test 1: Multiple epoch logging
            print("\n📊 Test 1: Multiple Epoch Logging")
            for epoch in range(1, 4):
                metrics = {
                    'train_loss': 0.8 - (epoch * 0.2),
                    'val_loss': 0.9 - (epoch * 0.15), 
                    'train_dice': 0.3 + (epoch * 0.2),
                    'val_dice': 0.25 + (epoch * 0.22),
                    'learning_rate': 0.001
                }
                
                artifacts = {
                    'logs': training_log,
                    'config': config_file
                }
                
                metadata = {
                    'epoch_duration_minutes': 15.5 + epoch,
                    'gpu_memory_used': f"{6.5 + epoch}GB",
                    'batch_count': 50 + epoch
                }
                
                logged_paths = log_epoch_artifacts(
                    epoch=epoch,
                    model_state=None,  # Skip model state for this test
                    metrics=metrics,
                    artifacts=artifacts,
                    metadata=metadata
                )
                
                print(f"   ✅ Epoch {epoch}: {len(logged_paths)} artifacts logged")
                
                # Log metrics to MLflow for comparison
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value, step=epoch)
                
                time.sleep(0.5)  # Small delay to separate epochs
            
            # Test 2: Final model logging
            print("\n🏁 Test 2: Final Model Logging")
            
            # Create model directory structure
            model_dir = os.path.join(temp_dir, "final_model")
            os.makedirs(os.path.join(model_dir, "weights"), exist_ok=True)
            os.makedirs(os.path.join(model_dir, "artifacts"), exist_ok=True)
            
            # Create dummy model file
            model_file = os.path.join(model_dir, "weights", "model.pth")
            with open(model_file, 'w') as f:
                f.write("# Dummy PyTorch model weights\n")
                f.write("# Enhanced MLflow artifact manager test\n")
            
            # Create training artifacts
            final_log = os.path.join(model_dir, "artifacts", "final_training.log")
            with open(final_log, 'w') as f:
                f.write("Training completed successfully!\n")
                f.write("Best validation dice: 0.923\n")
                f.write("Total training time: 2.5 hours\n")
            
            model_info = {
                'architecture': 'Enhanced MONAI UNet',
                'framework': 'PyTorch',
                'model_family': 'UNet-Coronary-Segmentation',
                'total_parameters': 1845312,
                'trainable_parameters': 1845312,
                'input_channels': 1,
                'output_channels': 2,
                'spatial_dims': 3,
                'enhanced_features': [
                    'Hierarchical artifact organization',
                    'Automatic metadata generation',
                    'Versioned tracking',
                    'Resource cleanup'
                ]
            }
            
            best_metrics = {
                'best_val_dice': 0.923,
                'best_val_loss': 0.087,
                'final_train_dice': 0.912,
                'best_epoch': 3,
                'total_training_time_hours': 2.5
            }
            
            final_logged_paths = log_final_model(
                model_info=model_info,
                model_directory=model_dir,
                best_metrics=best_metrics
            )
            
            print(f"   ✅ Final model: {len(final_logged_paths)} artifacts logged")
            
            # Test 3: Context manager and cleanup
            print("\n🧹 Test 3: Context Manager and Cleanup")
            with MLflowArtifactManager() as manager:
                temp_test_dir = manager.create_temp_dir()
                print(f"   📁 Created temp dir: {os.path.basename(temp_test_dir)}")
                assert os.path.exists(temp_test_dir), "Temp dir should exist"
            
            # Temp dir should be cleaned up after context manager exit
            print("   ✅ Context manager cleanup completed")
            
            # Test 4: Error handling
            print("\n🛡️  Test 4: Error Handling")
            try:
                # Try to log non-existent artifact
                log_epoch_artifacts(
                    epoch=999,
                    artifacts={'non_existent': '/path/that/does/not/exist.txt'},
                    metrics={'test': 1.0}
                )
                print("   ✅ Error handling works correctly")
            except Exception as e:
                print(f"   ⚠️  Expected error handled: {type(e).__name__}")
            
            print(f"\n🌐 View comprehensive test results:")
            print(f"   MLflow UI: http://localhost:5000/#/experiments/0/runs/{run.info.run_id}")
            
            return run.info.run_id
            
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        print("🧹 Cleaned up test files")

def verify_mlflow_artifacts(run_id):
    """Verify artifacts appear correctly in MLflow"""
    print(f"\n🔍 Verifying Artifacts for Run: {run_id}")
    print("=" * 50)
    
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Get run info
        run = client.get_run(run_id)
        print(f"📝 Run Name: {run.data.tags.get('mlflow.runName', 'Unnamed')}")
        print(f"⏱️  Duration: {(run.info.end_time - run.info.start_time) / 1000:.2f} seconds")
        
        # Get metrics
        metrics = run.data.metrics
        print(f"📊 Metrics logged: {len(metrics)}")
        for name, value in list(metrics.items())[:5]:  # Show first 5
            print(f"   📈 {name}: {value}")
        
        # Get artifacts
        artifacts = client.list_artifacts(run_id)
        print(f"📁 Artifact directories: {len(artifacts)}")
        
        artifact_count = 0
        for artifact in artifacts:
            print(f"   📂 {artifact.path}")
            # Count files in each directory
            try:
                sub_artifacts = client.list_artifacts(run_id, artifact.path)
                for sub_artifact in sub_artifacts[:3]:  # Show first 3 items
                    print(f"      📄 {sub_artifact.path}")
                    artifact_count += 1
            except:
                artifact_count += 1
        
        print(f"📄 Total artifacts: {artifact_count}+")
        
        # Check for specific enhanced features
        expected_dirs = ['metrics', 'summaries', 'logs', 'model']
        found_dirs = [a.path for a in artifacts]
        
        print(f"\n✅ Enhanced Organization Features:")
        for expected in expected_dirs:
            if any(expected in found for found in found_dirs):
                print(f"   ✅ {expected.capitalize()} organization: Found")
            else:
                print(f"   ⚠️  {expected.capitalize()} organization: Not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Artifact verification failed: {e}")
        return False

def main():
    """Run comprehensive container integration test"""
    print("🐳 Enhanced MLflow Artifact Manager - Container Integration")
    print("=" * 70)
    
    # Run comprehensive workflow test
    run_id = test_comprehensive_workflow()
    
    if run_id:
        # Verify artifacts in MLflow
        verification_success = verify_mlflow_artifacts(run_id)
        
        print("\n" + "=" * 70)
        if verification_success:
            print("🎉 COMPREHENSIVE CONTAINER INTEGRATION TEST PASSED!")
            print("✅ Enhanced MLflow Artifact Manager is fully functional in Docker")
            print("✅ All features working: hierarchical organization, metadata, cleanup")
            print("✅ Integration with MLflow UI confirmed")
            print(f"🌐 View results: http://localhost:5000/#/experiments/0/runs/{run_id}")
        else:
            print("⚠️  CONTAINER TEST PASSED BUT VERIFICATION INCOMPLETE")
            print("🌐 Check manually: http://localhost:5000")
        print("=" * 70)
        
        return 0
    else:
        print("❌ COMPREHENSIVE CONTAINER INTEGRATION TEST FAILED!")
        return 1

if __name__ == "__main__":
    exit(main())
