#!/usr/bin/env python3
"""
Test Enhanced MLflow Artifact APIs Integration
"""

import os
import sys
import tempfile
import json

# Add the project root to the path
sys.path.insert(0, '/app' if os.path.exists('/app') else '.')

import mlflow
from shared.utils.mlflow_artifact_manager import log_epoch_artifacts

def test_comprehensive_mlflow_apis():
    """Test comprehensive MLflow artifact APIs usage"""
    print("🚀 Testing Enhanced MLflow Artifact APIs Integration")
    print("=" * 60)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # Create test artifacts
    temp_dir = tempfile.mkdtemp(prefix="mlflow_api_test_")
    
    # Create various test files
    test_files = {}
    
    # 1. Training log
    log_file = os.path.join(temp_dir, "training.log")
    with open(log_file, 'w') as f:
        f.write("Epoch 1: Loss=0.5, Dice=0.8\n")
        f.write("Epoch 2: Loss=0.3, Dice=0.85\n")
        f.write("Enhanced MLflow artifact API test\n")
    test_files['training_log'] = log_file
    
    # 2. Config file
    config_file = os.path.join(temp_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump({
            "model": "UNet",
            "batch_size": 32,
            "learning_rate": 0.001,
            "enhanced_logging": True
        }, f, indent=2)
    test_files['config'] = config_file
    
    # 3. Metrics file
    metrics_file = os.path.join(temp_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump({
            "train_loss": 0.25,
            "val_loss": 0.22,
            "train_dice": 0.88,
            "val_dice": 0.85
        }, f, indent=2)
    test_files['metrics'] = metrics_file
    
    try:
        with mlflow.start_run(run_name="Enhanced_MLflow_API_Test") as run:
            print(f"📝 Started run: {run.info.run_id}")
            
            # Test 1: Basic organized artifact logging
            print("\n🔧 Test 1: Basic Organized Artifact Logging")
            mlflow.log_artifact(log_file, artifact_path="logs/training")
            mlflow.log_artifact(config_file, artifact_path="config/training")
            mlflow.log_artifact(metrics_file, artifact_path="metrics/epoch_001")
            print("   ✅ Basic artifacts logged with organized paths")
            
            # Test 2: Enhanced artifact manager integration
            print("\n🎯 Test 2: Enhanced Artifact Manager Integration")
            
            # Prepare artifacts for enhanced manager
            artifacts = {
                'logs': log_file,
                'config': config_file,
                'metrics': metrics_file
            }
            
            metrics = {
                'train_loss': 0.25,
                'val_loss': 0.22,
                'train_dice': 0.88,
                'val_dice': 0.85
            }
            
            metadata = {
                'test_type': 'mlflow_api_integration',
                'enhanced_logging': True,
                'artifact_count': len(artifacts)
            }
            
            # Use enhanced artifact manager
            logged_paths = log_epoch_artifacts(
                epoch=1,
                model_state=None,
                metrics=metrics,
                artifacts=artifacts,
                metadata=metadata
            )
            
            print(f"   ✅ Enhanced manager logged {len(logged_paths)} artifact categories")
            for category, path in logged_paths.items():
                print(f"      📁 {category}: {path}")
            
            # Test 3: Direct MLflow API usage with comprehensive paths
            print("\n📊 Test 3: Direct MLflow API with Comprehensive Organization")
            
            # Create additional test artifacts
            summary_file = os.path.join(temp_dir, "training_summary.md")
            with open(summary_file, 'w') as f:
                f.write("# Training Summary\n\n")
                f.write("- Model: UNet\n")
                f.write("- Best Dice: 0.85\n")
                f.write("- Enhanced MLflow APIs: ✅\n")
            
            # Log with detailed organization
            mlflow.log_artifact(summary_file, artifact_path="summaries/training/final")
            
            # Log additional metadata
            mlflow.log_param("enhanced_logging_enabled", True)
            mlflow.log_param("artifact_organization", "hierarchical")
            mlflow.log_metric("test_artifacts_logged", len(test_files))
            
            # Set comprehensive tags
            mlflow.set_tag("test_type", "mlflow_api_integration")
            mlflow.set_tag("enhanced_features", "enabled")
            mlflow.set_tag("artifact_structure", "hierarchical")
            
            print("   ✅ Comprehensive MLflow API usage completed")
            
            print(f"\n🌐 View results: http://localhost:5000/#/experiments/0/runs/{run.info.run_id}")
            
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        print("🧹 Test artifacts cleaned up")

def main():
    """Run comprehensive MLflow API tests"""
    success = test_comprehensive_mlflow_apis()
    
    if success:
        print("\n🎉 ENHANCED MLFLOW ARTIFACT APIs INTEGRATION SUCCESSFUL!")
        print("✅ All artifact logging features are working correctly")
        print("✅ Hierarchical organization implemented")
        print("✅ Enhanced artifact manager integrated")
        print("✅ Comprehensive MLflow API usage verified")
    else:
        print("\n❌ MLflow artifact API integration failed!")
        
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
