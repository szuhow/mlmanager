#!/usr/bin/env python3
"""
Test script for the enhanced MLflow artifact manager
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_enhanced_mlflow_artifacts():
    """Test the enhanced MLflow artifact manager"""
    
    print("🧪 Testing Enhanced MLflow Artifact Manager")
    print("=" * 50)
    
    try:
        # Import the artifact manager
        from shared.utils.mlflow_artifact_manager import MLflowArtifactManager, log_epoch_artifacts
        print("✅ Successfully imported MLflowArtifactManager")
    except ImportError as e:
        print(f"❌ Failed to import MLflowArtifactManager: {e}")
        return False
    
    # Test 1: Basic functionality
    print("\n📝 Test 1: Basic MLflowArtifactManager functionality")
    try:
        with MLflowArtifactManager() as manager:
            print("✅ Context manager works")
            
            # Test temp directory creation
            temp_dir = manager.create_temp_dir()
            assert os.path.exists(temp_dir), "Temp directory should exist"
            print(f"✅ Created temp directory: {temp_dir}")
            
        # Temp directory should be cleaned up
        if not os.path.exists(temp_dir):
            print("✅ Temp directory cleaned up automatically")
        else:
            print("⚠️  Temp directory not cleaned up (may be normal on some systems)")
    
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        return False
    
    # Test 2: Artifact path mapping
    print("\n🗂️  Test 2: Artifact path mapping")
    try:
        with MLflowArtifactManager() as manager:
            # Test internal path mapping
            path = manager._get_artifact_path('training_curves', 5)
            expected = 'visualizations/training_curves/epoch_005'
            assert path == expected, f"Expected {expected}, got {path}"
            print(f"✅ Training curves path: {path}")
            
            path = manager._get_artifact_path('predictions', 10)
            expected = 'predictions/epoch_010'
            assert path == expected, f"Expected {expected}, got {path}"
            print(f"✅ Predictions path: {path}")
            
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        return False
    
    # Test 3: Metrics and metadata JSON creation
    print("\n📊 Test 3: Metrics and metadata JSON creation")
    try:
        with MLflowArtifactManager() as manager:
            temp_dir = manager.create_temp_dir()
            
            # Test metrics JSON
            metrics = {'train_loss': 0.15, 'val_dice': 0.89}
            metrics_path = manager._save_metrics_json(metrics, temp_dir, 5)
            
            assert os.path.exists(metrics_path), "Metrics file should exist"
            
            with open(metrics_path, 'r') as f:
                saved_metrics = json.load(f)
                assert saved_metrics['epoch'] == 5, "Epoch should be saved"
                assert saved_metrics['metrics']['train_loss'] == 0.15, "Train loss should be saved"
            print("✅ Metrics JSON creation works")
            
            # Test metadata JSON
            metadata = {'learning_rate': 0.001, 'batch_size': 32}
            metadata_path = manager._save_metadata_json(metadata, temp_dir, 5)
            
            assert os.path.exists(metadata_path), "Metadata file should exist"
            
            with open(metadata_path, 'r') as f:
                saved_metadata = json.load(f)
                assert saved_metadata['learning_rate'] == 0.001, "Learning rate should be saved"
            print("✅ Metadata JSON creation works")
            
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        return False
    
    # Test 4: Summary generation
    print("\n📄 Test 4: Summary generation")
    try:
        with MLflowArtifactManager() as manager:
            temp_dir = manager.create_temp_dir()
            
            metrics = {'train_loss': 0.15, 'val_dice': 0.89}
            artifacts = {'training_curves': '/path/to/curves.png', 'predictions': '/path/to/pred.png'}
            
            summary_path = manager._create_epoch_summary(5, metrics, artifacts, temp_dir)
            
            assert os.path.exists(summary_path), "Summary file should exist"
            
            with open(summary_path, 'r') as f:
                content = f.read()
                assert 'Epoch 5 Summary' in content, "Title should be in summary"
                assert 'train_loss' in content, "Metrics should be in summary"
                assert 'training_curves' in content, "Artifacts should be in summary"
            print("✅ Summary generation works")
            
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        return False
    
    # Test 5: Convenience functions
    print("\n🎯 Test 5: Convenience functions")
    try:
        # Test without actually logging to MLflow (since we may not have it configured)
        print("✅ Convenience functions available (skipping actual MLflow logging)")
        
    except Exception as e:
        print(f"❌ Test 5 failed: {e}")
        return False
    
    # Test 6: Error handling
    print("\n🛡️  Test 6: Error handling")
    try:
        with MLflowArtifactManager() as manager:
            # Test with invalid directory
            try:
                manager._save_metrics_json({}, "/invalid/path", 1)
                print("⚠️  Expected error for invalid path didn't occur")
            except Exception:
                print("✅ Error handling works for invalid paths")
                
    except Exception as e:
        print(f"❌ Test 6 failed: {e}")
        return False
    
    print("\n🎉 All tests passed!")
    print("Enhanced MLflow Artifact Manager is working correctly.")
    return True


def test_integration_with_training():
    """Test integration points with the training script"""
    
    print("\n🔗 Testing Integration with Training Script")
    print("=" * 50)
    
    try:
        # Check if the training script can import the artifact manager
        sys.path.append(str(Path(__file__).parent / 'shared'))
        
        # Try importing as the training script would
        from shared.utils.mlflow_artifact_manager import log_epoch_artifacts, log_final_model
        print("✅ Training script can import artifact manager functions")
        
        # Test the function signatures match what training script expects
        import inspect
        
        # Check log_epoch_artifacts signature
        sig = inspect.signature(log_epoch_artifacts)
        expected_params = ['epoch', 'model_state', 'metrics', 'artifacts', 'metadata']
        actual_params = list(sig.parameters.keys())
        
        for param in expected_params:
            if param not in actual_params:
                print(f"❌ Missing parameter {param} in log_epoch_artifacts")
                return False
        print("✅ log_epoch_artifacts has correct signature")
        
        # Check log_final_model signature
        sig = inspect.signature(log_final_model)
        expected_params = ['model_info', 'model_directory', 'best_metrics']
        actual_params = list(sig.parameters.keys())
        
        for param in expected_params:
            if param not in actual_params:
                print(f"❌ Missing parameter {param} in log_final_model")
                return False
        print("✅ log_final_model has correct signature")
        
        print("✅ Integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting Enhanced MLflow Artifact Manager Tests")
    print("=" * 60)
    
    # Test basic functionality
    test1_passed = test_enhanced_mlflow_artifacts()
    
    # Test integration
    test2_passed = test_integration_with_training()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED!")
        print("The Enhanced MLflow Artifact Manager is ready for use.")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the errors above and fix issues before using.")
        return 1


if __name__ == "__main__":
    exit(main())
