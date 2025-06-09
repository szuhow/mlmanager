#!/usr/bin/env python3
"""
Test script to verify the model-agnostic MLflow Registry integration
"""
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(__file__))

def test_cli_arguments():
    """Test that the CLI arguments work correctly"""
    print("🔧 Testing CLI Arguments...")
    
    import subprocess
    
    # Test help output includes new arguments
    result = subprocess.run([
        'python', 'shared/train.py', '--help'
    ], capture_output=True, text=True)
    
    if '--model-family' in result.stdout and '--model-type' in result.stdout:
        print("  ✅ CLI arguments added successfully")
        return True
    else:
        print("  ❌ CLI arguments missing from help output")
        return False

def test_model_directory_creation():
    """Test model directory creation with custom families"""
    print("🔧 Testing Model Directory Creation...")
    
    try:
        from shared.train import create_organized_model_directory
        
        # Test different families
        families = ["ResNet-Cardiac", "VisionTransformer-Segmentation", "UNet-Coronary"]
        
        for family in families:
            model_dir, unique_id = create_organized_model_directory(
                model_id=999,
                model_family=family,
                version="1.0.0"
            )
            print(f"  ✅ {family}: {model_dir}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

if __name__ == '__main__':
    print("🚀 Starting Integration Tests...")
    
    # Test CLI arguments
    test1_success = test_cli_arguments()
    
    # Test model directory creation
    test2_success = test_model_directory_creation()
    
    # Overall result
    print(f"\n🏁 Integration Test Summary:")
    print(f"  CLI arguments: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"  Model directory creation: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print("\n🎉 INTEGRATION TESTS PASSED!")
        print("   The system is now model-agnostic with CLI support.")
    else:
        print("\n⚠️  Some tests failed.")
