#!/usr/bin/env python3
"""
Test script to verify ARCADE dataset root path fixes
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / 'ml'))
sys.path.insert(0, str(project_root / 'core'))

def test_arcade_root_function():
    """Test the get_arcade_dataset_root function"""
    print("🧪 Testing ARCADE Root Path Function")
    print("=" * 50)
    
    try:
        from ml.datasets.arcade_loader import get_arcade_dataset_root
        
        # Test different path scenarios
        test_cases = [
            "/app/data/datasets/arcade_challenge_datasets",
            "/app/data/datasets/arcade_challenge_datasets/dataset_phase_1", 
            "/app/data/datasets",
            "/app/data/datasets/some_other_dir/arcade_challenge_datasets",
            "/home/user/data/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset"
        ]
        
        for test_path in test_cases:
            try:
                result = get_arcade_dataset_root(test_path)
                print(f"📁 Input:  {test_path}")
                print(f"📁 Output: {result}")
                print()
            except Exception as e:
                print(f"❌ Error with {test_path}: {e}")
                print()
        
        return True
    except ImportError as e:
        print(f"❌ Could not import function: {e}")
        return False

def test_arcade_dataset_loading():
    """Test actual ARCADE dataset loading"""
    print("🧪 Testing ARCADE Dataset Loading")
    print("=" * 50)
    
    # Check common dataset paths
    possible_paths = [
        "/app/data/datasets",
        "/app/data",
        "./data/datasets",
        "./datasets"
    ]
    
    arcade_found = False
    
    for base_path in possible_paths:
        arcade_path = os.path.join(base_path, "arcade_challenge_datasets")
        if os.path.exists(arcade_path):
            print(f"✅ Found ARCADE dataset at: {arcade_path}")
            arcade_found = True
            
            # Test contents
            try:
                contents = os.listdir(arcade_path)
                print(f"📋 Contents: {contents}")
                
                # Check for expected directories  
                expected_dirs = ["dataset_phase_1", "dataset_final_phase"]
                for expected in expected_dirs:
                    if expected in contents:
                        print(f"✅ Found expected directory: {expected}")
                        
                        # Check deeper structure
                        phase_path = os.path.join(arcade_path, expected)
                        if os.path.exists(phase_path):
                            phase_contents = os.listdir(phase_path)
                            print(f"📋 {expected} contents: {phase_contents}")
                    else:
                        print(f"❌ Missing expected directory: {expected}")
                        
            except Exception as e:
                print(f"❌ Error checking contents: {e}")
                
            print()
            break
    
    if not arcade_found:
        print("❌ No ARCADE dataset found in common locations")
        print("💡 Make sure torch-arcade dataset is downloaded")
        
    return arcade_found

def test_torch_arcade_import():
    """Test if torch-arcade components can be imported"""
    print("🧪 Testing torch-arcade Imports")
    print("=" * 50)
    
    try:
        from ml.datasets.torch_arcade_loader import (
            ARCADEBinarySegmentation,
            ARCADESemanticSegmentation,
            ARCADEStenosisDetection,
            create_arcade_dataloader
        )
        print("✅ Successfully imported torch-arcade components")
        return True
    except ImportError as e:
        print(f"❌ Could not import torch-arcade components: {e}")
        print("💡 Make sure pycocotools is installed: pip install pycocotools")
        return False

def main():
    """Run all tests"""
    print("🚀 ARCADE Root Path Fix Test Suite")
    print("=" * 60)
    print()
    
    results = []
    
    # Test 1: Root path function
    results.append(("Root Path Function", test_arcade_root_function()))
    
    # Test 2: torch-arcade imports
    results.append(("torch-arcade Imports", test_torch_arcade_import()))
    
    # Test 3: Dataset loading
    results.append(("Dataset Loading", test_arcade_dataset_loading()))
    
    # Summary
    print("📊 Test Results Summary")
    print("=" * 30)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\n🎯 {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("🎉 All tests passed! ARCADE integration should work correctly.")
    else:
        print("⚠️  Some tests failed. Check the issues above.")

if __name__ == "__main__":
    main()
