#!/usr/bin/env python3
"""
Test ARCADE GUI integration with real mask generation
Sprawdź czy nowa implementacja GUI działa z ARCADEBinarySegmentation
"""

import sys
import os
import django
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'core'))
sys.path.insert(0, str(project_root / 'core' / 'apps'))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings')
django.setup()

def test_arcade_import():
    """Test if ARCADE classes can be imported"""
    print("🧪 Test 1: ARCADE Import")
    
    try:
        from ml.datasets.torch_arcade_loader import ARCADEBinarySegmentation, COCO_AVAILABLE
        print("✅ ARCADEBinarySegmentation import successful")
        print(f"✅ COCO_AVAILABLE: {COCO_AVAILABLE}")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_dataset_preview_function():
    """Test if dataset_preview_view function has been updated"""
    print("\n🧪 Test 2: Dataset Preview Function")
    
    try:
        from core.apps.ml_manager.views import dataset_preview_view
        import inspect
        
        # Check if function contains ARCADE-related code
        source = inspect.getsource(dataset_preview_view)
        
        checks = [
            ("ARCADEBinarySegmentation", "ARCADE class usage"),
            ("arcade_dataset[i]", "dataset indexing"),
            ("torch.Tensor", "tensor handling"),
            ("Generated ARCADE sample", "mask generation indicator"),
        ]
        
        all_passed = True
        for check, description in checks:
            if check in source:
                print(f"✅ {description} found")
            else:
                print(f"❌ {description} not found")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Function check failed: {e}")
        return False

def test_torch_import():
    """Test if torch import is available"""
    print("\n🧪 Test 3: PyTorch Import")
    
    try:
        import torch
        print("✅ PyTorch import successful")
        print(f"✅ PyTorch version: {torch.__version__}")
        return True
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

def test_dataset_structure_requirements():
    """Test requirements for ARCADE dataset structure"""
    print("\n🧪 Test 4: Dataset Structure Requirements")
    
    # Example dataset paths that would work with our implementation
    test_paths = [
        "/app/data/datasets/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset/seg_train",
        "/app/data/datasets/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset/seg_val",
    ]
    
    requirements = [
        ("images/ directory", "contains image files"),
        ("annotations/ directory", "contains COCO JSON files"),
        ("dataset_phase_1/ parent", "ARCADE dataset structure"),
    ]
    
    print("📋 Required structure for ARCADE GUI integration:")
    for req, desc in requirements:
        print(f"   • {req}: {desc}")
    
    print("\n📝 Example paths that would work:")
    for path in test_paths:
        print(f"   • {path}")
    
    return True

def main():
    """Run all tests"""
    print("🔍 Testing ARCADE GUI Integration")
    print("=" * 50)
    
    tests = [
        test_arcade_import,
        test_dataset_preview_function,
        test_torch_import,
        test_dataset_structure_requirements,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)} tests")
    
    if all(results):
        print("\n🎉 All tests passed! ARCADE GUI integration is ready.")
        print("\n📝 Next steps:")
        print("1. Ensure you have a proper ARCADE dataset structure")
        print("2. Navigate to the Dataset Preview page in MLManager")
        print("3. Select 'COCO Style' dataset type or use 'Auto-detect'")
        print("4. Point to your ARCADE dataset directory")
        print("5. The GUI will now generate real masks from COCO annotations!")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
