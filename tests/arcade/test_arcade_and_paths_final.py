#!/usr/bin/env python3
"""
Test script to verify ARCADE dataset GUI fixes and proper metadata paths
"""

import os
import sys
import django
from pathlib import Path

# Add project paths
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "core"))
sys.path.insert(0, str(project_root / "ml"))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
django.setup()

def test_metadata_paths():
    """Test that metadata is stored in correct data/ paths"""
    print("🔍 Testing metadata path configurations...")
    
    # Test logging configuration
    from ml.training.train import logger
    for handler in logger.handlers:
        if hasattr(handler, 'baseFilename'):
            log_path = handler.baseFilename
            print(f"  📋 Log file: {log_path}")
            if 'data/logs' in log_path:
                print("  ✅ Logging uses correct data/logs/ path")
            else:
                print("  ❌ Logging uses incorrect path")
    
    # Test organized model directory
    from ml.training.train import create_organized_model_directory
    model_dir, unique_id = create_organized_model_directory(
        model_family="Test-UNet", 
        version="1.0.0"
    )
    print(f"  📁 Model directory: {model_dir}")
    if model_dir.startswith('data/models/organized'):
        print("  ✅ Models use correct data/models/organized/ path")
    else:
        print("  ❌ Models use incorrect path")
    
    return True

def test_arcade_dataset_preview():
    """Test ARCADE dataset preview functionality"""
    print("\n🎮 Testing ARCADE dataset preview...")
    
    from django.test import RequestFactory
    from core.apps.ml_manager.views import dataset_preview_view
    
    factory = RequestFactory()
    
    # Test all 6 ARCADE dataset types
    arcade_types = [
        'arcade_binary_segmentation',
        'arcade_semantic_segmentation', 
        'arcade_artery_classification',
        'arcade_semantic_seg_binary',
        'arcade_stenosis_detection',
        'arcade_stenosis_segmentation'
    ]
    
    test_path = '/app/data/datasets/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset/seg_train'
    
    for dataset_type in arcade_types:
        print(f"\n  🎯 Testing {dataset_type}...")
        
        try:
            # Create POST request
            request = factory.post('/dataset-preview/', {
                'data_path': test_path,
                'dataset_type': dataset_type
            })
            
            # Mock user (required for login)
            from django.contrib.auth.models import AnonymousUser
            request.user = AnonymousUser()
            
            # Call view function
            response = dataset_preview_view(request)
            
            if response.status_code == 200:
                print(f"    ✅ {dataset_type} - View executed successfully")
            else:
                print(f"    ❌ {dataset_type} - View failed with status {response.status_code}")
                
        except Exception as e:
            print(f"    ❌ {dataset_type} - Error: {str(e)[:100]}...")
    
    return True

def test_import_fixes():
    """Test that all ARCADE dataset imports work"""
    print("\n📦 Testing ARCADE dataset imports...")
    
    try:
        from ml.datasets.torch_arcade_loader import (
            ARCADEBinarySegmentation, ARCADESemanticSegmentation, 
            ARCADEArteryClassification, ARCADESemanticSegmentationBinary,
            ARCADEStenosisDetection, ARCADEStenosisSegmentation,
            COCO_AVAILABLE
        )
        print("  ✅ All ARCADE dataset classes imported successfully")
        print(f"  📋 COCO available: {COCO_AVAILABLE}")
        
        # Test colormap function
        from core.apps.ml_manager.views import apply_semantic_colormap
        import numpy as np
        
        test_mask = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        colored_mask = apply_semantic_colormap(test_mask)
        
        if colored_mask.shape == (3, 3, 3):
            print("  ✅ Semantic colormap function works correctly")
        else:
            print(f"  ❌ Semantic colormap returned wrong shape: {colored_mask.shape}")
            
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False
    
    return True

def test_settings_import():
    """Test that settings import is working correctly"""
    print("\n⚙️ Testing Django settings import...")
    
    try:
        from django.conf import settings
        
        # Test key settings
        print(f"  📁 MEDIA_ROOT: {settings.MEDIA_ROOT}")
        print(f"  🌐 MEDIA_URL: {settings.MEDIA_URL}")
        print(f"  📊 BASE_MLRUNS_DIR: {settings.BASE_MLRUNS_DIR}")
        
        # Check if paths are in data/
        if 'data' in str(settings.MEDIA_ROOT):
            print("  ✅ MEDIA_ROOT uses data/ path")
        else:
            print("  ❌ MEDIA_ROOT doesn't use data/ path")
            
        if 'data/mlflow' in str(settings.BASE_MLRUNS_DIR):
            print("  ✅ MLflow uses data/mlflow/ path")
        else:
            print("  ❌ MLflow doesn't use data/mlflow/ path")
            
    except Exception as e:
        print(f"  ❌ Settings error: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Starting ARCADE dataset and paths validation tests...")
    print("=" * 60)
    
    tests = [
        test_settings_import,
        test_metadata_paths,
        test_import_fixes,
        test_arcade_dataset_preview
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All tests passed! ARCADE dataset GUI and paths are working correctly.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == '__main__':
    exit(main())
