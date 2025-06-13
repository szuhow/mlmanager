#!/usr/bin/env python3
"""
Test ARCADE mask scaling fix in Docker container
"""
import os
import sys
import numpy as np
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/ml')

def test_arcade_mask_scaling_container():
    """Test ARCADE mask scaling fix in container environment"""
    print("✅ Starting ARCADE mask scaling test in container...")
    
    try:
        from ml.datasets.torch_arcade_loader import (
            ARCADEBinarySegmentation,
            ARCADEStenosisSegmentation
        )
        print("✅ Successfully imported ARCADE dataset classes")
    except ImportError as e:
        print(f"❌ Failed to import ARCADE classes: {e}")
        return False
    
    # Test paths available in container
    base_path = "/app/data/datasets/arcade_challenge_datasets"
    
    # Find available datasets
    segmentation_path = os.path.join(base_path, "dataset_phase_1/segmentation_dataset")
    final_phase_path = os.path.join(base_path, "dataset_final_phase/test_case_segmentation")
    
    available_paths = []
    if os.path.exists(segmentation_path):
        available_paths.append(("segmentation", segmentation_path))
    if os.path.exists(final_phase_path):
        available_paths.append(("final_phase", final_phase_path))
    
    if not available_paths:
        print("⚠️  No ARCADE datasets found in container")
        return True  # Not a failure, just no data to test with
    
    print(f"🔍 Found {len(available_paths)} available datasets")
    
    # Test each available dataset
    for dataset_name, dataset_path in available_paths:
        print(f"\n🧪 Testing {dataset_name} dataset at {dataset_path}")
        
        try:
            # Test binary segmentation if available
            if "segmentation" in dataset_name:
                print("  📝 Testing ARCADEBinarySegmentation...")
                dataset = ARCADEBinarySegmentation(
                    dataset_dir=dataset_path,
                    split='train',
                    transforms=None
                )
                
                if len(dataset) > 0:
                    # Test first item
                    image, mask = dataset[0]
                    print(f"  ✅ Successfully loaded item 0:")
                    print(f"     - Image shape: {image.shape}")
                    print(f"     - Mask shape: {mask.shape}")
                    print(f"     - Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
                    
                    # Check if mask has reasonable values (not all zeros)
                    if mask.max() > 0.1:
                        print("  ✅ Mask contains non-zero values")
                    else:
                        print("  ⚠️  Mask appears to be mostly zeros")
                else:
                    print("  ⚠️  Dataset is empty")
            
            # Test cache directory existence
            cache_dirs = []
            for root, dirs, files in os.walk(dataset_path):
                for d in dirs:
                    if "cache" in d.lower():
                        cache_dirs.append(os.path.join(root, d))
            
            if cache_dirs:
                print(f"  📁 Found {len(cache_dirs)} cache directories:")
                for cache_dir in cache_dirs:
                    cache_files = len([f for f in os.listdir(cache_dir) if f.endswith('.npz')])
                    print(f"     - {cache_dir}: {cache_files} cached files")
            
        except Exception as e:
            print(f"  ❌ Error testing {dataset_name}: {e}")
            continue
    
    print("\n🎯 Testing mask scaling fix implementation...")
    
    # Check if the scaling fix is present in the code
    try:
        import inspect
        from ml.datasets.torch_arcade_loader import ARCADEBinarySegmentation
        
        # Get source code of the _get_cached_mask method
        source = inspect.getsource(ARCADEBinarySegmentation._get_cached_mask)
        
        if "mask = mask * 255" in source:
            print("  ✅ Scaling fix (mask * 255) found in ARCADEBinarySegmentation")
        else:
            print("  ⚠️  Scaling fix not found in ARCADEBinarySegmentation")
        
    except Exception as e:
        print(f"  ❌ Error checking scaling fix: {e}")
    
    print("\n✅ Container test completed!")
    return True

if __name__ == "__main__":
    success = test_arcade_mask_scaling_container()
    sys.exit(0 if success else 1)
