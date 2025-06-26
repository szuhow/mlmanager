#!/usr/bin/env python3

import os
import sys
import django

# Add the core directory to the Python path
sys.path.insert(0, '/app/core')

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
django.setup()

def test_arcade_direct():
    """Test ARCADE detection directly"""
    
    from ml.datasets.arcade_loader import (
        is_arcade_dataset, 
        detect_arcade_task_type, 
        get_arcade_dataset_root
    )
    
    print("=== Testing ARCADE Detection Directly ===")
    
    # Test path mapped to container
    test_path = "/app/data/datasets/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset"
    
    print(f"Testing path: {test_path}")
    print(f"Path exists: {os.path.exists(test_path)}")
    
    if os.path.exists(test_path):
        # Test detection
        is_arcade = is_arcade_dataset(test_path)
        print(f"Is ARCADE dataset: {is_arcade}")
        
        if is_arcade:
            task_type = detect_arcade_task_type(test_path)
            print(f"Detected task type: {task_type}")
            
            root = get_arcade_dataset_root(test_path)
            print(f"ARCADE root: {root}")
            
            # Test torch_arcade_loader import
            try:
                from ml.datasets.torch_arcade_loader import ARCADESemanticSegmentation
                print("✅ ARCADESemanticSegmentation import successful")
                
                # Try to create dataset
                try:
                    dataset = ARCADESemanticSegmentation(
                        root=root,
                        image_set='train',
                        download=False,
                        transforms=None
                    )
                    print(f"✅ Dataset created successfully, length: {len(dataset)}")
                    
                    # Try to get first sample
                    if len(dataset) > 0:
                        sample = dataset[0]
                        print(f"✅ Sample retrieved: {type(sample)}")
                        if isinstance(sample, tuple):
                            print(f"   Image shape: {sample[0].shape if hasattr(sample[0], 'shape') else 'N/A'}")
                            print(f"   Mask shape: {sample[1].shape if hasattr(sample[1], 'shape') else 'N/A'}")
                    
                except Exception as e:
                    print(f"❌ Dataset creation failed: {e}")
                    
            except Exception as e:
                print(f"❌ ARCADESemanticSegmentation import failed: {e}")
        
    else:
        print("❌ Test path does not exist")
    
    print(f"\n=== Test Complete ===")

if __name__ == "__main__":
    test_arcade_direct()
