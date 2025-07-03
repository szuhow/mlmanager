#!/usr/bin/env python3

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml', 'datasets'))

from arcade_loader import is_arcade_dataset, detect_arcade_task_type

def test_arcade_detection():
    """Test ARCADE dataset detection"""
    
    print("=== Testing ARCADE Dataset Detection ===")
    
    # Test path
    test_path = "/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments/data/datasets/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset"
    
    print(f"Testing path: {test_path}")
    print(f"Path exists: {os.path.exists(test_path)}")
    
    # Test detection
    is_arcade = is_arcade_dataset(test_path)
    print(f"Is ARCADE dataset: {is_arcade}")
    
    if is_arcade:
        task_type = detect_arcade_task_type(test_path)
        print(f"Detected task type: {task_type}")
        
        # Test semantic cache detection
        semantic_cache_path = os.path.join(test_path, "seg_val", "masks_semantic_cache")
        print(f"Semantic cache exists: {os.path.exists(semantic_cache_path)}")
        
        if os.path.exists(semantic_cache_path):
            files = os.listdir(semantic_cache_path)
            print(f"Semantic cache files: {len(files)} files")
            if files:
                print(f"Example files: {files[:3]}")
    
    print(f"\n=== Test Complete ===")

if __name__ == "__main__":
    test_arcade_detection()
