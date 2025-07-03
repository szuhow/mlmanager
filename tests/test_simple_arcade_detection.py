#!/usr/bin/env python3

import os
from pathlib import Path

def is_arcade_dataset(data_path: str) -> bool:
    """Check if path contains ARCADE dataset structure"""
    path = Path(data_path)
    
    # Check for ARCADE directory structure
    arcade_indicators = [
        "arcade_challenge_datasets",
        "dataset_phase_1",
        "segmentation_dataset",
        "stenosis_dataset",
        "seg_train",
        "seg_val", 
        "sten_train",
        "sten_val"
    ]
    
    # Check if any indicator exists in the path or its subdirectories
    for indicator in arcade_indicators:
        if (path / indicator).exists():
            return True
        # Also check if the current path name contains the indicator
        if indicator in str(path):
            return True
    
    return False

def detect_arcade_task_type(data_path: str) -> str:
    """Automatically detect ARCADE task type from path structure"""
    path = Path(data_path)
    
    # Check for segmentation dataset
    if (path / "segmentation_dataset").exists() or "segmentation" in str(path):
        # Check if semantic masks are available
        seg_train_semantic = path / "seg_train" / "masks_semantic_cache"
        seg_val_semantic = path / "seg_val" / "masks_semantic_cache"
        
        # Also check in subdirectories
        if not seg_train_semantic.exists():
            for subdir in ["seg_train", "segmentation_dataset/seg_train"]:
                potential_path = path / subdir / "masks_semantic_cache"
                if potential_path.exists():
                    seg_train_semantic = potential_path
                    break
        
        if not seg_val_semantic.exists():
            for subdir in ["seg_val", "segmentation_dataset/seg_val"]:
                potential_path = path / subdir / "masks_semantic_cache"
                if potential_path.exists():
                    seg_val_semantic = potential_path
                    break
        
        # If semantic masks exist, return semantic segmentation
        if seg_train_semantic.exists() or seg_val_semantic.exists():
            return "semantic_segmentation"
        else:
            return "binary_segmentation"
    
    # Check for stenosis dataset
    if (path / "stenosis_dataset").exists() or "stenosis" in str(path):
        return "stenosis_detection"
    
    # Check for specific subdirectories
    if (path / "seg_train").exists() or (path / "seg_val").exists():
        # Check if semantic masks are available in these directories
        seg_train_semantic = path / "seg_train" / "masks_semantic_cache"
        seg_val_semantic = path / "seg_val" / "masks_semantic_cache"
        
        if seg_train_semantic.exists() or seg_val_semantic.exists():
            return "semantic_segmentation"
        else:
            return "binary_segmentation"
    
    if (path / "sten_train").exists() or (path / "sten_val").exists():
        return "stenosis_detection"
    
    # Default fallback
    return "binary_segmentation"

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
