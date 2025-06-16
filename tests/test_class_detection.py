#!/usr/bin/env python3
"""
Test script for class detection functionality
"""

import sys
import os
import numpy as np
import torch

# Add the project root to the path
sys.path.insert(0, '/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments')

# Mock dataset for testing
class MockDataset:
    def __init__(self, mask_type='binary'):
        self.mask_type = mask_type
        self.length = 10
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Create fake image and mask data
        image = torch.randn(3, 128, 128)  # RGB image
        
        if self.mask_type == 'binary':
            # Binary mask: 0 and 1
            mask = torch.randint(0, 2, (1, 128, 128)).float()
        elif self.mask_type == 'semantic_onehot':
            # One-hot encoded semantic mask (like ARCADE)
            mask = torch.zeros(128, 128, 27)  # 27 channels as last dimension
            # Ensure we always activate at least a few channels with actual data
            for c in range(min(5, 27)):  # Activate first 5 channels consistently
                mask[:, :, c] = torch.randint(0, 2, (128, 128)).float()
            # Make sure we have some non-zero values
            mask[10:20, 10:20, 0] = 1.0  # Guarantee some activation
        elif self.mask_type == 'semantic_single':
            # Single channel with multiple classes
            mask = torch.randint(0, 5, (1, 128, 128)).float()  # 5 classes
        
        return image, mask

class MockDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset

def test_class_detection():
    print("Testing class detection functionality...")
    
    # Import the functions from the train module
    from ml.training.train import detect_num_classes_from_masks, _analyze_class_distribution
    
    # Test 1: Binary segmentation
    print("\n=== Test 1: Binary Segmentation ===")
    binary_dataset = MockDataset('binary')
    binary_loader = MockDataLoader(binary_dataset)
    
    result = detect_num_classes_from_masks((binary_loader, binary_loader), dataset_type="binary", max_samples=5)
    print(f"Result: {result}")
    
    expected = {'class_type': 'binary', 'num_classes': 1}
    assert result['class_type'] == expected['class_type'], f"Expected {expected['class_type']}, got {result['class_type']}"
    assert result['num_classes'] == expected['num_classes'], f"Expected {expected['num_classes']}, got {result['num_classes']}"
    print("✅ Binary segmentation test passed!")
    
    # Test 2: One-hot semantic segmentation (ARCADE style)
    print("\n=== Test 2: One-hot Semantic Segmentation ===")
    semantic_dataset = MockDataset('semantic_onehot')
    semantic_loader = MockDataLoader(semantic_dataset)
    
    # Debug: Check what our mock data looks like
    sample_image, sample_mask = semantic_dataset[0]
    print(f"Debug: Sample mask shape: {sample_mask.shape}")
    print(f"Debug: Sample mask unique values: {torch.unique(sample_mask)}")
    print(f"Debug: Sample mask min/max: {sample_mask.min()}/{sample_mask.max()}")
    
    result = detect_num_classes_from_masks((semantic_loader, semantic_loader), dataset_type="arcade", max_samples=5)
    print(f"Result: {result}")
    
    expected = {'class_type': 'semantic_onehot', 'max_channels': 27}
    assert result['class_type'] == expected['class_type'], f"Expected {expected['class_type']}, got {result['class_type']}"
    assert result['max_channels'] == expected['max_channels'], f"Expected {expected['max_channels']}, got {result['max_channels']}"
    print("✅ One-hot semantic segmentation test passed!")
    
    # Test 3: Single-channel multi-class segmentation
    print("\n=== Test 3: Single-channel Multi-class Segmentation ===")
    multi_dataset = MockDataset('semantic_single')
    multi_loader = MockDataLoader(multi_dataset)
    
    result = detect_num_classes_from_masks((multi_loader, multi_loader), dataset_type="auto", max_samples=5)
    print(f"Result: {result}")
    
    expected = {'class_type': 'semantic_single'}
    assert result['class_type'] == expected['class_type'], f"Expected {expected['class_type']}, got {result['class_type']}"
    assert result['num_classes'] >= 5, f"Expected at least 5 classes, got {result['num_classes']}"
    print("✅ Multi-class segmentation test passed!")
    
    print("\n🎉 All class detection tests passed!")
    
    # Test helper function directly
    print("\n=== Test 4: Direct helper function test ===")
    
    # Test binary case
    result = _analyze_class_distribution([0, 1], 1, "binary")
    assert result['class_type'] == 'binary'
    print("✅ Binary helper test passed!")
    
    # Test one-hot case
    result = _analyze_class_distribution([0, 1], 27, "arcade")
    assert result['class_type'] == 'semantic_onehot'
    assert result['max_channels'] == 27
    print("✅ One-hot helper test passed!")
    
    # Test multi-class case
    result = _analyze_class_distribution([0, 1, 2, 3, 4], 1, "auto")
    assert result['class_type'] == 'semantic_single'
    assert result['num_classes'] == 5
    print("✅ Multi-class helper test passed!")
    
    print("\n🎉 All tests completed successfully!")

if __name__ == "__main__":
    test_class_detection()
