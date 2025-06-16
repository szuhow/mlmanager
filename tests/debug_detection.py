#!/usr/bin/env python3
"""
Debug script for class detection
"""

import sys
import os
import numpy as np
import torch

# Add the project root to the path
sys.path.insert(0, '/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments')

# Create a simple test case
def debug_detection():
    print("Debugging class detection...")
    
    # Import the function
    from ml.training.train import _analyze_class_distribution
    
    # Test the function directly with known inputs
    print("\n=== Testing _analyze_class_distribution directly ===")
    
    # Test 1: Binary case
    result = _analyze_class_distribution([0, 1], 1, "binary")
    print(f"Binary test: {result}")
    assert result['class_type'] == 'binary'
    
    # Test 2: One-hot case with many channels
    result = _analyze_class_distribution([0, 1], 27, "arcade")
    print(f"One-hot test (27 channels): {result}")
    assert result['class_type'] == 'semantic_onehot'
    assert result['max_channels'] == 27
    
    # Test 3: Multi-class case
    result = _analyze_class_distribution([0, 1, 2, 3, 4], 1, "auto")
    print(f"Multi-class test: {result}")
    assert result['class_type'] == 'semantic_single'
    
    print("✅ Direct function tests passed!")
    
    # Now test the full detection with mock data
    print("\n=== Testing full detection with mock data ===")
    
    # Create a mock tensor similar to what we have in the test
    mask = torch.zeros(128, 128, 27)
    mask[10:20, 10:20, 0] = 1.0
    mask[30:40, 30:40, 5] = 1.0
    mask[50:60, 50:60, 10] = 1.0
    
    print(f"Mock mask shape: {mask.shape}")
    print(f"Mock mask unique values: {torch.unique(mask)}")
    
    # Convert to numpy and test our channel detection logic
    mask_array = mask.numpy()
    mask_shapes = [mask_array.shape]
    mask_channels = []
    
    # Apply the same logic as in our function
    if len(mask_array.shape) == 3:
        if mask_array.shape[0] < min(mask_array.shape[1], mask_array.shape[2]):
            # (C, H, W) format
            mask_channels.append(mask_array.shape[0])
            print(f"Detected as (C, H, W) format with {mask_array.shape[0]} channels")
        else:
            # (H, W, C) format
            mask_channels.append(mask_array.shape[2])
            print(f"Detected as (H, W, C) format with {mask_array.shape[2]} channels")
    
    max_channels = max(mask_channels) if mask_channels else 1
    print(f"Max channels detected: {max_channels}")
    
    # Test with our analysis function
    unique_values = [0, 1]  # We know this from the mock data
    result = _analyze_class_distribution(unique_values, max_channels, "arcade")
    print(f"Final result: {result}")
    
    if result['class_type'] == 'semantic_onehot':
        print("✅ Mock data test passed!")
    else:
        print(f"❌ Mock data test failed: expected semantic_onehot, got {result['class_type']}")

if __name__ == "__main__":
    debug_detection()
