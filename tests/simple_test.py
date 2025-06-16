#!/usr/bin/env python3
"""
Simple test to isolate the channel detection issue
"""

import sys
import numpy as np
import torch

sys.path.insert(0, '/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments')

# Test the channel detection logic directly
mask = torch.zeros(128, 128, 27)
mask[10:20, 10:20, 0] = 1.0
mask[30:40, 30:40, 5] = 1.0

print(f"Original mask shape: {mask.shape}")

# Convert to numpy (same as in the detection function)
mask_array = mask.numpy()
print(f"Numpy mask shape: {mask_array.shape}")

# Test our channel detection logic
mask_shapes = [mask_array.shape]
mask_channels = []

if len(mask_array.shape) == 3:
    # Either (C, H, W) or (H, W, C)
    if mask_array.shape[0] < min(mask_array.shape[1], mask_array.shape[2]):
        # Likely (C, H, W) format
        mask_channels.append(mask_array.shape[0])
        print(f"Detected as (C, H, W) format with {mask_array.shape[0]} channels")
    else:
        # Likely (H, W, C) format  
        mask_channels.append(mask_array.shape[2])
        print(f"Detected as (H, W, C) format with {mask_array.shape[2]} channels")

print(f"Detected channels: {mask_channels}")
max_channels = max(mask_channels) if mask_channels else 1
print(f"Max channels: {max_channels}")

# Now test the analysis function
from ml.training.train import _analyze_class_distribution
result = _analyze_class_distribution([0, 1], max_channels, "arcade")
print(f"Final result: {result}")
