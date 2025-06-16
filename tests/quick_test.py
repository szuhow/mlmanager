#!/usr/bin/env python3
import torch
import numpy as np

# Test the issue
mask = torch.zeros(128, 128, 27)
mask[10:20, 10:20, 0] = 1.0
print(f"Mask shape: {mask.shape}")
print(f"Unique values: {torch.unique(mask)}")

# Convert and analyze
mask_array = mask.numpy()
shape = mask_array.shape

if len(shape) == 3:
    if shape[0] < min(shape[1], shape[2]):
        channels = shape[0]
        print(f"(C,H,W) format: {channels} channels")
    else:
        channels = shape[2]
        print(f"(H,W,C) format: {channels} channels")

print(f"Detected {channels} channels")

# Test analysis
if channels > 2:
    print("Would detect as semantic_onehot")
else:
    print("Would detect as binary")
