#!/usr/bin/env python3
"""
Test intelligent binary detection system for masks
"""
import torch
import numpy as np

def test_binary_detection():
    """Test the intelligent binary detection logic we implemented"""
    
    # Test case 1: Perfect binary masks (0, 1)
    print("🧪 Test 1: Perfect binary masks (0, 1)")
    labels_binary = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1]).float()
    unique_labels = torch.unique(labels_binary)
    unique_values_array = unique_labels.cpu().numpy()
    print(f"   Raw unique values: {unique_values_array}")
    
    if len(unique_labels) == 2:
        min_val, max_val = unique_values_array.min(), unique_values_array.max()
        if (min_val == 0 and max_val == 1):
            print(f"   ✅ Binary segmentation confirmed (0,1 format)")
            positive_ratio = (labels_binary == 1).float().mean().item()
        elif (min_val == 0 and max_val == 255):
            print(f"   ✅ Binary segmentation confirmed (0,255 format - will be normalized)")
            positive_ratio = (labels_binary == 255).float().mean().item()
        else:
            print(f"   ✅ Binary segmentation confirmed (custom {min_val},{max_val} format)")
            positive_ratio = (labels_binary == max_val).float().mean().item()
        print(f"   📈 Class distribution: {positive_ratio:.2%} positive, {1-positive_ratio:.2%} background")
    
    # Test case 2: 8-bit binary masks (0, 255)
    print("\n🧪 Test 2: 8-bit binary masks (0, 255)")
    labels_8bit = torch.tensor([0, 0, 255, 255, 0, 255, 0, 255]).float()
    unique_labels = torch.unique(labels_8bit)
    unique_values_array = unique_labels.cpu().numpy()
    print(f"   Raw unique values: {unique_values_array}")
    
    if len(unique_labels) == 2:
        min_val, max_val = unique_values_array.min(), unique_values_array.max()
        if (min_val == 0 and max_val == 1):
            print(f"   ✅ Binary segmentation confirmed (0,1 format)")
            positive_ratio = (labels_8bit == 1).float().mean().item()
        elif (min_val == 0 and max_val == 255):
            print(f"   ✅ Binary segmentation confirmed (0,255 format - will be normalized)")
            positive_ratio = (labels_8bit == 255).float().mean().item()
        else:
            print(f"   ✅ Binary segmentation confirmed (custom {min_val},{max_val} format)")
            positive_ratio = (labels_8bit == max_val).float().mean().item()
        print(f"   📈 Class distribution: {positive_ratio:.2%} positive, {1-positive_ratio:.2%} background")
    
    # Test case 3: Grayscale masks (many values) - this was the problem case
    print("\n🧪 Test 3: Grayscale masks (many values) - the problematic case")
    # Simulate the problematic case: normalized grayscale values
    labels_grayscale = torch.tensor([0.0, 0.00392157, 0.00784314, 0.01176471, 0.5, 0.7, 0.9, 1.0]).float()
    unique_labels = torch.unique(labels_grayscale)
    unique_values_array = unique_labels.cpu().numpy()
    print(f"   Raw unique values: {unique_values_array}")
    
    label_min, label_max = labels_grayscale.min().item(), labels_grayscale.max().item()
    
    if len(unique_labels) == 2:
        min_val, max_val = unique_values_array.min(), unique_values_array.max()
        if (min_val == 0 and max_val == 1):
            print(f"   ✅ Binary segmentation confirmed (0,1 format)")
        elif (min_val == 0 and max_val == 255):
            print(f"   ✅ Binary segmentation confirmed (0,255 format - will be normalized)")
        else:
            print(f"   ✅ Binary segmentation confirmed (custom {min_val},{max_val} format)")
    elif len(unique_labels) == 1:
        single_val = unique_values_array[0]
        if single_val == 0:
            print(f"   ⚠️  Single class detected (all background) - check data!")
        elif single_val == 1 or single_val == 255:
            print(f"   ⚠️  Single class detected (all foreground) - check data!")
        else:
            print(f"   ⚠️  Single class detected (all {single_val}) - check data!")
    else:
        # Check if it's grayscale values that need thresholding
        if label_max <= 255 and label_min >= 0 and len(unique_labels) > 2:
            # Probably grayscale masks that need binarization
            print(f"   📝 Grayscale mask detected ({len(unique_labels)} values)")
            print(f"   💡 Recommend thresholding: values > 127 → 1, else → 0")
            # Show value distribution for debugging
            if len(unique_labels) <= 10:
                print(f"   🔍 All values: {sorted(unique_values_array)}")
        else:
            print(f"   ⚠️  Multi-class ({len(unique_labels)} classes) - not binary segmentation")
    
    # Test case 4: Test normalization
    print("\n🧪 Test 4: Testing auto-normalization")
    labels_255 = torch.tensor([0, 127, 255, 0, 255, 128, 255]).float()
    print(f"   Original values: {torch.unique(labels_255).cpu().numpy()}")
    print(f"   Range: [{labels_255.min().item():.1f}, {labels_255.max().item():.1f}]")
    
    # Apply normalization logic
    if labels_255.max() > 1:
        if labels_255.max() <= 255 and labels_255.min() >= 0:
            print(f"   🔧 Auto-normalizing masks from [0-255] to [0-1] range")
            labels_255 = labels_255 / 255.0
            # Ensure binary values after normalization
            labels_255 = (labels_255 > 0.5).float()
            print(f"   ✅ Masks normalized to range [{labels_255.min().item():.1f}-{labels_255.max().item():.1f}]")
            print(f"   Final unique values: {torch.unique(labels_255).cpu().numpy()}")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    test_binary_detection()
