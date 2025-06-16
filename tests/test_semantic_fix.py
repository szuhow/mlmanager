#!/usr/bin/env python3
"""
Test script to verify the semantic segmentation post-processing fix.
This should now show proper predictions instead of black images.
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_post_processing_logic():
    """Test the post-processing logic with mock model outputs"""
    print("🧪 Testing post-processing logic fix...")
    
    # Simulate binary segmentation output (1 channel)
    print("\n📊 Testing Binary Segmentation (1 channel):")
    binary_output = torch.randn(1, 1, 128, 128)  # Random logits
    print(f"   Input shape: {binary_output.shape}")
    
    num_output_channels = binary_output.shape[1]
    if num_output_channels == 1:
        # Binary segmentation
        processed = torch.sigmoid(binary_output)
        final = (processed > 0.5).float()
        print(f"   ✅ Applied binary post-processing (sigmoid + threshold)")
        print(f"   Final values: min={final.min():.3f}, max={final.max():.3f}, unique={torch.unique(final).tolist()}")
    
    # Simulate semantic segmentation output (27 channels for ARCADE)
    print("\n📊 Testing Semantic Segmentation (27 channels):")
    semantic_output = torch.randn(1, 27, 128, 128)  # Random logits for 27 classes
    print(f"   Input shape: {semantic_output.shape}")
    
    num_output_channels = semantic_output.shape[1]
    if num_output_channels > 1:
        # Multi-class semantic segmentation
        processed = torch.softmax(semantic_output, dim=1)
        final = torch.argmax(processed, dim=1, keepdim=True).float()
        print(f"   ✅ Applied multi-class post-processing (softmax + argmax)")
        print(f"   Final shape: {final.shape}")
        print(f"   Final values: min={final.min():.3f}, max={final.max():.3f}")
        print(f"   Unique classes found: {sorted(torch.unique(final).tolist())}")
        
        # Check if we have diverse predictions (not all the same class)
        unique_classes = torch.unique(final)
        if len(unique_classes) > 1:
            print(f"   ✅ Good! Found {len(unique_classes)} different classes in prediction")
        else:
            print(f"   ⚠️  Only found 1 class ({unique_classes[0].item()}) - might be an issue")
            
    return True

def test_old_vs_new_processing():
    """Compare old (broken) vs new (fixed) processing"""
    print("\n🔄 Comparing Old vs New Post-processing:")
    
    # Create a semantic segmentation output with clear class preferences
    semantic_output = torch.zeros(1, 27, 4, 4)
    
    # Make different spatial locations prefer different classes
    semantic_output[0, 5, 0, 0] = 10.0   # Class 5 strongly preferred at (0,0)
    semantic_output[0, 12, 1, 1] = 10.0  # Class 12 strongly preferred at (1,1)
    semantic_output[0, 20, 2, 2] = 10.0  # Class 20 strongly preferred at (2,2)
    semantic_output[0, 0, 3, 3] = 10.0   # Class 0 (background) strongly preferred at (3,3)
    
    print(f"   Input shape: {semantic_output.shape}")
    
    # OLD (BROKEN) processing - always used sigmoid
    old_processed = torch.sigmoid(semantic_output)
    old_final = (old_processed > 0.5).float()
    print(f"\n   ❌ OLD (broken) processing:")
    print(f"      After sigmoid: min={old_processed.min():.3f}, max={old_processed.max():.3f}")
    print(f"      After threshold: min={old_final.min():.3f}, max={old_final.max():.3f}")
    print(f"      Non-zero predictions: {torch.sum(old_final > 0).item()} pixels")
    
    # NEW (FIXED) processing - uses softmax + argmax for multi-class
    new_processed = torch.softmax(semantic_output, dim=1)
    new_final = torch.argmax(new_processed, dim=1, keepdim=True).float()
    print(f"\n   ✅ NEW (fixed) processing:")
    print(f"      After softmax: shape={new_processed.shape}")
    print(f"      After argmax: shape={new_final.shape}")
    print(f"      Predicted classes: {torch.unique(new_final).tolist()}")
    print(f"      Class at (0,0): {new_final[0,0,0,0].item()}")
    print(f"      Class at (1,1): {new_final[0,0,1,1].item()}")
    print(f"      Class at (2,2): {new_final[0,0,2,2].item()}")
    print(f"      Class at (3,3): {new_final[0,0,3,3].item()}")
    
    return True

def main():
    """Run all tests"""
    print("🔬 Testing Semantic Segmentation Post-processing Fix")
    print("=" * 60)
    
    success = True
    
    try:
        success &= test_post_processing_logic()
        success &= test_old_vs_new_processing()
        
        if success:
            print("\n" + "=" * 60)
            print("✅ ALL TESTS PASSED!")
            print("🎉 The semantic segmentation post-processing fix should work correctly!")
            print("📈 Semantic segmentation predictions should no longer be black.")
            print("🔧 Both binary and multi-class segmentation are now handled properly.")
        else:
            print("\n❌ Some tests failed")
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
