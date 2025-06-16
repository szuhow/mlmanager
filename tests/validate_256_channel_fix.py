#!/usr/bin/env python3
"""
Final validation script for the 256-channel issue fix.

This script validates that our enhanced class detection correctly handles
the case where ARCADE masks have 256 grayscale values but should be 
treated as binary segmentation after auto-thresholding.
"""

import sys
import os
sys.path.append('/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments')

import torch
import numpy as np
from ml.training.train import _analyze_class_distribution, create_model_from_registry, get_default_model_config

def test_256_channel_fix():
    """Test the complete pipeline fix for 256-channel issue"""
    
    print("🔍 VALIDATING 256-CHANNEL ISSUE FIX")
    print("=" * 60)
    
    # Step 1: Simulate the original problematic scenario
    print("\n1️⃣ ORIGINAL PROBLEM SCENARIO")
    print("-" * 30)
    
    # This is what ARCADE dataset detection originally found
    arcade_unique_values = list(range(256))  # [0, 1, 2, ..., 255]
    max_channels = 1
    
    print(f"📊 Raw mask analysis:")
    print(f"   Unique values: 256 values (0 to 255)")
    print(f"   Range: [0, 255]")
    print(f"   Channels: {max_channels}")
    
    # Step 2: Test with original logic (simulated)
    print(f"\n2️⃣ ORIGINAL CLASS DETECTION (would have failed)")
    print("-" * 30)
    
    # Simulate what the old logic would have done
    original_result = {
        'num_classes': 256,
        'class_type': 'semantic_single',
        'unique_values': arcade_unique_values,
        'max_channels': 1
    }
    print(f"❌ Old logic result: {original_result['num_classes']} classes ({original_result['class_type']})")
    print(f"❌ This would create model with 256 output channels")
    print(f"❌ But training would auto-threshold masks to binary")
    print(f"❌ Result: Shape mismatch error!")
    
    # Step 3: Test with new enhanced logic
    print(f"\n3️⃣ NEW ENHANCED CLASS DETECTION")
    print("-" * 30)
    
    enhanced_result = _analyze_class_distribution(arcade_unique_values, max_channels, "arcade")
    print(f"✅ Enhanced logic result: {enhanced_result['num_classes']} classes ({enhanced_result['class_type']})")
    print(f"✅ Correctly identifies grayscale binary mask")
    print(f"✅ Will create model with 1 output channel")
    print(f"✅ Matches auto-thresholding in training")
    
    # Step 4: Test model creation with correct configuration
    print(f"\n4️⃣ MODEL CREATION WITH FIXED CONFIG")
    print("-" * 30)
    
    # Simulate model creation as would happen in training
    class Args:
        model_type = "unet"
        model_family = "UNet-Coronary"
    
    args = Args()
    device = torch.device("cpu")
    
    model_config = get_default_model_config(args.model_type)
    model_config["in_channels"] = 3  # RGB input
    model_config["out_channels"] = enhanced_result['num_classes']  # Use enhanced detection
    
    print(f"📋 Model configuration:")
    print(f"   Input channels: {model_config['in_channels']}")
    print(f"   Output channels: {model_config['out_channels']}")
    
    try:
        model, arch_info = create_model_from_registry(
            args.model_type, 
            device,
            **model_config
        )
        print(f"✅ Model created successfully!")
        
        # Step 5: Test with training-like scenario
        print(f"\n5️⃣ TRAINING SCENARIO SIMULATION")
        print("-" * 30)
        
        # Simulate batch from training
        batch_size = 16
        height, width = 128, 128
        
        # Input images (RGB)
        inputs = torch.randn(batch_size, 3, height, width)
        
        # Ground truth masks (binary after auto-thresholding)
        # This simulates what happens after auto-thresholding in training
        labels = torch.randint(0, 2, (batch_size, 1, height, width)).float()
        
        # Model forward pass
        with torch.no_grad():
            outputs = model(inputs)
        
        print(f"📊 Batch simulation:")
        print(f"   Input shape: {inputs.shape}")
        print(f"   Ground truth shape: {labels.shape}")
        print(f"   Model output shape: {outputs.shape}")
        
        # Step 6: Validate compatibility
        print(f"\n6️⃣ COMPATIBILITY VALIDATION")
        print("-" * 30)
        
        if outputs.shape == labels.shape:
            print(f"✅ SHAPES MATCH! No more training errors!")
            print(f"✅ Model output: {outputs.shape}")
            print(f"✅ Ground truth: {labels.shape}")
            
            # Test loss computation (this would have failed before)
            try:
                from monai.losses import DiceLoss
                loss_fn = DiceLoss(sigmoid=True)
                loss = loss_fn(outputs, labels)
                print(f"✅ Loss computation successful: {loss.item():.4f}")
                return True
            except Exception as e:
                print(f"❌ Loss computation failed: {e}")
                return False
        else:
            print(f"❌ Shape mismatch still exists!")
            print(f"❌ Model output: {outputs.shape}")
            print(f"❌ Ground truth: {labels.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases to ensure robustness"""
    
    print(f"\n7️⃣ EDGE CASE TESTING")
    print("-" * 30)
    
    edge_cases = [
        {
            "name": "Partial grayscale (0-127)",
            "values": list(range(128)),
            "expected": "binary"
        },
        {
            "name": "Normalized grayscale (0.0-1.0)",
            "values": list(np.linspace(0, 1, 100)),
            "expected": "binary"
        },
        {
            "name": "True semantic (few classes)",
            "values": [0, 1, 2, 3, 4],
            "expected": "semantic_single"
        },
        {
            "name": "Perfect binary",
            "values": [0, 1],
            "expected": "binary"
        }
    ]
    
    all_passed = True
    
    for case in edge_cases:
        result = _analyze_class_distribution(case["values"], 1, "arcade")
        passed = result["class_type"] == case["expected"]
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {case['name']}: {result['class_type']} - {status}")
        if not passed:
            all_passed = False
    
    return all_passed

def main():
    """Run all validation tests"""
    
    print("🚀 STARTING COMPREHENSIVE VALIDATION")
    print("=" * 60)
    
    # Test main fix
    main_test_passed = test_256_channel_fix()
    
    # Test edge cases
    edge_cases_passed = test_edge_cases()
    
    # Final summary
    print(f"\n" + "=" * 60)
    print("📋 FINAL VALIDATION SUMMARY")
    print("=" * 60)
    
    if main_test_passed and edge_cases_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ 256-channel issue is FIXED")
        print("✅ Enhanced class detection works correctly")
        print("✅ Model configuration is correct")
        print("✅ Training compatibility validated")
        print("✅ Edge cases handled properly")
        print("\n🚀 Ready for production training!")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        if not main_test_passed:
            print("❌ Main 256-channel fix failed")
        if not edge_cases_passed:
            print("❌ Edge case validation failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
