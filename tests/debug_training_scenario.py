#!/usr/bin/env python3
"""
Test the exact training scenario that causes the 256 channel issue
"""

import sys
import os
sys.path.append('/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments')

import torch
from ml.training.train import create_model_from_registry, get_default_model_config

def test_training_scenario():
    """Test the exact scenario from training that causes 256 channel issue"""
    
    print("=" * 60)
    print("Testing Training Scenario")
    print("=" * 60)
    
    # Simulate arguments from training
    class Args:
        model_type = "unet"
        model_family = "UNet-Coronary"
    
    args = Args()
    device = torch.device("cpu")
    
    print(f"1. Getting default model config:")
    model_config = get_default_model_config(args.model_type)
    print(f"Initial model config: {model_config}")
    
    # Simulate class detection results (256-value grayscale that will be thresholded)
    print(f"\n2. Simulating class detection (256-value grayscale binary):")
    class_info = {
        'num_classes': 256,  # Before our fix
        'class_type': 'semantic_single',
        'unique_values': list(range(256)),  # 0, 1, 2, ..., 255
        'max_channels': 1
    }
    
    # Test our new logic
    from ml.training.train import _analyze_class_distribution
    print(f"Before fix: {class_info}")
    
    # Test with 256 grayscale values (0-255) - this should be detected as binary
    unique_values_256 = list(range(256))  # [0, 1, 2, 3, ..., 255]
    class_info_fixed = _analyze_class_distribution(unique_values_256, 1, "arcade")
    print(f"After fix: {class_info_fixed}")
    
    # Use the corrected class info
    class_info = class_info_fixed
    
    # Simulate the dynamic configuration from training
    input_channels = 3  # RGB input detected
    model_config["in_channels"] = input_channels
    
    if class_info:
        output_channels = class_info['num_classes']
        print(f"Detected {output_channels} output classes ({class_info['class_type']})")
        
        if class_info['class_type'] == 'semantic_onehot':
            model_config["out_channels"] = class_info['max_channels']
            print(f"Using {class_info['max_channels']} output channels for one-hot semantic segmentation")
        else:
            model_config["out_channels"] = output_channels
            print(f"Using {output_channels} output channels for {class_info['class_type']} segmentation")
    
    print(f"Final model configuration: {model_config}")
    
    print(f"\n3. Creating model with exact training configuration:")
    try:
        model, arch_info = create_model_from_registry(
            args.model_type, 
            device,
            **model_config
        )
        
        print(f"Model created successfully!")
        print(f"Architecture: {arch_info.display_name if hasattr(arch_info, 'display_name') else 'Unknown'}")
        
        # Test with exact dimensions from error
        print(f"\n4. Testing with error scenario dimensions:")
        test_input = torch.randn(2, 3, 128, 128)  # Input: [batch=2, channels=3, height=128, width=128]
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected: [2, 1, 128, 128] (binary segmentation)")
        
        if output.shape[1] == 256:
            print(f"❌ REPRODUCED ERROR: Model outputs 256 channels!")
            print(f"This is the same error as in training!")
            
            # Debug the model structure
            print(f"\n5. Debugging model structure:")
            print(f"Model type: {type(model)}")
            if hasattr(model, 'out'):
                print(f"Output layer: {model.out}")
            elif hasattr(model, 'segmentation_head'):
                print(f"Segmentation head: {model.segmentation_head}")
            elif hasattr(model, 'outc'):
                print(f"Output conv: {model.outc}")
            
            return False
        elif output.shape[1] == 1:
            print(f"✅ Model outputs correct 1 channel")
            return True
        else:
            print(f"❌ Model outputs {output.shape[1]} channels (unexpected)")
            return False
            
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_scenario()
    
    if not success:
        print("\n🔍 We need to identify why the model outputs 256 channels during training!")
    else:
        print("\n✅ Model works correctly in this test!")
        print("The issue might be in:")
        print("  1. Different model type being used during training")
        print("  2. Different configuration being passed")
        print("  3. Model state being modified after creation")
        print("  4. Different dataset loader affecting model structure")
