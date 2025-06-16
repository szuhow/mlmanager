#!/usr/bin/env python3
"""
Debug the exact model configuration flow during training
"""

import sys
import os
sys.path.append('/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments')

from ml.training.train import get_default_model_config, create_model_from_registry
from ml.utils.architecture_registry import registry as architecture_registry
import torch

def debug_model_configuration_flow():
    """Debug the model configuration flow that's used during training"""
    
    print("=" * 60)
    print("Debugging Model Configuration Flow")
    print("=" * 60)
    
    model_type = "unet"
    device = torch.device("cpu")
    
    print(f"\n1. Getting default model config for '{model_type}':")
    model_config = get_default_model_config(model_type)
    print(f"Default config: {model_config}")
    
    print(f"\n2. Simulating dynamic class detection (binary segmentation):")
    # Simulate what happens during class detection
    input_channels = 3  # RGB input
    output_channels = 1  # Binary segmentation
    
    model_config["in_channels"] = input_channels
    model_config["out_channels"] = output_channels
    
    print(f"Updated config: {model_config}")
    
    print(f"\n3. Creating model with configuration:")
    try:
        model, arch_info = create_model_from_registry(
            model_type, 
            device,
            **model_config
        )
        
        print(f"Model created successfully!")
        print(f"Architecture: {arch_info.display_name if hasattr(arch_info, 'display_name') else 'Unknown'}")
        
        # Test the model
        test_input = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"\n4. Testing model output:")
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: [2, 1, 128, 128]")
        print(f"Output channels match expected: {output.shape[1] == 1}")
        
        if output.shape[1] != 1:
            print(f"❌ ERROR: Model outputs {output.shape[1]} channels instead of 1!")
            return False
        else:
            print(f"✅ SUCCESS: Model outputs correct number of channels")
            return True
            
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_architecture_registry():
    """Debug the architecture registry to see what's available"""
    
    print("\n" + "=" * 60)
    print("Debugging Architecture Registry")
    print("=" * 60)
    
    print(f"Available architectures: {list(architecture_registry._architectures.keys())}")
    
    unet_info = architecture_registry._architectures.get('unet')
    if unet_info:
        print(f"\nUNet architecture info:")
        print(f"  Display name: {unet_info.display_name}")
        print(f"  Default config: {unet_info.default_config}")
    else:
        print("\n❌ UNet not found in registry!")

if __name__ == "__main__":
    debug_architecture_registry()
    success = debug_model_configuration_flow()
    
    if not success:
        print("\n🔍 Additional debugging needed - the model configuration flow has issues!")
    else:
        print("\n✅ Model configuration flow works correctly!")
