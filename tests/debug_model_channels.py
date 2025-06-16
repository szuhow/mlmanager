#!/usr/bin/env python3
"""
Simple debug script to understand why binary segmentation gets 256 output channels
Focuses on the model creation logic without requiring actual ARCADE dataset
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set Django settings if needed
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

# Configure Django if running as standalone script
try:
    import django
    django.setup()
except:
    pass

def test_model_creation_with_different_configs():
    """Test model creation with different output channel configurations"""
    print("="*60)
    print("TESTING MODEL CREATION WITH DIFFERENT OUTPUT CHANNELS")
    print("="*60)
    
    try:
        from ml.training.train import get_default_model_config, create_model_from_registry
        
        model_type = "unet"  # Default model type for ARCADE
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Test 1: Default configuration
        print(f"\n[TEST 1] Default model configuration...")
        default_config = get_default_model_config(model_type)
        print(f"  - Default config: {default_config}")
        
        # Test 2: Binary segmentation (1 output channel)
        print(f"\n[TEST 2] Binary segmentation configuration...")
        binary_config = default_config.copy()
        binary_config["in_channels"] = 1
        binary_config["out_channels"] = 1  # Binary segmentation should have 1 output channel
        print(f"  - Binary config: {binary_config}")
        
        model_binary, arch_info_binary = create_model_from_registry(
            model_type, 
            device,
            **binary_config
        )
        
        # Test with sample data
        test_input = torch.randn(1, 1, 128, 128).to(device)  # Single channel input
        with torch.no_grad():
            output_binary = model_binary(test_input)
            print(f"  - Input shape: {test_input.shape}")
            print(f"  - Output shape: {output_binary.shape}")
            print(f"  - Expected: torch.Size([1, 1, 128, 128])")
            print(f"  - Match: {output_binary.shape[1] == 1}")
        
        # Test 3: Multi-class segmentation (27 output channels like ARCADE semantic)
        print(f"\n[TEST 3] Multi-class segmentation configuration...")
        multi_config = default_config.copy()
        multi_config["in_channels"] = 1
        multi_config["out_channels"] = 27  # ARCADE semantic segmentation has 27 classes
        print(f"  - Multi-class config: {multi_config}")
        
        model_multi, arch_info_multi = create_model_from_registry(
            model_type, 
            device,
            **multi_config
        )
        
        with torch.no_grad():
            output_multi = model_multi(test_input)
            print(f"  - Input shape: {test_input.shape}")
            print(f"  - Output shape: {output_multi.shape}")
            print(f"  - Expected: torch.Size([1, 27, 128, 128])")
            print(f"  - Match: {output_multi.shape[1] == 27}")
        
        # Test 4: Check what happens with no explicit config
        print(f"\n[TEST 4] Model creation with minimal config...")
        minimal_config = {"in_channels": 1}  # Only specify input channels
        print(f"  - Minimal config: {minimal_config}")
        
        model_minimal, arch_info_minimal = create_model_from_registry(
            model_type, 
            device,
            **minimal_config
        )
        
        with torch.no_grad():
            output_minimal = model_minimal(test_input)
            print(f"  - Input shape: {test_input.shape}")
            print(f"  - Output shape: {output_minimal.shape}")
            print(f"  - Actual output channels: {output_minimal.shape[1]}")
        
        # Test 5: Test the architecture registry defaults
        print(f"\n[TEST 5] Check architecture registry defaults...")
        from ml.utils.architecture_registry import registry
        
        arch_info = registry.get_architecture(model_type)
        if arch_info:
            print(f"  - Registry key: {arch_info.key}")
            print(f"  - Display name: {arch_info.display_name}")
            print(f"  - Default config: {arch_info.default_config}")
            
            # Check if default config has problematic values
            default_out_channels = arch_info.default_config.get('out_channels', 'NOT_SET')
            default_channels = arch_info.default_config.get('channels', 'NOT_SET')
            print(f"  - Default out_channels: {default_out_channels}")
            print(f"  - Default channels: {default_channels}")
            
            # The issue might be in the 'channels' parameter which defines the feature channels
            if 'channels' in arch_info.default_config:
                channels = arch_info.default_config['channels']
                if isinstance(channels, tuple) and len(channels) > 0:
                    last_channel = channels[-1]  # This might be what's causing the issue
                    print(f"  - Last feature channel: {last_channel}")
                    if last_channel == 256:
                        print(f"  ⚠️  POTENTIAL ISSUE: Last feature channel is 256!")
                        print(f"     This might be affecting the final output layer.")
        
        # Summary
        print(f"\n[SUMMARY] Test Results:")
        print(f"  - Binary model output channels: {output_binary.shape[1]}")
        print(f"  - Multi-class model output channels: {output_multi.shape[1]}")
        print(f"  - Minimal config model output channels: {output_minimal.shape[1]}")
        
        if output_binary.shape[1] == 1 and output_multi.shape[1] == 27:
            print(f"\n✅ SUCCESS: Models are created with correct output channels")
        else:
            print(f"\n❌ ISSUE: Models not created with expected output channels")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_creation_with_different_configs()
    sys.exit(0 if success else 1)
