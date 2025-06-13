#!/usr/bin/env python3
"""
Simple test to verify create_model_from_registry function works with unet
"""

import os
import sys
import django

# Setup Django
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.testing')
django.setup()

import torch
from ml.training.train import create_model_from_registry

def test_unet_creation():
    """Test that we can create a unet model without errors"""
    print("Testing unet model creation...")
    
    try:
        device = torch.device("cpu")  # Use CPU to avoid CUDA issues
        
        print("Creating model with type 'unet'...")
        model, arch_info = create_model_from_registry(
            model_type='unet',
            device=device,
            spatial_dims=2,
            in_channels=1,
            out_channels=1
        )
        
        print(f"✅ Model created successfully!")
        print(f"Architecture: {arch_info.display_name}")
        print(f"Framework: {arch_info.framework}")
        print(f"Model type: {type(model).__name__}")
        
        # Test a simple forward pass
        test_input = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ Forward pass successful! Input: {test_input.shape}, Output: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_unet_creation()
    if success:
        print("🎉 Test passed! UNet creation works correctly.")
        sys.exit(0)
    else:
        print("💥 Test failed!")
        sys.exit(1)
