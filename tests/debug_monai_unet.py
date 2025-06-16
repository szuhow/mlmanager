#!/usr/bin/env python3
"""
Debug MONAI UNet output channels issue
"""

import torch
from monai.networks.nets import UNet as MonaiUNet

def test_monai_unet_output_channels():
    """Test MONAI UNet with different configurations"""
    
    print("=" * 60)
    print("Testing MONAI UNet Output Channels")
    print("=" * 60)
    
    # Test configuration 1: Binary segmentation (should output 1 channel)
    print("\n1. Binary segmentation configuration:")
    model1 = MonaiUNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,  # This should be the output channels
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    
    # Test input
    test_input = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        output1 = model1(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output1.shape}")
    print(f"Expected: [2, 1, 128, 128] (binary segmentation)")
    print(f"Actual: {list(output1.shape)}")
    print(f"Output channels correct: {output1.shape[1] == 1}")
    
    # Test configuration 2: Multi-class segmentation
    print("\n2. Multi-class segmentation configuration:")
    model2 = MonaiUNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=27,  # 27 classes
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    
    with torch.no_grad():
        output2 = model2(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output2.shape}")
    print(f"Expected: [2, 27, 128, 128] (multi-class segmentation)")
    print(f"Actual: {list(output2.shape)}")
    print(f"Output channels correct: {output2.shape[1] == 27}")
    
    # Test configuration 3: Check what happens with wrong config
    print("\n3. Debugging configuration issue:")
    print("Model 1 detailed structure:")
    
    # Check the final layer
    print(f"Model 1 final layer: {model1.out}")
    if hasattr(model1.out, 'conv'):
        print(f"Final conv layer: {model1.out.conv}")
        print(f"Final conv out_channels: {model1.out.conv.out_channels}")
    
    return model1, model2

if __name__ == "__main__":
    test_monai_unet_output_channels()
