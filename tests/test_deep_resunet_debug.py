#!/usr/bin/env python3
"""
Debug test for DeepResUNet with attention - detailed channel debugging
"""

import torch
import torch.nn as nn
import sys
import os

# Add the ml directory to Python path
sys.path.insert(0, '/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments/ml')

from training.models.resunet_model import DeepResUNet

def debug_deep_resunet_attention():
    """Test DeepResUNet with attention and debug channel sizes"""
    print("=== DeepResUNet with Attention Debug ===")
    
    # Test configurations
    configs = [
        (3, 1, 224, 224),  # RGB input
        (1, 1, 224, 224),  # Grayscale input
        (3, 3, 224, 224),  # RGB with 3 classes
    ]
    
    for in_channels, out_channels, h, w in configs:
        print(f"\nTesting config: in_channels={in_channels}, out_channels={out_channels}, size={h}x{w}")
        
        try:
            # Create model with attention
            model = DeepResUNet(
                n_channels=in_channels,
                n_classes=out_channels,
                use_attention=True,
                bilinear=False  # Use transposed convolutions
            )
            
            # Create input tensor
            x = torch.randn(1, in_channels, h, w)
            print(f"Input shape: {x.shape}")
            
            # Forward pass with debug
            model.eval()
            with torch.no_grad():
                # Add some debug prints to understand the flow
                print("\nForward pass...")
                try:
                    output = model(x)
                    print(f"✓ Success! Output shape: {output.shape}")
                except Exception as e:
                    print(f"✗ Error: {e}")
                    print(f"Error type: {type(e)}")
                    
                    # Try to get more info about where it failed
                    if "size mismatch" in str(e) or "channel" in str(e).lower():
                        print("Channel mismatch detected!")
                        # Try to trace the issue
                        debug_forward_pass(model, x)
                    
        except Exception as e:
            print(f"✗ Model creation failed: {e}")

def debug_forward_pass(model, x):
    """Debug forward pass step by step"""
    print("\n--- Step-by-step debug ---")
    
    try:
        # Encoder
        print("Encoder...")
        x1 = model.inc(x)
        print(f"After inc: {x1.shape}")
        
        x2 = model.down1(x1)
        print(f"After down1: {x2.shape}")
        
        x3 = model.down2(x2)
        print(f"After down2: {x3.shape}")
        
        x4 = model.down3(x3)
        print(f"After down3: {x4.shape}")
        
        x5 = model.down4(x4)
        print(f"After down4: {x5.shape}")
        
        x6 = model.down5(x5)
        print(f"After down5: {x6.shape}")
        
        # Decoder with attention
        print("\nDecoder with attention...")
        
        # First upsampling
        print("Up1...")
        x5_att = model.att5(x6, x5)
        print(f"x5_att shape: {x5_att.shape}")
        print(f"x6 shape: {x6.shape}")
        
        # Debug ResUp manually
        print("ResUp debug...")
        x6_up = model.up1.up(x6)
        print(f"x6 upsampled: {x6_up.shape}")
        
        # This is where the concatenation happens
        print(f"Concatenating: {x6_up.shape} + {x5_att.shape}")
        concat_result = torch.cat([x6_up, x5_att], dim=1)
        print(f"Concatenated: {concat_result.shape}")
        
        # Now through the conv layers
        print("Through conv layers...")
        result = model.up1.conv(concat_result)
        print(f"After conv: {result.shape}")
        
    except Exception as e:
        print(f"Debug failed at step: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_deep_resunet_attention()
