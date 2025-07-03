#!/usr/bin/env python3
"""
Test script to verify DeepResUNet with attention architecture fix.
"""
import sys
import os

# Add paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'ml'))

import torch
import torch.nn as nn

def test_deep_resunet_attention():
    """Test that DeepResUNet with attention doesn't have channel mismatch"""
    print("Testing DeepResUNet with attention...")
    
    try:
        # Import after path setup
        from ml.training.models.resunet_model import DeepResUNet
        
        # Test configurations
        test_configs = [
            {'n_channels': 1, 'n_classes': 1, 'bilinear': False, 'use_attention': True},
            {'n_channels': 3, 'n_classes': 1, 'bilinear': False, 'use_attention': True},
            {'n_channels': 1, 'n_classes': 1, 'bilinear': True, 'use_attention': True},
        ]
        
        for i, config in enumerate(test_configs):
            print(f"\nTest {i+1}: {config}")
            
            # Create model
            model = DeepResUNet(**config)
            model.eval()
            
            # Test input (batch_size=2, channels=n_channels, height=256, width=256)
            test_input = torch.randn(2, config['n_channels'], 256, 256)
            
            print(f"  Input shape: {test_input.shape}")
            
            # Forward pass
            with torch.no_grad():
                output = model(test_input)
                
            print(f"  Output shape: {output.shape}")
            print(f"  ✅ Success - no channel mismatch!")
            
            # Check output dimensions
            expected_shape = (2, config['n_classes'], 256, 256)
            if output.shape == expected_shape:
                print(f"  ✅ Output shape matches expected: {expected_shape}")
            else:
                print(f"  ❌ Output shape mismatch. Expected: {expected_shape}, Got: {output.shape}")
        
        print(f"\n🎉 All tests passed! DeepResUNet attention architecture is fixed.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_deep_resunet_attention()
