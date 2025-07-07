#!/usr/bin/env python3
"""
Debug checkpoint format to understand the structure
"""
import os
import sys
import torch

# Add ML to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml'))

def debug_checkpoint():
    """Debug checkpoint structure"""
    
    # Find model weights
    weights_paths = []
    for root, dirs, files in os.walk("data/mlflow"):
        for file in files:
            if file.endswith(('.pth', '.pt')):
                weights_paths.append(os.path.join(root, file))
    
    if not weights_paths:
        print("❌ No model weights found")
        return
    
    model_path = weights_paths[0]
    print(f"Debugging checkpoint: {model_path}")
    
    try:
        # Load checkpoint with CPU mapping
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        print(f"\nCheckpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Check each key
            for key in checkpoint.keys():
                value = checkpoint[key]
                print(f"\n{key}:")
                print(f"  Type: {type(value)}")
                
                if isinstance(value, dict):
                    print(f"  Dict keys (first 5): {list(value.keys())[:5]}")
                    if len(value.keys()) > 5:
                        print(f"  ... and {len(value.keys()) - 5} more")
                elif hasattr(value, 'shape'):
                    print(f"  Shape: {value.shape}")
                else:
                    print(f"  Value: {str(value)[:100]}")
            
            # Look for the actual model state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"\nmodel_state_dict keys (first 10):")
                for i, key in enumerate(list(state_dict.keys())[:10]):
                    print(f"  {key}: {state_dict[key].shape if hasattr(state_dict[key], 'shape') else type(state_dict[key])}")
                    
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 Debugging Checkpoint Format")
    print("=" * 50)
    debug_checkpoint()
