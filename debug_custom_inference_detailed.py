#!/usr/bin/env python3
"""
Debug custom inference in detail to find where it fails
"""
import os
import sys
sys.path.append('/app')
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_custom_inference():
    """Debug custom inference step by step"""
    print("🔍 Debugging Custom Inference Step by Step")
    print("=" * 60)
    
    model_path = "data/mlflow/307f67e76c6140c79d282e409c1f36c0/artifacts/final_model/weights/model.pth"
    
    try:
        print("Step 1: Loading checkpoint...")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        print(f"✅ Checkpoint loaded successfully")
        
        print("Step 2: Extracting state_dict and metadata...")
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            metadata = checkpoint.get('model_metadata', {})
            print(f"✅ Extracted state_dict with {len(state_dict)} keys")
            print(f"✅ Extracted metadata: {metadata}")
        else:
            print("❌ No model_state_dict found")
            return
            
        print("Step 3: Setting up model configuration...")
        model_type = metadata.get('model_architecture', 'unet')
        in_channels = metadata.get('input_channels', 3)
        out_channels = metadata.get('num_classes', 1)
        print(f"✅ Model config: type={model_type}, in={in_channels}, out={out_channels}")
        
        print("Step 4: Creating model...")
        from ml.training.train import create_model_from_registry, get_default_model_config
        
        model_config = get_default_model_config(model_type)
        model_config["in_channels"] = in_channels
        model_config["out_channels"] = out_channels
        print(f"✅ Model config prepared: {model_config}")
        
        model, arch_info = create_model_from_registry(model_type, 'cpu', **model_config)
        print(f"✅ Model created: {arch_info}")
        
        print("Step 5: Loading state dict...")
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✅ State dict loaded successfully with strict=True")
        except RuntimeError as e:
            print(f"❌ Strict loading failed: {e}")
            print("Trying with strict=False...")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys[:3]}...")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:3]}...")
                
        model.eval()
        print("✅ Model set to eval mode")
        
        print("Step 6: Testing transforms...")
        from ml.training.train import get_inference_transforms
        transforms = get_inference_transforms(image_size=(512, 512), use_original_size=False)
        print("✅ Transforms created")
        
        print("Step 7: Testing image loading...")
        test_image_path = "test_image.jpg"
        if os.path.exists(test_image_path):
            print(f"✅ Test image found: {test_image_path}")
            try:
                img = transforms(test_image_path)
                print(f"✅ Image transforms applied, shape: {img.shape}")
                
                img_batch = img.unsqueeze(0)  # Add batch dimension
                print(f"✅ Batch created, shape: {img_batch.shape}")
                
                print("Step 8: Running inference...")
                with torch.no_grad():
                    output = model(img_batch)
                    print(f"✅ Inference successful, output shape: {output.shape}")
                    
            except Exception as e:
                print(f"❌ Image processing failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"❌ Test image not found: {test_image_path}")
            
        print("\n🎉 Custom inference debugging completed successfully!")
        
    except Exception as e:
        print(f"❌ Custom inference failed at main level: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_custom_inference()
