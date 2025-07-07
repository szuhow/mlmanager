#!/usr/bin/env python3
"""
Test fixed enhanced inference with automatic checkpoint format detection
"""
import os
import sys
import tempfile

# Add ML to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml'))

def test_enhanced_inference_with_format_detection():
    """Test enhanced inference with automatic format detection"""
    
    # Find model weights and test images
    weights_paths = []
    for root, dirs, files in os.walk("data/mlflow"):
        for file in files:
            if file.endswith(('.pth', '.pt')):
                weights_paths.append(os.path.join(root, file))
    
    test_images = []
    for root, dirs, files in os.walk("data/datasets"):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(root, file))
                break
    
    if not weights_paths or not test_images:
        print("❌ Missing model weights or test images")
        return False
    
    # Test with different checkpoint types
    checkpoint_types = {}
    for path in weights_paths[:3]:  # Test first 3
        if 'final_model' in path:
            checkpoint_types['final_model'] = path
        elif 'best_model' in path:
            checkpoint_types['best_model'] = path  
        elif 'epoch' in path:
            checkpoint_types['epoch_checkpoint'] = path
    
    input_image = test_images[0]
    
    for checkpoint_type, model_path in checkpoint_types.items():
        print(f"\n🧪 Testing {checkpoint_type}: {os.path.basename(model_path)}")
        
        try:
            from ml.utils.enhanced_inference import run_enhanced_inference
            
            with tempfile.TemporaryDirectory() as output_dir:
                config = {
                    'model_type': 'unet',
                    'resolution': 512,
                    'threshold': 0.5,
                    'post_processing': {
                        'morphological_ops': False,  # Disable to avoid sklearn issues
                        'noise_filtering': False,
                        'connected_components': False,
                        'edge_enhancement': False,
                        'test_time_augmentation': False
                    }
                }
                
                result = run_enhanced_inference(
                    model_path=model_path,
                    input_image_path=input_image,
                    output_dir=output_dir,
                    config=config
                )
                
                if result and result.get('success'):
                    print(f"✅ {checkpoint_type} inference succeeded")
                    print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
                    print(f"   Files: {list(result.get('files', {}).keys())}")
                    print(f"   Detected objects: {result.get('detected_objects_count', 0)}")
                else:
                    print(f"❌ {checkpoint_type} inference failed: {result.get('error_message', 'Unknown error')}")
                    
        except Exception as e:
            print(f"❌ {checkpoint_type} inference error: {e}")
    
    return True

def debug_checkpoint_format():
    """Debug checkpoint formats to understand structure"""
    
    weights_paths = []
    for root, dirs, files in os.walk("data/mlflow"):
        for file in files:
            if file.endswith(('.pth', '.pt')):
                weights_paths.append(os.path.join(root, file))
    
    print("\n🔍 Checkpoint Format Analysis:")
    
    for i, path in enumerate(weights_paths[:3]):
        print(f"\n{i+1}. {path}")
        
        try:
            import torch
            checkpoint = torch.load(path, map_location='cpu')
            
            print(f"   Type: {type(checkpoint)}")
            if isinstance(checkpoint, dict):
                keys = list(checkpoint.keys())
                print(f"   Keys: {keys}")
                
                # Check format indicators
                if 'model_state_dict' in checkpoint:
                    print("   ➜ NESTED FORMAT (model_state_dict)")
                elif 'state_dict' in checkpoint:
                    print("   ➜ NESTED FORMAT (state_dict)")
                elif any(key in checkpoint for key in ['model_metadata', 'training_args']):
                    print("   ➜ FULL CHECKPOINT FORMAT")
                elif any('conv' in key or 'fc' in key for key in keys[:5]):
                    print("   ➜ DIRECT STATE_DICT FORMAT")
                else:
                    print("   ➜ UNKNOWN FORMAT")
                    
        except Exception as e:
            print(f"   Error: {e}")

if __name__ == "__main__":
    print("🧪 Testing Enhanced Inference with Format Detection")
    print("=" * 60)
    
    # Debug checkpoint formats first
    debug_checkpoint_format()
    
    # Test enhanced inference
    print("\n📋 Testing Enhanced Inference")
    success = test_enhanced_inference_with_format_detection()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Testing completed!")
    else:
        print("⚠️ Some tests may have failed.")
