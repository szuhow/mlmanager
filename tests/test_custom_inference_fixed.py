#!/usr/bin/env python3
"""
Test custom inference with proper device mapping
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add ML to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml'))

def test_custom_inference_direct():
    """Test custom inference directly"""
    
    # Find model weights and images
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
                break  # Just get one
    
    if not weights_paths or not test_images:
        print("❌ Missing files")
        return False
    
    model_path = weights_paths[0]
    input_image = test_images[0] 
    
    print(f"Testing custom inference directly:")
    print(f"Model: {model_path}")
    print(f"Image: {input_image}")
    
    try:
        from ml.utils.enhanced_inference import run_custom_inference
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'model_type': 'unet',
                'resolution': 512,
                'threshold': 0.5
            }
            
            # Test with explicit device mapping
            result = run_custom_inference(
                model_path=model_path,
                input_image_path=input_image,
                output_dir=temp_dir,
                config=config,
                device='cpu'
            )
            
            if result:
                print("✅ Custom inference succeeded")
                
                # List output files
                output_files = list(Path(temp_dir).glob("*"))
                print(f"Generated {len(output_files)} files:")
                for f in output_files:
                    print(f"  - {f.name} ({f.stat().st_size} bytes)")
                    
                return True
            else:
                print("❌ Custom inference failed")
                return False
                
    except Exception as e:
        print(f"❌ Custom inference error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_with_better_checkpoint():
    """Test with better checkpoint discovery"""
    
    # Find different types of checkpoints
    checkpoint_types = {
        'final_model': [],
        'best_model': [],
        'epoch_checkpoints': []
    }
    
    for root, dirs, files in os.walk("data/mlflow"):
        for file in files:
            if file.endswith(('.pth', '.pt')):
                full_path = os.path.join(root, file)
                if 'final_model' in root:
                    checkpoint_types['final_model'].append(full_path)
                elif 'best_model' in root:
                    checkpoint_types['best_model'].append(full_path)
                else:
                    checkpoint_types['epoch_checkpoints'].append(full_path)
    
    print("Available checkpoints:")
    for cp_type, paths in checkpoint_types.items():
        print(f"  {cp_type}: {len(paths)} files")
        if paths:
            print(f"    Example: {paths[0]}")
    
    # Try enhanced inference with a different checkpoint type
    test_paths = []
    if checkpoint_types['best_model']:
        test_paths.extend(checkpoint_types['best_model'][:1])
    if checkpoint_types['epoch_checkpoints']:
        test_paths.extend(checkpoint_types['epoch_checkpoints'][:1])
    if checkpoint_types['final_model']:
        test_paths.extend(checkpoint_types['final_model'][:1])
    
    # Find test image
    test_images = []
    for root, dirs, files in os.walk("data/datasets"):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(root, file))
                break
    
    if not test_images:
        print("❌ No test images found")
        return False
    
    input_image = test_images[0]
    
    for i, model_path in enumerate(test_paths):
        print(f"\n🧪 Test {i+1}: {model_path}")
        
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
                    print(f"✅ Enhanced inference succeeded")
                    print(f"Files: {list(result.get('files', {}).keys())}")
                    return True
                else:
                    print(f"❌ Enhanced inference failed: {result.get('error_message', 'Unknown error')}")
                    
        except Exception as e:
            print(f"❌ Enhanced inference error: {e}")
    
    return False

if __name__ == "__main__":
    print("🧪 Testing Custom Inference - Fixed Version")
    print("=" * 60)
    
    # Test 1: Direct custom inference
    print("\n📋 Test 1: Direct Custom Inference")
    success1 = test_custom_inference_direct()
    
    # Test 2: Enhanced inference with different checkpoints
    print("\n📋 Test 2: Enhanced Inference with Different Checkpoints")
    success2 = test_enhanced_with_better_checkpoint()
    
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS:")
    print(f"  Direct Custom: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"  Enhanced with Different Checkpoints: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 or success2:
        print("\n🎉 At least one method works!")
    else:
        print("\n⚠️ All tests failed.")
