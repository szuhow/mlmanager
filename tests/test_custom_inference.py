#!/usr/bin/env python3
"""
Test custom inference functionality with real model weights
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add ML to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml'))

def test_custom_inference():
    """Test custom inference with real model weights"""
    
    # Find model weights
    weights_paths = []
    for root, dirs, files in os.walk("data/mlflow"):
        for file in files:
            if file.endswith(('.pth', '.pt')):
                weights_paths.append(os.path.join(root, file))
    
    if not weights_paths:
        print("❌ No model weights found")
        return False
    
    print(f"Found {len(weights_paths)} model weight files:")
    for path in weights_paths[:5]:  # Show first 5
        print(f"  - {path}")
    
    # Use first model
    model_path = weights_paths[0]
    print(f"\nTesting with: {model_path}")
    
    # Find test images
    test_images = []
    for root, dirs, files in os.walk("data/datasets"):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(root, file))
    
    if not test_images:
        print("❌ No test images found")
        return False
    
    print(f"Found {len(test_images)} test images")
    
    # Use first image
    input_image = test_images[0]
    print(f"Using image: {input_image}")
    
    # Test custom inference function
    try:
        from ml.utils.enhanced_inference import run_custom_inference
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'model_type': 'unet',
                'resolution': 512,
                'threshold': 0.5
            }
            
            # Test different device options
            devices = ['cpu']
            try:
                import torch
                if torch.cuda.is_available():
                    devices.append('cuda')
            except ImportError:
                pass
            
            for device in devices:
                print(f"\n🧪 Testing custom inference on {device}...")
                
                result = run_custom_inference(
                    model_path=model_path,
                    input_image_path=input_image,
                    output_dir=temp_dir,
                    config=config,
                    device=device
                )
                
                if result:
                    print(f"✅ Custom inference succeeded on {device}")
                    
                    # Check output files
                    output_files = list(Path(temp_dir).glob("*"))
                    print(f"Generated {len(output_files)} files:")
                    for f in output_files:
                        print(f"  - {f.name} ({f.stat().st_size} bytes)")
                else:
                    print(f"❌ Custom inference failed on {device}")
                
                # Clean temp dir for next test
                for f in Path(temp_dir).glob("*"):
                    f.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Custom inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_inference_with_custom():
    """Test enhanced inference using custom fallback"""
    
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
    
    if not weights_paths or not test_images:
        print("❌ Missing model weights or test images")
        return False
    
    model_path = weights_paths[0]
    input_image = test_images[0]
    
    print(f"\n🧪 Testing enhanced inference with custom fallback...")
    print(f"Model: {model_path}")
    print(f"Image: {input_image}")
    
    try:
        from ml.utils.enhanced_inference import run_enhanced_inference
        
        with tempfile.TemporaryDirectory() as output_dir:
            
            # Configuration with post-processing options
            config = {
                'model_type': 'unet',
                'resolution': 512,
                'threshold': 0.5,
                'post_processing': {
                    'morphological_ops': True,
                    'noise_filtering': True,
                    'connected_components': True,
                    'edge_enhancement': True,
                    'test_time_augmentation': False
                }
            }
            
            result = run_enhanced_inference(
                model_path=model_path,
                input_image_path=input_image,
                output_dir=output_dir,
                config=config
            )
            
            if result['success']:
                print("✅ Enhanced inference succeeded")
                print(f"Files generated: {result['files']}")
                print(f"Metrics: {result['metrics']}")
                
                # Check if files exist
                for file_type, file_path in result['files'].items():
                    if os.path.exists(file_path):
                        size = os.path.getsize(file_path)
                        print(f"  ✅ {file_type}: {file_path} ({size} bytes)")
                    else:
                        print(f"  ❌ {file_type}: {file_path} (missing)")
            else:
                print(f"❌ Enhanced inference failed: {result.get('error', 'Unknown error')}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Custom Inference Implementation")
    print("=" * 60)
    
    # Test 1: Basic custom inference
    print("\n📋 Test 1: Custom Inference Function")
    success1 = test_custom_inference()
    
    # Test 2: Enhanced inference with custom fallback  
    print("\n📋 Test 2: Enhanced Inference with Custom Fallback")
    success2 = test_enhanced_inference_with_custom()
    
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS:")
    print(f"  Custom Inference: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"  Enhanced with Custom: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! Custom inference is ready.")
    else:
        print("\n⚠️ Some tests failed. Check the output above.")
