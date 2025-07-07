#!/usr/bin/env python3
"""
Test script for enhanced inference functionality
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'core'))
sys.path.insert(0, str(project_root / 'ml'))

# Set Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')

import django
django.setup()

from core.apps.ml_manager.models import MLModel

def test_enhanced_inference():
    """Test the enhanced inference function"""
    print("=== Testing Enhanced Inference ===")
    
    # Find a completed model
    completed_models = MLModel.objects.filter(status='completed')
    
    if not completed_models.exists():
        print("❌ No completed models found")
        return
    
    model = completed_models.first()
    print(f"✅ Using model: {model.name} (ID: {model.id})")
    print(f"   Model type: {getattr(model, 'model_type', 'unet')}")
    print(f"   Model directory: {model.model_directory}")
    
    # Check if model weights exist
    model_weights_path = model.model_weights_path
    if not model_weights_path or not os.path.exists(model_weights_path):
        if model.model_directory and os.path.exists(model.model_directory):
            # Check weights directory first
            weights_dir = os.path.join(model.model_directory, 'weights')
            if os.path.exists(weights_dir):
                weights_file = os.path.join(weights_dir, 'model.pth')
                if os.path.exists(weights_file):
                    model_weights_path = weights_file
                    print(f"   Found weights in weights directory: {model_weights_path}")
            
            # If not found in weights directory, search for any .pth file
            if not model_weights_path or not os.path.exists(model_weights_path):
                for root, dirs, files in os.walk(model.model_directory):
                    for file in files:
                        if file.endswith('.pth'):
                            model_weights_path = os.path.join(root, file)
                            print(f"   Found weights: {model_weights_path}")
                            break
                    if model_weights_path and os.path.exists(model_weights_path):
                        break
    
    if not model_weights_path or not os.path.exists(model_weights_path):
        print(f"❌ Model weights not found for model {model.name}")
        return
    
    print(f"✅ Model weights: {model_weights_path}")
    
    # Check if we can import enhanced inference
    try:
        from ml.utils.enhanced_inference import run_enhanced_inference
        print("✅ Enhanced inference module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import enhanced inference: {e}")
        return
    
    # Create a dummy image for testing (if needed)
    test_image_path = os.path.join(project_root, 'test_image.jpg')
    if not os.path.exists(test_image_path):
        try:
            from PIL import Image
            import numpy as np
            
            # Create a simple test image
            test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            Image.fromarray(test_img).save(test_image_path)
            print(f"✅ Created test image: {test_image_path}")
        except Exception as e:
            print(f"❌ Failed to create test image: {e}")
            return
    
    # Test configuration
    test_config = {
        'threshold': 0.5,
        'resolution': 512,
        'confidence_threshold': 0.5,
        'min_component_size': 100,
        'morphology_kernel_size': 3,
        'apply_opening': True,
        'apply_closing': True,
        'apply_dilation': False,
        'apply_erosion': False,
        'fill_holes': True,
        'smooth_boundaries': False,
        'remove_border_objects': False,
        'model_type': getattr(model, 'model_type', 'unet'),
    }
    
    # Create output directory
    with tempfile.TemporaryDirectory() as output_dir:
        print(f"📁 Output directory: {output_dir}")
        
        try:
            print("🚀 Running enhanced inference...")
            
            results = run_enhanced_inference(
                model_path=model_weights_path,
                input_image_path=test_image_path,
                output_dir=output_dir,
                config=test_config,
                device="cpu"  # Use CPU for testing
            )
            
            print(f"✅ Inference completed!")
            print(f"   Status: {results['status']}")
            print(f"   Processing time: {results.get('processing_time', 'N/A'):.2f}s")
            print(f"   Objects detected: {results.get('detected_objects_count', 0)}")
            print(f"   Total area: {results.get('total_area_pixels', 0)} pixels")
            print(f"   Confidence scores: {results.get('confidence_scores', [])}")
            
            if 'output_files' in results:
                print("📄 Generated files:")
                for key, path in results['output_files'].items():
                    print(f"   {key}: {path}")
                    if os.path.exists(path):
                        print(f"      ✅ File exists ({os.path.getsize(path)} bytes)")
                    else:
                        print(f"      ❌ File missing")
                        
        except Exception as e:
            print(f"❌ Enhanced inference failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_inference()
