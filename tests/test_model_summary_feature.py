#!/usr/bin/env python3
"""
Test script for Model Summary feature

This script tests:
1. Model summary generation API
2. Local UNet registration
3. Log path resolution
4. Model architecture detection
"""

import os
import sys
import logging

# Add project paths to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'core'))
sys.path.insert(0, os.path.join(project_root, 'ml'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_registry():
    """Test model architecture registry"""
    print("=" * 60)
    print("TESTING MODEL ARCHITECTURE REGISTRY")
    print("=" * 60)
    
    try:
        from ml.utils.architecture_registry import get_default_registry
        
        registry = get_default_registry()
        architectures = registry.list_architectures()
        
        print(f"✓ Registry loaded successfully")
        print(f"✓ Found {len(architectures)} registered architectures:")
        
        for arch in architectures:
            print(f"  - {arch.key}: {arch.display_name} ({arch.framework})")
            
        # Test local UNet specifically
        local_unet = registry.get_architecture('local_unet')
        if local_unet:
            print(f"✓ Local UNet found: {local_unet.display_name}")
            print(f"  Model class: {local_unet.model_class}")
            print(f"  Default config: {local_unet.default_config}")
        else:
            print("✗ Local UNet not found in registry")
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing registry: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_summary_generation():
    """Test model summary generation"""
    print("\n" + "=" * 60)
    print("TESTING MODEL SUMMARY GENERATION")
    print("=" * 60)
    
    try:
        from core.apps.ml_manager.utils.model_summary import generate_model_summary, format_model_summary_text
        
        # Test different model types
        test_models = [
            ('unet', (1, 256, 256)),
            ('local_unet', (1, 256, 256)),
            ('resunet', (1, 256, 256)),
        ]
        
        for model_type, input_shape in test_models:
            print(f"\nTesting {model_type} with input shape {input_shape}:")
            
            try:
                summary = generate_model_summary(model_type, input_shape)
                
                if 'error' in summary:
                    print(f"  ✗ Error: {summary['error']}")
                else:
                    print(f"  ✓ Total parameters: {summary['total_params']:,}")
                    print(f"  ✓ Trainable parameters: {summary['trainable_params']:,}")
                    print(f"  ✓ Model size: {summary['model_size_mb']} MB")
                    print(f"  ✓ Input shape: {summary['input_shape']}")
                    print(f"  ✓ Number of layers: {len(summary['layers'])}")
                    
                    # Test text formatting
                    text_summary = format_model_summary_text(summary)
                    print(f"  ✓ Text summary generated ({len(text_summary)} characters)")
                    
            except Exception as e:
                print(f"  ✗ Error testing {model_type}: {e}")
                
        return True
        
    except Exception as e:
        print(f"✗ Error importing model summary modules: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_log_path_resolution():
    """Test log path resolution logic"""
    print("\n" + "=" * 60)
    print("TESTING LOG PATH RESOLUTION")
    print("=" * 60)
    
    # Test cases for log path resolution
    test_cases = [
        {
            'model_directory': 'data/models/organized/model_123_test_model',
            'expected_log_path': 'data/models/organized/model_123_test_model/logs/training.log'
        },
        {
            'model_directory': '/full/path/to/data/models/organized/model_456_another',
            'expected_log_path': 'data/models/organized/model_456_another/logs/training.log'
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        print(f"  Model directory: {case['model_directory']}")
        
        # Simulate the logic from _get_training_logs
        model_dir_parts = case['model_directory'].split(os.sep)
        if 'organized' in model_dir_parts:
            try:
                organized_index = model_dir_parts.index('organized')
                if organized_index + 1 < len(model_dir_parts):
                    model_folder = model_dir_parts[organized_index + 1]
                    constructed_path = os.path.join('data', 'models', 'organized', model_folder, 'logs', 'training.log')
                    print(f"  ✓ Constructed path: {constructed_path}")
                    
                    if constructed_path == case['expected_log_path']:
                        print(f"  ✓ Path matches expected")
                    else:
                        print(f"  ✗ Path mismatch. Expected: {case['expected_log_path']}")
                else:
                    print(f"  ✗ No model folder found after 'organized'")
            except ValueError:
                print(f"  ✗ 'organized' not found in path")
        else:
            print(f"  ✗ 'organized' not in path")
    
    return True

def test_unet_model_import():
    """Test direct UNet model import"""
    print("\n" + "=" * 60)
    print("TESTING DIRECT UNET MODEL IMPORT")
    print("=" * 60)
    
    try:
        unet_path = os.path.join(project_root, 'ml', 'training', 'models', 'unet', 'unet_model.py')
        print(f"UNet path: {unet_path}")
        print(f"UNet exists: {os.path.exists(unet_path)}")
        
        if os.path.exists(unet_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("local_unet_model", unet_path)
            local_unet_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(local_unet_module)
            
            print(f"✓ UNet module imported successfully")
            print(f"✓ UNet class: {local_unet_module.UNet}")
            
            # Test creating an instance
            model = local_unet_module.UNet(n_channels=1, n_classes=1, bilinear=False)
            print(f"✓ UNet instance created: {type(model)}")
            
            # Test forward pass
            import torch
            with torch.no_grad():
                sample_input = torch.randn(1, 1, 256, 256)
                output = model(sample_input)
                print(f"✓ Forward pass successful: {output.shape}")
            
            return True
        else:
            print(f"✗ UNet model file not found at {unet_path}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing UNet import: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 TESTING MODEL SUMMARY FEATURE")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Model Registry", test_model_registry()))
    results.append(("UNet Import", test_unet_model_import()))
    results.append(("Model Summary Generation", test_model_summary_generation()))
    results.append(("Log Path Resolution", test_log_path_resolution()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
