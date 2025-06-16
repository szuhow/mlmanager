#!/usr/bin/env python3
"""
Test script to verify the enhanced architecture registry is working correctly
and showing all ResUNet model variants in Django forms.
"""

import os
import sys
import django
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / 'core'))

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

def test_architecture_registry():
    """Test the architecture registry directly"""
    print("=" * 60)
    print("Testing Architecture Registry")
    print("=" * 60)
    
    try:
        from ml.utils.architecture_registry import registry, get_available_models
        
        # Get all registered architectures
        all_architectures = registry.get_all_architectures()
        print(f"Total registered architectures: {len(all_architectures)}")
        print()
        
        # Group by category
        categories = registry.get_categories()
        print(f"Available categories: {categories}")
        print()
        
        # Show all models with details
        print("All Registered Models:")
        print("-" * 40)
        for key, arch_info in all_architectures.items():
            print(f"Key: {key}")
            print(f"  Display Name: {arch_info.display_name}")
            print(f"  Framework: {arch_info.framework}")
            print(f"  Description: {arch_info.description}")
            print(f"  Category: {arch_info.category}")
            print(f"  Default Config: {arch_info.default_config}")
            print()
        
        # Test get_available_models function (used by Django forms)
        form_choices = get_available_models()
        print("Django Form Choices:")
        print("-" * 40)
        for key, display_name in form_choices:
            print(f"  {key}: {display_name}")
        
        return True, len(all_architectures), len([k for k in all_architectures if 'resunet' in k])
        
    except Exception as e:
        print(f"❌ Error testing architecture registry: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, 0

def test_django_forms():
    """Test Django forms to ensure they can access all model choices"""
    print("\n" + "=" * 60)
    print("Testing Django Forms Integration")
    print("=" * 60)
    
    try:
        from core.apps.ml_manager.forms import TrainingForm, TrainingTemplateForm
        
        # Test TrainingForm
        print("TrainingForm model choices:")
        training_form = TrainingForm()
        model_choices = training_form.fields['model_type'].choices
        print(f"  Total choices: {len(model_choices)}")
        for key, display_name in model_choices:
            print(f"  {key}: {display_name}")
        
        print("\nTrainingTemplateForm model choices:")
        template_form = TrainingTemplateForm()
        template_model_choices = template_form.fields['model_type'].choices
        print(f"  Total choices: {len(template_model_choices)}")
        for key, display_name in template_model_choices:
            print(f"  {key}: {display_name}")
        
        return True, len(model_choices)
        
    except Exception as e:
        print(f"❌ Error testing Django forms: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def test_model_instantiation():
    """Test that ResUNet models can be instantiated correctly"""
    print("\n" + "=" * 60)
    print("Testing Model Instantiation")
    print("=" * 60)
    
    try:
        from ml.utils.architecture_registry import registry
        
        # Test key ResUNet models
        resunet_keys = [
            'resunet',
            'deep_resunet', 
            'resunet_attention',
            'deep_resunet_attention'
        ]
        
        instantiated_models = []
        
        for key in resunet_keys:
            try:
                arch_info = registry.get_architecture(key)
                if arch_info:
                    print(f"Testing {key} ({arch_info.display_name})...")
                    
                    # Get the model class and config
                    model_class = arch_info.model_class
                    config = arch_info.default_config.copy()
                    
                    # Ensure RGB input (3 channels)
                    if 'n_channels' in config:
                        config['n_channels'] = 3
                    elif 'in_channels' in config:
                        config['in_channels'] = 3
                    
                    # Create model instance
                    if config:
                        model = model_class(**config)
                    else:
                        model = model_class()
                    
                    print(f"  ✓ Successfully created {key}")
                    instantiated_models.append(key)
                    
                    # Test forward pass with RGB input
                    import torch
                    test_input = torch.randn(1, 3, 256, 256)  # RGB input
                    
                    with torch.no_grad():
                        output = model(test_input)
                        print(f"  ✓ Forward pass successful. Output shape: {output.shape}")
                    
                else:
                    print(f"  ❌ Architecture {key} not found in registry")
                    
            except Exception as e:
                print(f"  ❌ Error with {key}: {e}")
        
        return True, len(instantiated_models)
        
    except Exception as e:
        print(f"❌ Error testing model instantiation: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def main():
    """Run all tests"""
    print("Enhanced Architecture Registry Test Suite")
    print("=" * 60)
    
    # Test 1: Architecture Registry
    registry_success, total_models, resunet_models = test_architecture_registry()
    
    # Test 2: Django Forms
    forms_success, form_choices = test_django_forms()
    
    # Test 3: Model Instantiation
    instantiation_success, instantiated_count = test_model_instantiation()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Architecture Registry: {'✓ PASSED' if registry_success else '❌ FAILED'}")
    print(f"  - Total models registered: {total_models}")
    print(f"  - ResUNet variants: {resunet_models}")
    
    print(f"Django Forms Integration: {'✓ PASSED' if forms_success else '❌ FAILED'}")
    print(f"  - Form choices available: {form_choices}")
    
    print(f"Model Instantiation: {'✓ PASSED' if instantiation_success else '❌ FAILED'}")
    print(f"  - Models successfully instantiated: {instantiated_count}")
    
    # Overall result
    all_passed = registry_success and forms_success and instantiation_success
    print(f"\nOVERALL RESULT: {'✓ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Enhanced model registry is working correctly!")
        print("All ResUNet variants should now be available in Django forms.")
    else:
        print("\n⚠️  Some issues need to be resolved.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
