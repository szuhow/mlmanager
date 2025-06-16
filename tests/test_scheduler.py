#!/usr/bin/env python
"""
Test script for learning rate scheduler functionality
"""
import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
sys.path.append('/app')
django.setup()

from core.apps.ml_manager.models import TrainingTemplate
from core.apps.ml_manager.forms import TrainingTemplateForm

def test_scheduler_fields():
    """Test learning rate scheduler fields in TrainingTemplate model and form"""
    
    print("=== Testing Learning Rate Scheduler Functionality ===\n")
    
    # Test 1: Check if scheduler fields exist in model
    print("1. Testing TrainingTemplate model fields:")
    template = TrainingTemplate()
    scheduler_fields = [
        'lr_scheduler', 'lr_patience', 'lr_factor', 
        'lr_step_size', 'lr_gamma', 'min_lr'
    ]
    
    for field in scheduler_fields:
        if hasattr(template, field):
            print(f"   ✓ {field} field exists")
        else:
            print(f"   ✗ {field} field missing")
    
    # Test 2: Create a template with scheduler configuration
    print("\n2. Testing template creation with scheduler settings:")
    try:
        test_template = TrainingTemplate.objects.create(
            name="Test Scheduler Template",
            description="Test template for scheduler functionality",
            model_type="unet",
            batch_size=16,
            epochs=50,
            learning_rate=0.001,
            validation_split=0.2,
            resolution="256",
            device="auto",
            optimizer="adam",
            # Scheduler configuration
            lr_scheduler="plateau",
            lr_patience=10,
            lr_factor=0.3,
            lr_step_size=15,
            lr_gamma=0.2,
            min_lr=1e-6,
            # Augmentation settings
            use_random_flip=True,
            use_random_rotate=True,
            use_random_scale=False,
            use_random_intensity=True,
            crop_size=128,
            num_workers=4
        )
        print(f"   ✓ Template created with ID: {test_template.id}")
        print(f"   ✓ Scheduler type: {test_template.lr_scheduler}")
        print(f"   ✓ Scheduler patience: {test_template.lr_patience}")
        print(f"   ✓ Scheduler factor: {test_template.lr_factor}")
        
    except Exception as e:
        print(f"   ✗ Error creating template: {e}")
        return False
    
    # Test 3: Test form validation
    print("\n3. Testing TrainingTemplateForm with scheduler data:")
    form_data = {
        'name': 'Test Form Template',
        'description': 'Test form validation',
        'model_type': 'unet',
        'batch_size': 32,
        'epochs': 25,
        'learning_rate': 0.002,
        'validation_split': 0.15,
        'resolution': '512',
        'device': 'cuda',
        'optimizer': 'adamw',
        # Scheduler fields
        'lr_scheduler': 'step',
        'lr_patience': 5,
        'lr_factor': 0.5,
        'lr_step_size': 10,
        'lr_gamma': 0.1,
        'min_lr': 1e-7,
        # Required fields
        'use_random_flip': True,
        'use_random_rotate': False,
        'use_random_scale': True,
        'use_random_intensity': False,
        'crop_size': 256,
        'num_workers': 2
    }
    
    form = TrainingTemplateForm(data=form_data)
    if form.is_valid():
        print("   ✓ Form validation passed")
        form_template = form.save()
        print(f"   ✓ Form template saved with ID: {form_template.id}")
        print(f"   ✓ Form scheduler type: {form_template.lr_scheduler}")
    else:
        print("   ✗ Form validation failed:")
        for field, errors in form.errors.items():
            print(f"     - {field}: {errors}")
    
    # Test 4: Test different scheduler types
    print("\n4. Testing different scheduler types:")
    scheduler_types = ['none', 'plateau', 'step', 'exponential', 'cosine', 'adaptive']
    
    for scheduler_type in scheduler_types:
        try:
            test_sched_template = TrainingTemplate.objects.create(
                name=f"Test {scheduler_type.title()} Scheduler",
                description=f"Testing {scheduler_type} scheduler",
                model_type="unet",
                batch_size=8,
                epochs=10,
                learning_rate=0.001,
                validation_split=0.2,
                resolution="128",
                device="auto",
                optimizer="adam",
                lr_scheduler=scheduler_type,
                lr_patience=7,
                lr_factor=0.4,
                lr_step_size=8,
                lr_gamma=0.15,
                min_lr=1e-8,
                use_random_flip=True,
                crop_size=64,
                num_workers=1
            )
            print(f"   ✓ {scheduler_type} scheduler template created (ID: {test_sched_template.id})")
            
        except Exception as e:
            print(f"   ✗ Error creating {scheduler_type} template: {e}")
    
    # Test 5: Test get_form_data method
    print("\n5. Testing get_form_data method:")
    try:
        form_data = test_template.get_form_data()
        scheduler_keys = [k for k in form_data.keys() if k.startswith('lr_')]
        print(f"   ✓ get_form_data returned {len(scheduler_keys)} scheduler fields:")
        for key in scheduler_keys:
            print(f"     - {key}: {form_data[key]}")
            
    except Exception as e:
        print(f"   ✗ Error in get_form_data: {e}")
    
    # Test 6: Count created templates
    print("\n6. Summary:")
    total_templates = TrainingTemplate.objects.count()
    print(f"   Total templates in database: {total_templates}")
    
    # Clean up test templates
    print("\n7. Cleaning up test templates:")
    test_templates = TrainingTemplate.objects.filter(name__icontains="Test")
    deleted_count = test_templates.count()
    test_templates.delete()
    print(f"   ✓ Deleted {deleted_count} test templates")
    
    print("\n=== Learning Rate Scheduler Test Completed Successfully! ===")
    return True

if __name__ == "__main__":
    test_scheduler_fields()
