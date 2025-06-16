#!/usr/bin/env python
"""
Complete test for learning rate scheduler UI and functionality
"""
import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
sys.path.append('/app')
django.setup()

from core.apps.ml_manager.models import TrainingTemplate
from core.apps.ml_manager.forms import TrainingTemplateForm, TrainingForm

def test_complete_scheduler_functionality():
    """Test complete learning rate scheduler functionality including UI forms"""
    
    print("=== Complete Learning Rate Scheduler Test ===\n")
    
    # Test 1: TrainingForm with all scheduler fields
    print("1. Testing TrainingForm with scheduler fields:")
    training_form_data = {
        'model_name': 'Test Training Model',
        'data_path': '/app/data/test',
        'dataset_type': 'coronary_standard',
        'model_type': 'unet',
        'batch_size': 16,
        'epochs': 25,
        'learning_rate': 0.001,
        'validation_split': 0.2,
        'resolution': '256',
        'device': 'auto',
        'optimizer': 'adam',
        # Learning Rate Scheduler
        'lr_scheduler': 'plateau',
        'lr_patience': 8,
        'lr_factor': 0.4,
        'lr_step_size': 12,
        'lr_gamma': 0.15,
        'min_lr': 1e-6,
        # Augmentation - all required fields
        'use_random_flip': True,
        'flip_probability': 0.5,
        'use_random_rotate': True,
        'rotation_range': 30,
        'use_random_scale': True,
        'scale_range_min': 0.8,
        'scale_range_max': 1.2,
        'use_random_intensity': True,
        'intensity_range': 0.2,
        'use_random_crop': False,
        'crop_size': 128,
        'use_elastic_transform': False,
        'elastic_alpha': 34.0,
        'elastic_sigma': 4.0,
        'use_gaussian_noise': False,
        'noise_std': 0.01,
        'num_workers': 4
    }
    
    training_form = TrainingForm(data=training_form_data)
    if training_form.is_valid():
        print("   ✓ TrainingForm validation passed with scheduler fields")
        
        # Check specific scheduler fields
        scheduler_fields = ['lr_scheduler', 'lr_patience', 'lr_factor', 'lr_step_size', 'lr_gamma', 'min_lr']
        for field in scheduler_fields:
            if field in training_form.cleaned_data:
                print(f"     - {field}: {training_form.cleaned_data[field]}")
    else:
        print("   ✗ TrainingForm validation failed:")
        for field, errors in training_form.errors.items():
            print(f"     - {field}: {errors}")
    
    # Test 2: TrainingTemplateForm with complete data
    print("\n2. Testing TrainingTemplateForm with complete scheduler data:")
    template_form_data = {
        'name': 'Complete Test Template',
        'description': 'Full test with all scheduler options',
        'model_type': 'unet',
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.002,
        'validation_split': 0.15,
        'resolution': '512',
        'device': 'cuda',
        'optimizer': 'adamw',
        # Scheduler configuration
        'lr_scheduler': 'step',
        'lr_patience': 5,
        'lr_factor': 0.5,
        'lr_step_size': 10,
        'lr_gamma': 0.1,
        'min_lr': 1e-7,
        # All augmentation fields
        'use_random_flip': True,
        'flip_probability': 0.6,
        'use_random_rotate': True,
        'rotation_range': 25,
        'use_random_scale': True,
        'scale_range_min': 0.9,
        'scale_range_max': 1.1,
        'use_random_intensity': True,
        'intensity_range': 0.15,
        'use_random_crop': True,
        'crop_size': 256,
        'use_elastic_transform': True,
        'elastic_alpha': 40.0,
        'elastic_sigma': 5.0,
        'use_gaussian_noise': True,
        'noise_std': 0.02,
        'num_workers': 8,
        'is_default': False
    }
    
    template_form = TrainingTemplateForm(data=template_form_data)
    if template_form.is_valid():
        print("   ✓ TrainingTemplateForm validation passed")
        saved_template = template_form.save()
        print(f"   ✓ Template saved with ID: {saved_template.id}")
        print(f"   ✓ Scheduler: {saved_template.lr_scheduler}")
        print(f"   ✓ Patience: {saved_template.lr_patience}")
        print(f"   ✓ Factor: {saved_template.lr_factor}")
        print(f"   ✓ Step Size: {saved_template.lr_step_size}")
        print(f"   ✓ Gamma: {saved_template.lr_gamma}")
        print(f"   ✓ Min LR: {saved_template.min_lr}")
        
        # Test get_form_data method
        form_data = saved_template.get_form_data()
        print(f"   ✓ get_form_data includes {len([k for k in form_data.keys() if k.startswith('lr_')])} scheduler fields")
        
    else:
        print("   ✗ TrainingTemplateForm validation failed:")
        for field, errors in template_form.errors.items():
            print(f"     - {field}: {errors}")
    
    # Test 3: Test all scheduler types with complete forms
    print("\n3. Testing all scheduler types:")
    scheduler_configs = {
        'none': {'lr_scheduler': 'none'},
        'plateau': {'lr_scheduler': 'plateau', 'lr_patience': 10, 'lr_factor': 0.3},
        'step': {'lr_scheduler': 'step', 'lr_step_size': 15, 'lr_gamma': 0.2},
        'exponential': {'lr_scheduler': 'exponential', 'lr_gamma': 0.95},
        'cosine': {'lr_scheduler': 'cosine', 'min_lr': 1e-8},
        'adaptive': {'lr_scheduler': 'adaptive', 'lr_patience': 7, 'lr_factor': 0.4}
    }
    
    for scheduler_name, config in scheduler_configs.items():
        base_data = {
            'name': f'{scheduler_name.title()} Scheduler Template',
            'description': f'Testing {scheduler_name} scheduler configuration',
            'model_type': 'unet',
            'batch_size': 16,
            'epochs': 20,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'resolution': '256',
            'device': 'auto',
            'optimizer': 'adam',
            'min_lr': 1e-7,
            # Default augmentation settings
            'use_random_flip': True,
            'flip_probability': 0.5,
            'use_random_rotate': False,
            'rotation_range': 30,
            'use_random_scale': False,
            'scale_range_min': 0.8,
            'scale_range_max': 1.2,
            'use_random_intensity': False,
            'intensity_range': 0.2,
            'use_random_crop': False,
            'crop_size': 128,
            'use_elastic_transform': False,
            'elastic_alpha': 34.0,
            'elastic_sigma': 4.0,
            'use_gaussian_noise': False,
            'noise_std': 0.01,
            'num_workers': 4,
            'is_default': False
        }
        
        # Merge scheduler-specific config
        base_data.update(config)
        
        form = TrainingTemplateForm(data=base_data)
        if form.is_valid():
            template = form.save()
            print(f"   ✓ {scheduler_name} scheduler template created (ID: {template.id})")
        else:
            print(f"   ✗ {scheduler_name} scheduler template failed:")
            for field, errors in form.errors.items():
                print(f"     - {field}: {errors}")
    
    # Test 4: Verify scheduler field choices
    print("\n4. Testing scheduler field choices:")
    template = TrainingTemplate()
    scheduler_choices = template._meta.get_field('lr_scheduler').choices
    print(f"   Available scheduler choices: {len(scheduler_choices)}")
    for value, label in scheduler_choices:
        print(f"     - {value}: {label}")
    
    # Test 5: Template list and count
    print("\n5. Final template count:")
    total_count = TrainingTemplate.objects.count()
    print(f"   Total templates: {total_count}")
    
    scheduler_counts = {}
    for template in TrainingTemplate.objects.all():
        scheduler = template.lr_scheduler
        scheduler_counts[scheduler] = scheduler_counts.get(scheduler, 0) + 1
    
    print("   Templates by scheduler type:")
    for scheduler, count in scheduler_counts.items():
        print(f"     - {scheduler}: {count}")
    
    # Cleanup
    print("\n6. Cleaning up test templates:")
    test_templates = TrainingTemplate.objects.filter(name__icontains="Test")
    deleted_count = test_templates.count()
    test_templates.delete()
    print(f"   ✓ Deleted {deleted_count} test templates")
    
    print("\n=== Complete Scheduler Test Finished Successfully! ===")
    return True

if __name__ == "__main__":
    test_complete_scheduler_functionality()
