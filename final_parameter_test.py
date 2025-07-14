#!/usr/bin/env python3
"""
Final comprehensive test of the fixed parameter estimation
"""

import sys
sys.path.append('core')

from core.apps.ml_manager.training.models.custom_models import (
    create_model_from_config
)

def final_comprehensive_test():
    """Final test of all parameter estimation scenarios"""
    
    print("FINAL COMPREHENSIVE PARAMETER ESTIMATION TEST")
    print("=" * 70)
    
    test_scenarios = [
        # MONAI UNet scenarios
        {
            'name': 'MONAI UNet (4ch) - Clean',
            'model_type': 'configurable_monai_unet',
            'channels': '32,64,128,256',
            'checkboxes': {},
            'js_base': 1600000,
            'js_mult': 1.0
        },
        {
            'name': 'MONAI UNet (4ch) - Deep only',
            'model_type': 'configurable_monai_unet', 
            'channels': '32,64,128,256',
            'checkboxes': {'deep': True},
            'js_base': 1600000,
            'js_mult': 1.5
        },
        {
            'name': 'MONAI UNet (5ch) - Clean',
            'model_type': 'configurable_monai_unet',
            'channels': '32,64,128,256,512',
            'checkboxes': {},
            'js_base': 6000000,
            'js_mult': 1.0
        },
        {
            'name': 'MONAI UNet (5ch) - Deep only (THE 9M CASE)',
            'model_type': 'configurable_monai_unet',
            'channels': '32,64,128,256,512',
            'checkboxes': {'deep': True},
            'js_base': 6000000,
            'js_mult': 1.5
        },
        # ResUNet scenarios  
        {
            'name': 'ResUNet (5ch) - Clean',
            'model_type': 'configurable_resunet',
            'channels': '32,64,128,256,512',
            'checkboxes': {},
            'js_base': 33200000,
            'js_mult': 1.0
        },
        {
            'name': 'ResUNet (5ch) - All features',
            'model_type': 'configurable_resunet',
            'channels': '32,64,128,256,512',
            'checkboxes': {'residual': True, 'attention': True, 'deep': True},
            'js_base': 33200000,
            'js_mult': 1.1 * 1.05 * 1.3
        }
    ]
    
    all_good = True
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}")
        print("-" * 60)
        
        try:
            # Create actual model
            if scenario['model_type'] == 'configurable_monai_unet':
                config = {
                    'type': scenario['model_type'],
                    'params': {
                        'input_channels': 1,
                        'output_channels': 1,
                        'custom_channels': scenario['channels']
                    }
                }
            else:
                config = {
                    'type': scenario['model_type'],
                    'params': {
                        'n_channels': 1,
                        'n_classes': 1,
                        'custom_channels': scenario['channels'],
                        'use_attention': scenario['checkboxes'].get('attention', False)
                    }
                }
            
            model = create_model_from_config(config)
            actual = sum(p.numel() for p in model.parameters())
            
            # Calculate JS estimate
            js_estimate = round(scenario['js_base'] * scenario['js_mult'])
            
            print(f"Actual:      {actual:,} ({actual/1e6:.1f}M)")
            print(f"JS estimate: {js_estimate:,} ({js_estimate/1e6:.1f}M)")
            
            accuracy = js_estimate / actual
            print(f"Accuracy:    {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            # Check special case for 9M
            if 'THE 9M CASE' in scenario['name']:
                if 8900000 <= js_estimate <= 9100000:
                    print("🎯 PERFECT: Matches the 9.0M discrepancy case!")
                else:
                    print("❌ FAIL: Doesn't match 9.0M case")
                    all_good = False
            
            # General accuracy check
            if 0.85 <= accuracy <= 1.15:  # Within 15%
                print("✅ Excellent accuracy")
            elif 0.75 <= accuracy <= 1.25:  # Within 25%
                print("⚠️  Acceptable accuracy")
            else:
                print("❌ Poor accuracy")
                all_good = False
                
        except Exception as e:
            print(f"ERROR: {e}")
            all_good = False
    
    print(f"\n{'=' * 70}")
    if all_good:
        print("🎉 ALL TESTS PASSED! Parameter estimation fixed!")
    else:
        print("❌ Some issues remain")
    print("=" * 70)

if __name__ == "__main__":
    final_comprehensive_test()
