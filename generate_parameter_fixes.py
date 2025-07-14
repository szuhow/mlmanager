#!/usr/bin/env python3
"""
Fix the parameter estimation discrepancy in the start_training interface
"""

import sys
sys.path.append('core')

from core.apps.ml_manager.training.models.custom_models import (
    create_model_from_config,
    analyze_model_complexity
)

def generate_accurate_parameter_estimates():
    """Generate accurate parameter estimates for each model type and size"""
    
    print("ACCURATE PARAMETER ESTIMATES FOR START_TRAINING INTERFACE")
    print("=" * 80)
    
    # Model sizes from start_training.html
    sizes = {
        'micro': '8,16,32,64',
        'tiny': '16,32,64,128,256', 
        'small': '32,64,128,256,512',
        'standard': '32,64,128,256,512',
        'large': '32,64,128,256,512',
        'xl': '64,128,256,512,1024'
    }
    
    # Model types from interface
    model_types = [
        ('configurable_monai_unet', 'MONAI UNet'),
        ('configurable_resunet', 'ResUNet'),
        ('configurable_resunet', 'ResUNet + Attention', {'use_attention': True}),
        ('hybrid_resunet', 'Hybrid ResUNet')
    ]
    
    results = {}
    
    for size_name, channels in sizes.items():
        print(f"\n{'-' * 60}")
        print(f"SIZE: {size_name} (channels: {channels})")
        print(f"{'-' * 60}")
        
        for model_type, type_name, *extra in model_types:
            extra_params = extra[0] if extra else {}
            
            try:
                if model_type == 'configurable_monai_unet':
                    config = {
                        'type': model_type,
                        'params': {
                            'input_channels': 1,
                            'output_channels': 1,
                            'custom_channels': channels,
                            **extra_params
                        }
                    }
                else:
                    config = {
                        'type': model_type,
                        'params': {
                            'n_channels': 1,
                            'n_classes': 1,
                            'custom_channels': channels,
                            'use_attention': False,
                            **extra_params
                        }
                    }
                
                model = create_model_from_config(config)
                total_params = sum(p.numel() for p in model.parameters())
                
                # Format for JavaScript
                if total_params >= 1e6:
                    param_str = f"~{total_params/1e6:.1f}M"
                else:
                    param_str = f"~{total_params/1e3:.0f}K"
                
                print(f"{type_name:25} {total_params:>12,} ({param_str})")
                
                # Store for JavaScript update
                key = f"{size_name}_{model_type}"
                if extra_params:
                    key += "_" + "_".join(f"{k}_{v}" for k, v in extra_params.items())
                results[key] = {
                    'params': total_params,
                    'formatted': param_str,
                    'memory_gb': max(1, round(total_params * 4 / (1024**3) * 2)),  # Rough memory estimate
                    'speed': 'Slow' if total_params > 30e6 else 'Medium' if total_params > 10e6 else 'Fast'
                }
                
            except Exception as e:
                print(f"{type_name:25} ERROR: {e}")
    
    return results

def generate_javascript_updates(results):
    """Generate JavaScript code to update the parameter estimates"""
    
    print(f"\n{'=' * 80}")
    print("JAVASCRIPT UPDATE FOR start_training.html")
    print("=" * 80)
    
    print("""
// Updated model size to architecture mapping with accurate parameters
const sizeToArchMapping = {""")
    
    # Standard mappings with actual parameters
    standard_mappings = {
        'micro': {
            'model_type': 'configurable_monai_unet',
            'channels': '8,16,32,64',
            'description': 'Micro MONAI U-Net'
        },
        'tiny': {
            'model_type': 'configurable_monai_unet', 
            'channels': '16,32,64,128,256',
            'description': 'Tiny MONAI U-Net'
        },
        'small': {
            'model_type': 'configurable_monai_unet',
            'channels': '32,64,128,256,512', 
            'description': 'Small MONAI U-Net'
        },
        'standard': {
            'model_type': 'configurable_resunet',
            'channels': '32,64,128,256,512',
            'description': 'Standard ResU-Net'
        },
        'large': {
            'model_type': 'configurable_resunet',
            'channels': '32,64,128,256,512',
            'description': 'Large ResU-Net with Attention',
            'use_attention': True
        },
        'xl': {
            'model_type': 'hybrid_resunet',
            'channels': '64,128,256,512,1024',
            'description': 'XL Hybrid ResU-Net'
        }
    }
    
    for size, mapping in standard_mappings.items():
        # Find matching result
        model_type = mapping['model_type']
        key_base = f"{size}_{model_type}"
        
        if mapping.get('use_attention'):
            key = key_base + "_use_attention_True"
        else:
            key = key_base
            
        if key in results:
            result = results[key]
            print(f"""    '{size}': {{
        model_type: '{mapping['model_type']}',
        channels: '{mapping['channels']}',
        params: '{result['formatted']}',
        memory: '~{result['memory_gb']}GB',
        speed: '{result['speed']}',
        use_attention: {str(mapping.get('use_attention', False)).lower()}
    }},""")
    
    print("""};

// Updated parameter estimation function - uses server-side calculation for accuracy
async function getAccurateParameterCount(modelType, channels, options = {}) {
    try {
        const config = {
            type: modelType,
            params: {}
        };
        
        if (modelType === 'configurable_monai_unet') {
            config.params = {
                input_channels: 1,
                output_channels: 1,
                custom_channels: channels,
                ...options
            };
        } else {
            config.params = {
                n_channels: 1,
                n_classes: 1,
                custom_channels: channels,
                use_attention: false,
                ...options
            };
        }
        
        // Call server to get accurate count
        const response = await fetch('/ml-manager/api/model-parameters/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
            },
            body: JSON.stringify({ model_config: config })
        });
        
        const data = await response.json();
        return data.total_parameters;
        
    } catch (error) {
        console.error('Error getting parameter count:', error);
        return null;
    }
}""")

if __name__ == "__main__":
    results = generate_accurate_parameter_estimates()
    generate_javascript_updates(results)
