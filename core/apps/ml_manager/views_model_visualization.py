"""
Django Integration for Model Architecture Visualization
Provides web interface for model visualization and comparison
"""

from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import json
import torch
import torch.nn as nn
from pathlib import Path
import logging
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import model classes
try:
    from ..training.models.custom_models import (
        get_all_available_models,
        create_model_from_config,
        analyze_model_complexity,
        compare_model_architectures,
        create_model_comparison_report,
        preview_model_architecture_simple
    )
    from ..training.models.visualization import (
        ModelArchitectureVisualizer,
        visualize_model,
        compare_models
    )
    MODELS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import model classes: {e}")
    try:
        # Try fallback to simple visualizer only
        from ..training.models.custom_models import (
            get_all_available_models,
            create_model_from_config,
            analyze_model_complexity,
            compare_model_architectures,
            preview_model_architecture_simple
        )
        MODELS_AVAILABLE = True
        ADVANCED_VIZ_AVAILABLE = False
        logger.warning("Using simple visualizer only - advanced features disabled")
    except ImportError:
        MODELS_AVAILABLE = False
        logger.error("Model visualization completely unavailable")


def model_architecture_dashboard(request):
    """Main dashboard for model architecture visualization"""
    if not MODELS_AVAILABLE:
        return render(request, 'ml_manager/error.html', {
            'error': 'Model visualization not available - missing dependencies'
        })
    
    # Get available model types
    available_models = get_all_available_models()
    
    context = {
        'available_models': list(available_models.keys()),
        'page_title': 'Model Architecture Visualization',
        'description': 'Visualize and compare neural network architectures'
    }
    
    return render(request, 'ml_manager/model_visualization_dashboard.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def create_model_visualization(request):
    """Create model visualization via AJAX"""
    if not MODELS_AVAILABLE:
        return JsonResponse({'error': 'Model visualization not available'}, status=500)
    
    try:
        data = json.loads(request.body)
        model_config = data.get('model_config', {})
        visualization_options = data.get('visualization_options', {})
        
        # Create model
        model = create_model_from_config(model_config)
        model_name = model_config.get('name', model.__class__.__name__)
        
        # Use simple visualizer
        preview_result = preview_model_architecture_simple(model, model_name)
        
        if 'error' in preview_result:
            return JsonResponse({'error': preview_result['error']}, status=500)
        
        # Generate visualizations
        visualizations = {}
        
        # Architecture diagram
        if visualization_options.get('architecture_diagram', True) and 'architecture_figure' in preview_result:
            fig = preview_result['architecture_figure']
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            visualizations['architecture'] = f"data:image/png;base64,{img_base64}"
            plt.close(fig)
        
        # Layer diagram
        if visualization_options.get('layer_diagram', True) and 'details_figure' in preview_result:
            fig = preview_result['details_figure']
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            visualizations['layers'] = f"data:image/png;base64,{img_base64}"
            plt.close(fig)
        
        # Parameter analysis - use summary data
        if visualization_options.get('parameter_analysis', True) and 'summary' in preview_result:
            summary = preview_result['summary']
            
            # Create simple parameter chart
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Parameter breakdown pie chart
            if 'layer_breakdown' in summary:
                layer_types = list(summary['layer_breakdown'].keys())
                counts = list(summary['layer_breakdown'].values())
                ax1.pie(counts, labels=layer_types, autopct='%1.1f%%', startangle=90)
                ax1.set_title('Layer Type Distribution')
            
            # Model stats
            stats_text = f"""
Model: {summary['model_name']}
Class: {summary['model_class']}
Parameters: {summary['total_parameters']:,}
Layers: {summary['total_layers']}
Size: {summary['model_size_mb']:.1f} MB
            """
            ax2.text(0.1, 0.5, stats_text.strip(), transform=ax2.transAxes, 
                    fontsize=12, verticalalignment='center', fontfamily='monospace')
            ax2.set_title('Model Statistics')
            ax2.axis('off')
            
            plt.tight_layout()
            
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            visualizations['parameters'] = f"data:image/png;base64,{img_base64}"
            plt.close(fig)
        
        # Model analysis
        complexity_analysis = analyze_model_complexity(model)
        
        # Architecture summary
        summary = preview_result.get('summary', {})
        summary.update(complexity_analysis)  # Merge analysis data
        
        return JsonResponse({
            'success': True,
            'visualizations': visualizations,
            'complexity_analysis': complexity_analysis,
            'summary': summary,
            'model_name': model_name
        })
        
    except Exception as e:
        logger.error(f"Model visualization failed: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def compare_models_view(request):
    """Compare multiple models"""
    if not MODELS_AVAILABLE:
        return JsonResponse({'error': 'Model comparison not available'}, status=500)
    
    try:
        data = json.loads(request.body)
        model_configs = data.get('models', [])
        
        if len(model_configs) < 2:
            return JsonResponse({'error': 'At least 2 models required for comparison'}, status=400)
        
        # Create models
        models = {}
        for config in model_configs:
            try:
                model = create_model_from_config(config)
                model_name = config.get('name', model.__class__.__name__)
                models[model_name] = model
            except Exception as e:
                logger.warning(f"Failed to create model {config.get('name', 'unknown')}: {e}")
        
        if len(models) < 2:
            return JsonResponse({'error': 'Failed to create enough models for comparison'}, status=400)
        
        # Compare models
        comparison_data = compare_model_architectures(models)
        
        # Create comparison visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        model_names = list(models.keys())
        param_counts = [comparison_data[name].get('total_parameters', 0) for name in model_names]
        
        # Parameter count comparison
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(model_names)]
        bars = ax1.bar(model_names, param_counts, color=colors)
        ax1.set_title('Parameter Count Comparison')
        ax1.set_ylabel('Parameters')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, count in zip(bars, param_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}' if count < 1e6 else f'{count/1e6:.1f}M',
                    ha='center', va='bottom', fontsize=9)
        
        # Model size comparison
        model_sizes = [comparison_data[name].get('model_size_mb', 0) for name in model_names]
        ax2.pie(model_sizes, labels=model_names, autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.set_title('Relative Model Sizes (MB)')
        
        # Inference time comparison
        inference_times = [comparison_data[name].get('inference_time_ms', 0) for name in model_names]
        ax3.bar(model_names, inference_times, color=colors)
        ax3.set_title('Inference Time Comparison')
        ax3.set_ylabel('Time (ms)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Efficiency scatter plot
        efficiencies = [comparison_data[name].get('parameter_efficiency', 1.0) for name in model_names]
        ax4.scatter(param_counts, inference_times, s=100, c=colors[:len(model_names)], alpha=0.7)
        for i, name in enumerate(model_names):
            ax4.annotate(name, (param_counts[i], inference_times[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax4.set_xlabel('Parameters')
        ax4.set_ylabel('Inference Time (ms)')
        ax4.set_title('Parameter vs Performance')
        
        plt.suptitle('Model Architecture Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        comparison_plot = f"data:image/png;base64,{img_base64}"
        plt.close()
        
        return JsonResponse({
            'success': True,
            'comparison_data': comparison_data,
            'comparison_plot': comparison_plot,
            'models_compared': list(models.keys())
        })
        
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def get_model_templates(request):
    """Get predefined model configuration templates"""
    templates = {
        'small_unet': {
            'name': 'Small U-Net',
            'type': 'configurable_monai_unet',
            'params': {
                'input_channels': 1,
                'output_channels': 1,
                'custom_channels': '16,32,64,128'
            },
            'description': 'Lightweight U-Net for small images or quick testing'
        },
        'standard_resunet': {
            'name': 'Standard ResU-Net',
            'type': 'configurable_resunet',
            'params': {
                'n_channels': 1,
                'n_classes': 1,
                'custom_channels': '32,64,128,256',
                'use_attention': False
            },
            'description': 'Standard ResU-Net with residual connections'
        },
        'attention_resunet': {
            'name': 'ResU-Net with Attention',
            'type': 'configurable_resunet',
            'params': {
                'n_channels': 1,
                'n_classes': 1,
                'custom_channels': '32,64,128,256,512',
                'use_attention': True
            },
            'description': 'ResU-Net with attention gates for better feature focus'
        },
        # 'sam_light_small': {
        #     'name': 'SAM-Light (Small) - WIP',
        #     'type': 'sam_light_unet',
        #     'params': {
        #         'input_channels': 1,
        #         'output_channels': 1,
        #         'embed_dim': 256,
        #         'encoder_depth': 4,
        #         'decoder_channels': '128,64,32'
        #     },
        #     'description': 'Lightweight SAM-inspired model - Work In Progress'
        # },
        # 'sam_light_large': {
        #     'name': 'SAM-Light (Large) - WIP',
        #     'type': 'sam_light_unet',
        #     'params': {
        #         'input_channels': 1,
        #         'output_channels': 1,
        #         'embed_dim': 384,
        #         'encoder_depth': 6,
        #         'decoder_channels': '256,128,64,32'
        #     },
        #     'description': 'Larger SAM-inspired model - Work In Progress'
        # },
        'transformer_unet': {
            'name': 'Transformer U-Net',
            'type': 'transformer_unet',
            'params': {
                'input_channels': 1,
                'output_channels': 1,
                'base_channels': 64,
                'transformer_layers': 4,
                'num_heads': 8
            },
            'description': 'Hybrid CNN-Transformer architecture'
        },
        'hybrid_resunet': {
            'name': 'Hybrid ResU-Net',
            'type': 'hybrid_resunet',
            'params': {
                'n_channels': 1,
                'n_classes': 1,
                'custom_channels': '32,64,128,256,512',
                'use_efficient_attention': True,
                'dropout_rate': 0.1
            },
            'description': 'Advanced ResU-Net with efficient attention and dropout'
        }
    }
    
    return JsonResponse({'templates': templates})


@require_http_methods(["GET"])
def download_model_architecture(request, model_config_b64):
    """Download model architecture as JSON"""
    try:
        # Decode base64 model config
        model_config_json = base64.b64decode(model_config_b64).decode('utf-8')
        model_config = json.loads(model_config_json)
        
        # Create model and analyze
        model = create_model_from_config(model_config)
        visualizer = ModelArchitectureVisualizer(model, model_config.get('name', 'Model'))
        
        # Export comprehensive summary
        summary = visualizer.export_architecture_summary('')
        
        # Create response
        response = HttpResponse(
            json.dumps(summary, indent=2),
            content_type='application/json'
        )
        response['Content-Disposition'] = f'attachment; filename="{model_config.get("name", "model")}_architecture.json"'
        
        return response
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def model_architecture_api_docs(request):
    """API documentation for model architecture endpoints"""
    context = {
        'endpoints': [
            {
                'url': '/ml-manager/api/models/visualize/',
                'method': 'POST',
                'description': 'Create model visualization',
                'parameters': {
                    'model_config': 'Model configuration object',
                    'visualization_options': 'Visualization options object'
                }
            },
            {
                'url': '/ml-manager/api/models/compare/',
                'method': 'POST', 
                'description': 'Compare multiple models',
                'parameters': {
                    'models': 'Array of model configuration objects'
                }
            },
            {
                'url': '/ml-manager/api/models/templates/',
                'method': 'GET',
                'description': 'Get predefined model templates'
            }
        ]
    }
    
    return render(request, 'ml_manager/api_docs.html', context)
