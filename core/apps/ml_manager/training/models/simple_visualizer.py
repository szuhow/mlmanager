"""
Simple Model Architecture Preview
Quick visualization tool for model architecture without advanced features
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class SimpleModelVisualizer:
    """Simple model architecture visualizer for quick previews"""
    
    def __init__(self, model: nn.Module, model_name: str = "Model"):
        self.model = model
        self.model_name = model_name
        self.layer_info = self._extract_layer_info()
        
    def _extract_layer_info(self) -> Dict:
        """Extract basic layer information from model"""
        layers = []
        total_params = 0
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                params = sum(p.numel() for p in module.parameters())
                total_params += params
                
                layer_type = type(module).__name__
                
                # Get input/output dimensions if available
                in_dim = getattr(module, 'in_channels', None) or getattr(module, 'in_features', None)
                out_dim = getattr(module, 'out_channels', None) or getattr(module, 'out_features', None)
                
                layers.append({
                    'name': name,
                    'type': layer_type,
                    'params': params,
                    'in_dim': in_dim,
                    'out_dim': out_dim
                })
        
        return {
            'layers': layers,
            'total_params': total_params,
            'total_layers': len(layers)
        }
    
    def create_simple_architecture_preview(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Create simple architecture preview diagram"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]})
        
        # Left side: Architecture flow
        self._draw_architecture_flow(ax1)
        
        # Right side: Model statistics
        self._draw_model_stats(ax2)
        
        fig.suptitle(f'{self.model_name} - Architecture Preview', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _draw_architecture_flow(self, ax):
        """Draw simplified architecture flow"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Color scheme for different layer types
        colors = {
            'Conv2d': '#FF6B6B',
            'ConvTranspose2d': '#4ECDC4',
            'Linear': '#45B7D1',
            'BatchNorm2d': '#96CEB4',
            'ReLU': '#FFEAA7',
            'MaxPool2d': '#DDA0DD',
            'Dropout': '#FFB6C1',
            'default': '#D3D3D3'
        }
        
        # Group layers by type for simplified view
        layer_groups = {}
        for layer in self.layer_info['layers']:
            layer_type = layer['type']
            if layer_type not in layer_groups:
                layer_groups[layer_type] = []
            layer_groups[layer_type].append(layer)
        
        # Draw simplified blocks
        y_pos = 10
        for layer_type, layers in layer_groups.items():
            if len(layers) == 0:
                continue
                
            color = colors.get(layer_type, colors['default'])
            
            # Calculate total parameters for this layer type
            total_params = sum(layer['params'] for layer in layers)
            
            # Draw block
            block = FancyBboxPatch((1, y_pos), 8, 1.2,
                                 boxstyle="round,pad=0.1", 
                                 facecolor=color, 
                                 edgecolor='black', linewidth=1)
            ax.add_patch(block)
            
            # Add text
            param_text = self._format_params(total_params)
            ax.text(5, y_pos + 0.6, f'{layer_type} ({len(layers)} layers)\n{param_text}', 
                   ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Add arrow to next block
            if y_pos > 1:
                ax.arrow(5, y_pos - 0.1, 0, -0.6, 
                        head_width=0.2, head_length=0.1, fc='gray', ec='gray')
            
            y_pos -= 1.8
        
        # Add input/output labels
        ax.text(5, 11.5, 'INPUT', ha='center', va='center', 
               fontsize=12, fontweight='bold', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
        ax.text(5, 0.5, 'OUTPUT', ha='center', va='center', 
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
    
    def _draw_model_stats(self, ax):
        """Draw model statistics"""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Model statistics
        stats = [
            f"Model: {self.model_name}",
            f"Class: {self.model.__class__.__name__}",
            f"Total Layers: {self.layer_info['total_layers']}",
            f"Parameters: {self._format_params(self.layer_info['total_params'])}",
            f"Size: {self.layer_info['total_params'] * 4 / (1024*1024):.1f} MB"
        ]
        
        # Layer type breakdown
        layer_counts = {}
        for layer in self.layer_info['layers']:
            layer_type = layer['type']
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
        
        stats.append("\nLayer Breakdown:")
        for layer_type, count in sorted(layer_counts.items()):
            stats.append(f"  {layer_type}: {count}")
        
        # Display stats
        stats_text = '\n'.join(stats)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    def _format_params(self, params: int) -> str:
        """Format parameter count for display"""
        if params >= 1e6:
            return f"{params/1e6:.1f}M"
        elif params >= 1e3:
            return f"{params/1e3:.1f}K"
        else:
            return str(params)
    
    def create_layer_details_view(self, figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """Create detailed layer view"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Filter out zero-parameter layers for cleaner view
        significant_layers = [layer for layer in self.layer_info['layers'] if layer['params'] > 0]
        
        if not significant_layers:
            ax.text(0.5, 0.5, 'No trainable layers found', ha='center', va='center',
                   transform=ax.transAxes, fontsize=16)
            ax.axis('off')
            return fig
        
        # Create table-like visualization
        layer_names = [layer['name'][:20] + '...' if len(layer['name']) > 20 else layer['name'] 
                      for layer in significant_layers]
        layer_types = [layer['type'] for layer in significant_layers]
        layer_params = [layer['params'] for layer in significant_layers]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(layer_names))
        bars = ax.barh(y_pos, layer_params, color='skyblue', alpha=0.7)
        
        # Customize chart
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{name}\n({ltype})" for name, ltype in zip(layer_names, layer_types)])
        ax.set_xlabel('Parameters')
        ax.set_title(f'{self.model_name} - Layer Parameter Distribution')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add parameter count labels on bars
        for i, (bar, params) in enumerate(zip(bars, layer_params)):
            if params > 0:
                ax.text(bar.get_width() + max(layer_params) * 0.01, bar.get_y() + bar.get_height()/2,
                       self._format_params(params), ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def export_summary(self) -> Dict:
        """Export model summary as dictionary"""
        return {
            'model_name': self.model_name,
            'model_class': self.model.__class__.__name__,
            'total_parameters': self.layer_info['total_params'],
            'total_layers': self.layer_info['total_layers'],
            'model_size_mb': self.layer_info['total_params'] * 4 / (1024*1024),
            'layer_breakdown': self._get_layer_breakdown(),
            'significant_layers': [
                {
                    'name': layer['name'],
                    'type': layer['type'],
                    'parameters': layer['params']
                }
                for layer in self.layer_info['layers']
                if layer['params'] > 0
            ][:10]  # Top 10 layers by parameter count
        }
    
    def _get_layer_breakdown(self) -> Dict[str, int]:
        """Get breakdown of layers by type"""
        breakdown = {}
        for layer in self.layer_info['layers']:
            layer_type = layer['type']
            breakdown[layer_type] = breakdown.get(layer_type, 0) + 1
        return breakdown


def preview_model_architecture(model: nn.Module, model_name: str = "Model") -> Dict:
    """
    Quick preview of model architecture
    
    Args:
        model: PyTorch model to preview
        model_name: Name for the model
    
    Returns:
        Dictionary with visualization info and summary
    """
    visualizer = SimpleModelVisualizer(model, model_name)
    
    # Create visualizations (in memory)
    arch_fig = visualizer.create_simple_architecture_preview()
    detail_fig = visualizer.create_layer_details_view()
    
    # Get summary
    summary = visualizer.export_summary()
    
    return {
        'architecture_figure': arch_fig,
        'details_figure': detail_fig,
        'summary': summary,
        'visualizer': visualizer
    }


def test_model_preview():
    """Test the model preview functionality"""
    print("🧪 Testing Model Architecture Preview")
    
    # Test with simple models
    test_models = {
        'Simple CNN': nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10)
        ),
        'Simple UNet-like': nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )
    }
    
    for name, model in test_models.items():
        print(f"\n📊 Previewing {name}...")
        try:
            preview = preview_model_architecture(model, name)
            summary = preview['summary']
            
            print(f"   ✅ {summary['total_parameters']:,} parameters")
            print(f"   ✅ {summary['total_layers']} layers")
            print(f"   ✅ {summary['model_size_mb']:.1f} MB")
            
            # Save figures
            output_dir = Path("model_previews")
            output_dir.mkdir(exist_ok=True)
            
            preview['architecture_figure'].savefig(output_dir / f"{name.lower().replace(' ', '_')}_arch.png", 
                                                  dpi=150, bbox_inches='tight')
            preview['details_figure'].savefig(output_dir / f"{name.lower().replace(' ', '_')}_details.png", 
                                             dpi=150, bbox_inches='tight')
            
            plt.close('all')  # Clean up
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ Model previews saved to 'model_previews' directory")


if __name__ == "__main__":
    test_model_preview()
