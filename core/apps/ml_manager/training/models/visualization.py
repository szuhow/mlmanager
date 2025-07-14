"""
Model Architecture Visualization Tools
Provides comprehensive visualization capabilities for neural network architectures
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Try to import optional dependencies for advanced visualization
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    logger.warning("Graphviz not available - some visualizations will be limited")

try:
    from torchviz import make_dot
    TORCHVIZ_AVAILABLE = True
except ImportError:
    TORCHVIZ_AVAILABLE = False
    logger.warning("Torchviz not available - computational graph visualization disabled")


class ModelArchitectureVisualizer:
    """Comprehensive model architecture visualization with multiple visualization types"""
    
    def __init__(self, model: nn.Module, model_name: str = "Model"):
        self.model = model
        self.model_name = model_name
        self.layers_info = self._analyze_model_structure()
        
    def _analyze_model_structure(self) -> Dict[str, Any]:
        """Analyze the model structure and extract layer information"""
        layers_info = {
            'encoder_layers': [],
            'decoder_layers': [],
            'skip_connections': [],
            'attention_layers': [],
            'special_layers': [],
            'total_params': 0,
            'layer_details': {}
        }
        
        # Count parameters
        layers_info['total_params'] = sum(p.numel() for p in self.model.parameters())
        
        # Analyze layers
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_info = {
                    'name': name,
                    'type': type(module).__name__,
                    'params': sum(p.numel() for p in module.parameters()),
                    'input_size': getattr(module, 'in_channels', None) or getattr(module, 'in_features', None),
                    'output_size': getattr(module, 'out_channels', None) or getattr(module, 'out_features', None),
                }
                
                # Categorize layers
                if 'down' in name.lower() or 'encoder' in name.lower():
                    layers_info['encoder_layers'].append(layer_info)
                elif 'up' in name.lower() or 'decoder' in name.lower():
                    layers_info['decoder_layers'].append(layer_info)
                elif 'att' in name.lower() or 'attention' in name.lower():
                    layers_info['attention_layers'].append(layer_info)
                else:
                    layers_info['special_layers'].append(layer_info)
                
                layers_info['layer_details'][name] = layer_info
        
        return layers_info
    
    def create_unet_architecture_diagram(self, save_path: Optional[str] = None, 
                                       figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """Create a detailed U-Net style architecture diagram"""
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 15)
        ax.axis('off')
        
        # Color scheme
        colors = {
            'encoder': '#FF6B6B',
            'decoder': '#4ECDC4', 
            'skip': '#45B7D1',
            'attention': '#FFA07A',
            'conv': '#96CEB4',
            'pool': '#FFEAA7',
            'upconv': '#DDA0DD'
        }
        
        # Draw encoder path (left side)
        encoder_x = 2
        encoder_levels = len(self.layers_info['encoder_layers']) if self.layers_info['encoder_layers'] else 5
        
        encoder_blocks = []
        for i in range(encoder_levels):
            y_pos = 12 - i * 2.5
            width = 2.5
            height = 1.5
            
            # Main encoder block
            block = FancyBboxPatch((encoder_x, y_pos), width, height,
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['encoder'], 
                                 edgecolor='black', linewidth=2)
            ax.add_patch(block)
            
            # Add text
            ax.text(encoder_x + width/2, y_pos + height/2, 
                   f'Encoder {i+1}\n{2**(6+i)} channels', 
                   ha='center', va='center', fontsize=9, fontweight='bold')
            
            encoder_blocks.append((encoder_x + width/2, y_pos + height/2))
            
            # Add downsampling arrow (except for the last block)
            if i < encoder_levels - 1:
                ax.arrow(encoder_x + width/2, y_pos - 0.1, 0, -0.8, 
                        head_width=0.2, head_length=0.1, fc='red', ec='red')
        
        # Draw decoder path (right side)
        decoder_x = 15
        decoder_levels = len(self.layers_info['decoder_layers']) if self.layers_info['decoder_layers'] else 4
        
        decoder_blocks = []
        for i in range(decoder_levels):
            y_pos = 4.5 + i * 2.5
            width = 2.5
            height = 1.5
            
            # Main decoder block
            block = FancyBboxPatch((decoder_x, y_pos), width, height,
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['decoder'], 
                                 edgecolor='black', linewidth=2)
            ax.add_patch(block)
            
            # Add text
            ax.text(decoder_x + width/2, y_pos + height/2, 
                   f'Decoder {i+1}\n{2**(8-i)} channels', 
                   ha='center', va='center', fontsize=9, fontweight='bold')
            
            decoder_blocks.append((decoder_x + width/2, y_pos + height/2))
            
            # Add upsampling arrow (except for the last block)
            if i < decoder_levels - 1:
                ax.arrow(decoder_x + width/2, y_pos + height + 0.1, 0, 0.8, 
                        head_width=0.2, head_length=0.1, fc='blue', ec='blue')
        
        # Draw bottleneck
        bottleneck_y = 2
        bottleneck = FancyBboxPatch((8.5, bottleneck_y), 3, 1.5,
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#FF9999', 
                                  edgecolor='black', linewidth=2)
        ax.add_patch(bottleneck)
        ax.text(10, bottleneck_y + 0.75, 'Bottleneck\n1024 channels', 
               ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Draw skip connections
        for i in range(min(len(encoder_blocks), len(decoder_blocks))):
            enc_x, enc_y = encoder_blocks[i]
            dec_x, dec_y = decoder_blocks[-(i+1)]
            
            # Skip connection arc
            connection = ConnectionPatch((enc_x + 1.25, enc_y), (dec_x - 1.25, dec_y), 
                                       "data", "data",
                                       arrowstyle="->", shrinkA=5, shrinkB=5,
                                       mutation_scale=20, fc=colors['skip'], 
                                       ec=colors['skip'], linewidth=2,
                                       connectionstyle="arc3,rad=0.3")
            ax.add_patch(connection)
        
        # Add attention gates if present
        if self.layers_info['attention_layers']:
            for i, (enc_x, enc_y) in enumerate(encoder_blocks[:len(decoder_blocks)]):
                dec_x, dec_y = decoder_blocks[-(i+1)]
                att_x = (enc_x + dec_x) / 2
                att_y = (enc_y + dec_y) / 2
                
                attention_gate = FancyBboxPatch((att_x - 0.5, att_y - 0.3), 1, 0.6,
                                              boxstyle="round,pad=0.05", 
                                              facecolor=colors['attention'], 
                                              edgecolor='orange', linewidth=1.5)
                ax.add_patch(attention_gate)
                ax.text(att_x, att_y, 'ATT', ha='center', va='center', 
                       fontsize=8, fontweight='bold')
        
        # Add title and legend
        ax.text(10, 14.5, f'{self.model_name} Architecture', 
               ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Model info
        param_count = self.layers_info['total_params']
        param_text = f"Total Parameters: {param_count:,}"
        if param_count > 1e6:
            param_text += f" ({param_count/1e6:.1f}M)"
        elif param_count > 1e3:
            param_text += f" ({param_count/1e3:.1f}K)"
            
        ax.text(10, 0.5, param_text, ha='center', va='center', 
               fontsize=12, style='italic')
        
        # Legend
        legend_elements = [
            patches.Patch(color=colors['encoder'], label='Encoder Blocks'),
            patches.Patch(color=colors['decoder'], label='Decoder Blocks'),
            patches.Patch(color=colors['skip'], label='Skip Connections'),
        ]
        if self.layers_info['attention_layers']:
            legend_elements.append(patches.Patch(color=colors['attention'], label='Attention Gates'))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Architecture diagram saved to {save_path}")
        
        return fig
    
    def create_detailed_layer_diagram(self, save_path: Optional[str] = None,
                                    figsize: Tuple[int, int] = (20, 14)) -> plt.Figure:
        """Create a detailed layer-by-layer diagram"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Analyze all layers
        all_layers = []
        for layer_type in ['encoder_layers', 'decoder_layers', 'attention_layers', 'special_layers']:
            all_layers.extend(self.layers_info[layer_type])
        
        if not all_layers:
            # Fallback: analyze from model structure
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d)):
                    all_layers.append({
                        'name': name,
                        'type': type(module).__name__,
                        'params': sum(p.numel() for p in module.parameters()) if hasattr(module, 'parameters') else 0
                    })
        
        # Layout parameters
        layers_per_row = 8
        row_height = 2
        col_width = 2.5
        
        rows = (len(all_layers) + layers_per_row - 1) // layers_per_row
        ax.set_xlim(0, layers_per_row * col_width)
        ax.set_ylim(0, rows * row_height + 2)
        ax.axis('off')
        
        # Color mapping for different layer types
        type_colors = {
            'Conv2d': '#FF6B6B',
            'ConvTranspose2d': '#4ECDC4',
            'BatchNorm2d': '#45B7D1',
            'ReLU': '#96CEB4',
            'MaxPool2d': '#FFEAA7',
            'Dropout': '#DDA0DD',
            'Linear': '#FF9999',
            'AdaptiveAvgPool2d': '#98FB98',
            'default': '#D3D3D3'
        }
        
        # Draw layers
        for i, layer in enumerate(all_layers):
            row = i // layers_per_row
            col = i % layers_per_row
            
            x = col * col_width + 0.25
            y = (rows - row - 1) * row_height + 0.5
            
            layer_type = layer.get('type', 'Unknown')
            color = type_colors.get(layer_type, type_colors['default'])
            
            # Draw layer box
            box = FancyBboxPatch((x, y), col_width - 0.5, row_height - 0.5,
                               boxstyle="round,pad=0.1", 
                               facecolor=color, 
                               edgecolor='black', linewidth=1)
            ax.add_patch(box)
            
            # Add layer information
            layer_name = layer.get('name', f'Layer {i}')
            if len(layer_name) > 15:
                layer_name = layer_name[:12] + '...'
            
            param_count = layer.get('params', 0)
            param_text = f"{param_count}" if param_count > 0 else "0"
            if param_count > 1000:
                param_text = f"{param_count//1000}K"
            
            ax.text(x + (col_width - 0.5)/2, y + (row_height - 0.5)/2 + 0.2, 
                   layer_name, ha='center', va='center', 
                   fontsize=8, fontweight='bold')
            ax.text(x + (col_width - 0.5)/2, y + (row_height - 0.5)/2 - 0.1, 
                   layer_type, ha='center', va='center', 
                   fontsize=7)
            ax.text(x + (col_width - 0.5)/2, y + (row_height - 0.5)/2 - 0.3, 
                   f"Params: {param_text}", ha='center', va='center', 
                   fontsize=6, style='italic')
        
        # Add title
        ax.text(layers_per_row * col_width / 2, rows * row_height + 1.5, 
               f'{self.model_name} - Detailed Layer Structure', 
               ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Add legend
        legend_elements = [patches.Patch(color=color, label=layer_type) 
                          for layer_type, color in type_colors.items() 
                          if layer_type != 'default']
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1, 1), ncol=2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Detailed layer diagram saved to {save_path}")
        
        return fig
    
    def create_computational_graph(self, input_tensor: torch.Tensor, 
                                 save_path: Optional[str] = None) -> Optional[Any]:
        """Create computational graph visualization using torchviz"""
        if not TORCHVIZ_AVAILABLE:
            logger.warning("Torchviz not available - cannot create computational graph")
            return None
        
        try:
            # Forward pass to create computation graph
            self.model.eval()
            output = self.model(input_tensor)
            
            # Create graph
            graph = make_dot(output, params=dict(self.model.named_parameters()),
                           show_attrs=True, show_saved=True)
            graph.attr(rankdir='TB')
            graph.attr('node', shape='box')
            
            if save_path:
                base_path = Path(save_path).stem
                graph.render(base_path, format='png', cleanup=True)
                logger.info(f"Computational graph saved to {base_path}.png")
            
            return graph
        except Exception as e:
            logger.error(f"Failed to create computational graph: {e}")
            return None
    
    def create_parameter_distribution_plot(self, save_path: Optional[str] = None,
                                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Create parameter distribution visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Collect all parameters
        all_params = []
        layer_params = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                all_params.extend(param.data.cpu().numpy().flatten())
                layer_params[name] = param.numel()
        
        all_params = np.array(all_params)
        
        # Parameter value distribution
        ax1.hist(all_params, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Parameter Value Distribution')
        ax1.set_xlabel('Parameter Value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Parameter count per layer
        layer_names = list(layer_params.keys())[:10]  # Top 10 layers
        layer_counts = [layer_params[name] for name in layer_names]
        
        ax2.barh(range(len(layer_names)), layer_counts, color='lightcoral')
        ax2.set_yticks(range(len(layer_names)))
        ax2.set_yticklabels([name.split('.')[-1] for name in layer_names], fontsize=8)
        ax2.set_title('Parameters per Layer (Top 10)')
        ax2.set_xlabel('Parameter Count')
        ax2.grid(True, alpha=0.3)
        
        # Parameter statistics
        stats_text = f"""
        Total Parameters: {len(all_params):,}
        Mean: {np.mean(all_params):.6f}
        Std: {np.std(all_params):.6f}
        Min: {np.min(all_params):.6f}
        Max: {np.max(all_params):.6f}
        """
        ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='center', fontfamily='monospace')
        ax3.set_title('Parameter Statistics')
        ax3.axis('off')
        
        # Layer type distribution
        layer_types = {}
        for name, module in self.model.named_modules():
            module_type = type(module).__name__
            if module_type not in layer_types:
                layer_types[module_type] = 0
            layer_types[module_type] += 1
        
        # Remove 'Module' and other parent classes
        layer_types = {k: v for k, v in layer_types.items() 
                      if k not in ['Module', self.model.__class__.__name__] and v > 0}
        
        if layer_types:
            ax4.pie(layer_types.values(), labels=layer_types.keys(), autopct='%1.1f%%',
                   startangle=90)
            ax4.set_title('Layer Type Distribution')
        else:
            ax4.text(0.5, 0.5, 'No layer data available', transform=ax4.transAxes,
                    ha='center', va='center')
            ax4.set_title('Layer Type Distribution')
        
        plt.suptitle(f'{self.model_name} - Parameter Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Parameter distribution plot saved to {save_path}")
        
        return fig
    
    def export_architecture_summary(self, save_path: str) -> Dict[str, Any]:
        """Export comprehensive architecture summary to JSON"""
        summary = {
            'model_name': self.model_name,
            'model_class': self.model.__class__.__name__,
            'total_parameters': self.layers_info['total_params'],
            'layer_analysis': self.layers_info,
            'architecture_type': self._detect_architecture_type(),
            'model_size_mb': self.layers_info['total_params'] * 4 / (1024 * 1024),  # Assuming float32
        }
        
        # Add layer summary
        layer_summary = {}
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                layer_summary[name] = {
                    'type': type(module).__name__,
                    'parameters': sum(p.numel() for p in module.parameters()),
                    'input_shape': getattr(module, 'in_channels', None) or getattr(module, 'in_features', None),
                    'output_shape': getattr(module, 'out_channels', None) or getattr(module, 'out_features', None),
                }
        
        summary['detailed_layers'] = layer_summary
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Architecture summary exported to {save_path}")
        return summary
    
    def _detect_architecture_type(self) -> str:
        """Detect the type of architecture (UNet, ResNet, etc.)"""
        model_name = self.model.__class__.__name__.lower()
        
        if 'unet' in model_name:
            return 'U-Net'
        elif 'resnet' in model_name or 'res' in model_name:
            return 'ResNet-based'
        elif 'attention' in model_name:
            return 'Attention-based'
        elif 'transformer' in model_name:
            return 'Transformer'
        else:
            return 'Custom'
    
    def generate_all_visualizations(self, output_dir: str, 
                                  input_tensor: Optional[torch.Tensor] = None) -> Dict[str, str]:
        """Generate all available visualizations and save to output directory"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        
        # Architecture diagram
        try:
            arch_path = output_path / f"{self.model_name}_architecture.png"
            self.create_unet_architecture_diagram(str(arch_path))
            generated_files['architecture'] = str(arch_path)
        except Exception as e:
            logger.error(f"Failed to generate architecture diagram: {e}")
        
        # Detailed layer diagram
        try:
            layer_path = output_path / f"{self.model_name}_layers.png"
            self.create_detailed_layer_diagram(str(layer_path))
            generated_files['layers'] = str(layer_path)
        except Exception as e:
            logger.error(f"Failed to generate layer diagram: {e}")
        
        # Parameter distribution
        try:
            param_path = output_path / f"{self.model_name}_parameters.png"
            self.create_parameter_distribution_plot(str(param_path))
            generated_files['parameters'] = str(param_path)
        except Exception as e:
            logger.error(f"Failed to generate parameter plot: {e}")
        
        # Computational graph (if input tensor provided)
        if input_tensor is not None:
            try:
                graph_path = output_path / f"{self.model_name}_graph"
                self.create_computational_graph(input_tensor, str(graph_path))
                generated_files['computational_graph'] = str(graph_path) + ".png"
            except Exception as e:
                logger.error(f"Failed to generate computational graph: {e}")
        
        # Architecture summary
        try:
            summary_path = output_path / f"{self.model_name}_summary.json"
            self.export_architecture_summary(str(summary_path))
            generated_files['summary'] = str(summary_path)
        except Exception as e:
            logger.error(f"Failed to generate architecture summary: {e}")
        
        logger.info(f"Generated {len(generated_files)} visualization files in {output_dir}")
        return generated_files


def visualize_model(model: nn.Module, model_name: str = "Model", 
                   output_dir: str = "visualizations",
                   input_shape: Tuple[int, ...] = (1, 1, 256, 256)) -> Dict[str, str]:
    """
    Convenience function to visualize any model
    
    Args:
        model: PyTorch model to visualize
        model_name: Name for the model (used in titles and filenames)
        output_dir: Directory to save visualizations
        input_shape: Input tensor shape for computational graph
    
    Returns:
        Dictionary mapping visualization types to file paths
    """
    visualizer = ModelArchitectureVisualizer(model, model_name)
    
    # Create sample input tensor
    input_tensor = torch.randn(input_shape)
    
    return visualizer.generate_all_visualizations(output_dir, input_tensor)


def compare_models(models: Dict[str, nn.Module], output_dir: str = "model_comparison") -> Dict[str, Any]:
    """
    Compare multiple models and create comparison visualizations
    
    Args:
        models: Dictionary mapping model names to model instances
        output_dir: Directory to save comparison plots
    
    Returns:
        Comparison data and file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    comparison_data = {}
    
    # Analyze each model
    for name, model in models.items():
        visualizer = ModelArchitectureVisualizer(model, name)
        comparison_data[name] = {
            'total_params': visualizer.layers_info['total_params'],
            'encoder_layers': len(visualizer.layers_info['encoder_layers']),
            'decoder_layers': len(visualizer.layers_info['decoder_layers']),
            'attention_layers': len(visualizer.layers_info['attention_layers']),
            'architecture_type': visualizer._detect_architecture_type()
        }
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    model_names = list(models.keys())
    param_counts = [comparison_data[name]['total_params'] for name in model_names]
    
    # Parameter count comparison
    bars = ax1.bar(model_names, param_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(model_names)])
    ax1.set_title('Model Parameter Count Comparison')
    ax1.set_ylabel('Parameters')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, param_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({count/1e6:.1f}M)' if count > 1e6 else f'{count:,}',
                ha='center', va='bottom', fontsize=9)
    
    # Layer count comparison
    encoder_counts = [comparison_data[name]['encoder_layers'] for name in model_names]
    decoder_counts = [comparison_data[name]['decoder_layers'] for name in model_names]
    attention_counts = [comparison_data[name]['attention_layers'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    ax2.bar(x - width, encoder_counts, width, label='Encoder Layers', color='#FF6B6B')
    ax2.bar(x, decoder_counts, width, label='Decoder Layers', color='#4ECDC4')
    ax2.bar(x + width, attention_counts, width, label='Attention Layers', color='#FFA07A')
    
    ax2.set_title('Layer Count Comparison')
    ax2.set_ylabel('Number of Layers')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45)
    ax2.legend()
    
    # Model size (MB)
    model_sizes = [param_counts[i] * 4 / (1024 * 1024) for i in range(len(model_names))]
    ax3.pie(model_sizes, labels=model_names, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Relative Model Sizes (MB)')
    
    # Architecture types
    arch_types = [comparison_data[name]['architecture_type'] for name in model_names]
    arch_counts = {}
    for arch in arch_types:
        arch_counts[arch] = arch_counts.get(arch, 0) + 1
    
    ax4.bar(arch_counts.keys(), arch_counts.values(), color='lightgreen')
    ax4.set_title('Architecture Type Distribution')
    ax4.set_ylabel('Count')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Model Architecture Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = output_path / "model_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    
    # Save comparison data
    data_path = output_path / "comparison_data.json"
    with open(data_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    logger.info(f"Model comparison saved to {output_dir}")
    
    return {
        'comparison_data': comparison_data,
        'comparison_plot': str(comparison_path),
        'data_file': str(data_path)
    }
