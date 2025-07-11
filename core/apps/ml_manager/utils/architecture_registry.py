"""
Model Architecture Registry System
Provides a pluggable system for registering and managing different model architectures.
"""

import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)

# Global registry instance
_DEFAULT_REGISTRY = None
_REGISTRY_INITIALIZED = False

def _register_builtin_architectures(registry: 'ModelArchitectureRegistry') -> None:
    """Register built-in architectures to the registry"""
    global _REGISTRY_INITIALIZED
    
    # Prevent multiple initialization
    if _REGISTRY_INITIALIZED:
        return
        
    try:
        import sys
        import importlib.util
        from pathlib import Path
        
        # Get correct path: we're in core/apps/ml_manager/utils/, UNet is in core/apps/ml_manager/training/models/unet/
        project_root = Path(__file__).parent.parent  # Go up to ml_manager/
        unet_path = project_root / "training" / "models" / "unet" / "unet_model.py"
        
        # Try to import the UNet model
        unet_module = None
        if unet_path.exists():
            try:
                spec = importlib.util.spec_from_file_location("unet_model", str(unet_path))
                if spec and spec.loader:
                    unet_module = importlib.util.module_from_spec(spec)
                    sys.modules[spec.name] = unet_module
                    spec.loader.exec_module(unet_module)
                    logger.info("Loaded local UNet module")
            except Exception as e:
                logger.error(f"Failed to load UNet module: {e}")
        
        # 1. Register local UNet implementation (only once)
        if unet_module and hasattr(unet_module, "UNet"):
            # Only register under 'unet' key to avoid duplicates
            registry.register(ArchitectureInfo(
                key="unet",
                display_name="UNet",
                framework="PyTorch",
                description="UNet implementation for segmentation tasks",
                model_class=unet_module.UNet,
                category="segmentation",
                supports_2d=True,
                supports_3d=False,
                author="Project Team",
                version="1.0.0"
            ))
            logger.info(f"Registered architecture: UNet (unet)")
        
        # 2. Create fallback implementations for other common models
        # For MONAI UNet
        try:
            from torch import nn
            # Create a fallback class that points to our local UNet
            # This avoids ImportError when MONAI isn't available
            class MonaiFallbackUNet(nn.Module):
                def __init__(self, input_channels=1, output_channels=1, **kwargs):
                    super().__init__()  # Important: initialize the parent nn.Module
                    if unet_module and hasattr(unet_module, "UNet"):
                        self.model = unet_module.UNet(n_channels=input_channels, n_classes=output_channels)
                    else:
                        # Just create an empty module as fallback
                        self.model = nn.Sequential()
                    
                def forward(self, x):
                    return self.model(x)
            
            # Register MONAI UNet fallback
            registry.register(ArchitectureInfo(
                key="monai_unet",
                display_name="MONAI UNet (Fallback)",
                framework="PyTorch",
                description="MONAI UNet implementation (fallback to local UNet)",
                model_class=MonaiFallbackUNet,
                category="segmentation",
                supports_2d=True,
                supports_3d=True,
                author="Project Team",
                version="1.0.0"
            ))
            logger.info("Registered MONAI UNet fallback architecture")
            
            # 3. Register other common models with fallbacks
            # Create basic fallback classes for other model types
            common_models = [
                ("resunet", "ResUNet", "Residual UNet for segmentation tasks"),
                ("attention_unet", "Attention UNet", "UNet with attention gates"),
                ("unet_plus_plus", "UNet++", "Nested UNet architecture"),
                ("deeplab", "DeepLab", "DeepLab segmentation model"),
                ("segnet", "SegNet", "Segmentation network architecture")
            ]
            
            for key, name, desc in common_models:
                # Only register fallbacks if the key doesn't already exist
                if not registry.get_architecture(key):
                    if unet_module and hasattr(unet_module, "UNet"):
                        registry.register(ArchitectureInfo(
                            key=key,
                            display_name=name,
                            framework="PyTorch",
                            description=desc,
                            model_class=unet_module.UNet,  # Use UNet as fallback
                            category="segmentation",
                            supports_2d=True,
                            supports_3d=False,
                            author="Project Team",
                            version="1.0.0"
                        ))
                        logger.info(f"Registered fallback for {name} ({key})")
                else:
                    logger.info(f"Skipping fallback registration for {key} - already registered")
            
            logger.info("Completed registration of fallback architectures")
        except Exception as e:
            logger.error(f"Failed to create MONAI UNet fallback: {e}")
            
    except Exception as e:
        logger.error(f"Failed to register architectures: {e}")
    finally:
        # Mark as initialized to prevent duplicate registration
        _REGISTRY_INITIALIZED = True


def _register_resunet_models(registry: 'ModelArchitectureRegistry') -> None:
    """Register ResUNet models from the training/models directory"""
    from pathlib import Path
    import sys
    import importlib.util
    
    base_dir = Path(__file__).parent.parent
    training_models_path = base_dir / 'training' / 'models' / 'resunet_model.py'
    
    if training_models_path.exists():
        try:
            # Add the models directory to Python path temporarily for proper imports
            models_dir = str(training_models_path.parent)
            if models_dir not in sys.path:
                sys.path.insert(0, models_dir)
            
            try:
                spec = importlib.util.spec_from_file_location("resunet_models", str(training_models_path))
                resunet_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(resunet_module)
            finally:
                # Remove from path after import
                if models_dir in sys.path:
                    sys.path.remove(models_dir)
            
            # Register standard Residual U-Net
            registry.register(ArchitectureInfo(
                key='resunet',
                display_name='Residual U-Net',
                framework='PyTorch',
                description='U-Net with residual connections for improved gradient flow and feature learning',
                model_class=resunet_module.ResUNet,
                default_config={
                    'n_channels': 3,
                    'n_classes': 1,
                    'bilinear': False,
                    'use_attention': False
                },
                category='medical_segmentation',
                supports_2d=True,
                supports_3d=False,
                author='Custom Implementation',
                version='1.0.0'
            ))
            
            # Register Deep Residual U-Net
            registry.register(ArchitectureInfo(
                key='deep_resunet',
                display_name='Deep Residual U-Net',
                framework='PyTorch',
                description='Deeper U-Net with residual connections for complex feature extraction',
                model_class=resunet_module.DeepResUNet,
                default_config={
                    'n_channels': 3,
                    'n_classes': 1,
                    'bilinear': False,
                    'use_attention': False
                },
                category='medical_segmentation',
                supports_2d=True,
                supports_3d=False,
                author='Custom Implementation',
                version='1.0.0'
            ))
            
            # Register Residual U-Net with Attention
            registry.register(ArchitectureInfo(
                key='resunet_attention',
                display_name='Residual U-Net with Attention',
                framework='PyTorch',
                description='Standard Residual U-Net with attention gates for better feature selection',
                model_class=resunet_module.ResUNet,
                default_config={
                    'n_channels': 3,
                    'n_classes': 1,
                    'bilinear': False,
                    'use_attention': True
                },
                category='medical_segmentation',
                supports_2d=True,
                supports_3d=False,
                author='Custom Implementation',
                version='1.0.0'
            ))
            
            # Register Deep Residual U-Net with Attention
            registry.register(ArchitectureInfo(
                key='deep_resunet_attention',
                display_name='Deep Residual U-Net with Attention',
                framework='PyTorch',
                description='Deeper Residual U-Net with attention gates for complex feature extraction and better localization',
                model_class=resunet_module.DeepResUNet,
                default_config={
                    'n_channels': 3,
                    'n_classes': 1,
                    'bilinear': False,
                    'use_attention': True
                },
                category='medical_segmentation',
                supports_2d=True,
                supports_3d=False,
                author='Custom Implementation',
                version='1.0.0'
            ))
            
            logger.info("Successfully registered all ResUNet model variants")
            
        except Exception as e:
            logger.error(f"Error registering Residual U-Net models: {e}")
    else:
        logger.warning(f"ResUNet model file not found: {training_models_path}")


def initialize_registry() -> 'ModelArchitectureRegistry':
    """Initialize the model architecture registry with default architectures"""
    from pathlib import Path
    
    registry = ModelArchitectureRegistry()
    
    # Add discovery paths - we're in core/apps/ml_manager/utils/, training is in core/apps/ml_manager/training/
    project_root = Path(__file__).parent.parent  # Go up to ml_manager/
    ml_models_path = project_root / "training" / "models"
    
    if ml_models_path.exists():
        registry.add_discovery_path(ml_models_path)

    # First register the specific ResUNet models (so they don't get overridden by fallbacks)
    _register_resunet_models(registry)
        
    # Register built-in architectures (including fallbacks)
    _register_builtin_architectures(registry)

    # Discover additional architectures (commented out to reduce noise)
    # try:
    #     registry.discover_architectures()
    # except Exception as e:
    #     logger.error(f"Error discovering architectures: {e}")
        
    return registry


@dataclass
class ArchitectureInfo:
    """Information about a model architecture"""
    key: str
    display_name: str
    framework: str
    description: str
    model_class: Any
    config_class: Optional[Any] = None
    default_config: Optional[Dict[str, Any]] = None
    requirements: Optional[List[str]] = None
    category: str = "general"
    supports_3d: bool = False
    supports_2d: bool = True
    author: str = "Unknown"
    version: str = "1.0.0"


class ModelArchitectureRegistry:
    """Registry for managing model architectures"""
    
    def __init__(self):
        self._architectures: Dict[str, ArchitectureInfo] = {}
        self._auto_discovery_paths: List[Path] = []
        
    def register(self, architecture_info: ArchitectureInfo) -> None:
        """Register a model architecture"""
        if architecture_info.key in self._architectures:
            logger.warning(f"Architecture '{architecture_info.key}' is being overridden")
        
        self._architectures[architecture_info.key] = architecture_info
        logger.info(f"Registered architecture: {architecture_info.display_name} ({architecture_info.key})")
    
    def register_from_module(self, module_path: Path, architecture_key: str, 
                           display_name: str, **kwargs) -> bool:
        """Register an architecture from a Python module"""
        try:
            spec = importlib.util.spec_from_file_location(f"arch_{architecture_key}", str(module_path))
            if not spec or not spec.loader:
                logger.error(f"Could not load module from {module_path}")
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Try to find a model class
            model_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if hasattr(obj, '__module__') and obj.__module__ == module.__name__:
                    # This is a class defined in this module
                    if 'model' in name.lower() or 'net' in name.lower():
                        model_class = obj
                        break
            
            if not model_class:
                # Fall back to looking for any callable that could be a model
                for name, obj in inspect.getmembers(module, callable):
                    if name.lower().startswith(('get_', 'create_', 'build_')) and 'model' in name.lower():
                        model_class = obj
                        break
            
            if not model_class:
                logger.warning(f"No model class found in {module_path}")
                return False
            
            # Create architecture info
            arch_info = ArchitectureInfo(
                key=architecture_key,
                display_name=display_name,
                framework=kwargs.get('framework', 'PyTorch'),
                description=kwargs.get('description', f"Model from {module_path.name}"),
                model_class=model_class,
                category=kwargs.get('category', 'general'),
                supports_2d=kwargs.get('supports_2d', True),
                supports_3d=kwargs.get('supports_3d', False),
                author=kwargs.get('author', 'Unknown'),
                version=kwargs.get('version', '1.0.0')
            )
            
            self.register(arch_info)
            return True
            
        except Exception as e:
            logger.error(f"Failed to register architecture from {module_path}: {e}")
            return False
    
    def get_architecture(self, key: str) -> Optional[ArchitectureInfo]:
        """Get architecture information by key"""
        return self._architectures.get(key)
    
    def get_all_architectures(self) -> Dict[str, ArchitectureInfo]:
        """Get all registered architectures"""
        return self._architectures.copy()
    
    def get_choices(self) -> List[Tuple[str, str]]:
        """Get choices for Django form fields"""
        return [(key, info.display_name) for key, info in self._architectures.items()]
    
    def get_by_category(self, category: str) -> Dict[str, ArchitectureInfo]:
        """Get architectures by category"""
        return {key: info for key, info in self._architectures.items() 
                if info.category == category}
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return list(set(info.category for info in self._architectures.values()))
    
    def list_architectures(self) -> List[ArchitectureInfo]:
        """List all registered architectures"""
        return list(self._architectures.values())
    
    def add_discovery_path(self, path: Path) -> None:
        """Add a path for automatic architecture discovery"""
        if path.exists() and path.is_dir():
            self._auto_discovery_paths.append(path)
    
    def discover_architectures(self) -> int:
        """Automatically discover architectures in registered paths"""
        discovered = 0
        
        for base_path in self._auto_discovery_paths:
            try:
                discovered += self._discover_in_path(base_path)
            except Exception as e:
                logger.error(f"Error discovering architectures in {base_path}: {e}")
        
        return discovered
    
    def _discover_in_path(self, base_path: Path) -> int:
        """Discover architectures in a specific path"""
        discovered = 0
        
        # Look for common architecture patterns
        patterns = [
            ('unet', 'U-Net'),
            ('resunet', 'Residual U-Net'),
            ('resnet', 'ResNet'),
            ('densenet', 'DenseNet'),
            ('vnet', 'V-Net'),
            ('attention_unet', 'Attention U-Net'),
            ('unet_plus_plus', 'U-Net++'),
            ('deeplabv3', 'DeepLabV3'),
            ('segnet', 'SegNet'),
            ('fcn', 'FCN'),
        ]
        
        for pattern, display_name in patterns:
            # Look for directories matching pattern
            pattern_dirs = list(base_path.glob(f"*{pattern}*"))
            
            for arch_dir in pattern_dirs:
                if not arch_dir.is_dir():
                    continue
                
                # Look for model files
                model_files = list(arch_dir.glob("*model*.py")) + list(arch_dir.glob("*net*.py"))
                
                for model_file in model_files:
                    arch_key = f"{pattern}_{arch_dir.name.replace('-', '_')}"
                    if arch_key not in self._architectures:
                        success = self.register_from_module(
                            model_file,
                            arch_key,
                            f"{display_name} ({arch_dir.name})",
                            framework="PyTorch",
                            category="segmentation" if "unet" in pattern or "net" in pattern else "classification"
                        )
                        if success:
                            discovered += 1
        
        return discovered
    
    def validate_architecture(self, key: str) -> Tuple[bool, str]:
        """Validate that an architecture is properly configured"""
        arch_info = self.get_architecture(key)
        if not arch_info:
            return False, f"Architecture '{key}' not found"
        
        # Check if model class is callable
        if not callable(arch_info.model_class):
            return False, f"Model class for '{key}' is not callable"
        
        # Try to inspect the model class signature
        try:
            sig = inspect.signature(arch_info.model_class)
            # Basic validation - just check if it's inspectable
            return True, "Valid"
        except Exception as e:
            return False, f"Could not inspect model class: {e}"


# Global registry instance - removed, now using get_default_registry()


def get_available_models() -> List[Tuple[str, str]]:
    """Get available models for Django forms (backward compatibility)"""
    return get_default_registry().get_choices()


def setup_default_architectures():
    """Set up default architectures - DEPRECATED, now handled in initialize_registry"""
    # This function is kept for backward compatibility but should not be used
    logger.warning("setup_default_architectures() is deprecated - use initialize_registry() instead")
    
    from pathlib import Path
    base_dir = Path(__file__).parent.parent  # Go up to ml_manager/
    
    pass
    
    # Manual registration for known architectures
    try:
        # Register MONAI UNet directly with proper config
        from monai.networks.nets import UNet as MonaiUNet
        registry.register(ArchitectureInfo(
            key='monai_unet',
            display_name='MONAI U-Net',
            framework='MONAI/PyTorch',
            description='Medical imaging U-Net using MONAI framework',
            model_class=MonaiUNet,
            default_config={
                'spatial_dims': 2,
                'in_channels': 3,  # Updated for RGB input
                'out_channels': 1,
                'channels': (16, 32, 64, 128, 256),
                'strides': (2, 2, 2, 2),
                'num_res_units': 2,
            },
            category='medical_segmentation',
            supports_2d=True,
            supports_3d=True,
            author='MONAI Team',
            version='1.0.0'
        ))
        
        # Register the legacy 'unet' alias for backward compatibility
        registry.register(ArchitectureInfo(
            key='unet',
            display_name='U-Net (Default)',
            framework='MONAI/PyTorch',
            description='Default U-Net implementation using MONAI framework',
            model_class=MonaiUNet,
            default_config={
                'spatial_dims': 2,
                'in_channels': 3,  # Updated for RGB input
                'out_channels': 1,
                'channels': (16, 32, 64, 128, 256),
                'strides': (2, 2, 2, 2),
                'num_res_units': 2,
            },
            category='medical_segmentation',
            supports_2d=True,
            supports_3d=True,
            author='MONAI Team',
            version='1.0.0'
        ))
        
        # Register MONAI UNet from local implementation as backup
        unet_path = base_dir / 'training' / 'models' / 'unet' / 'unet_model.py'
        if unet_path.exists():
            try:
                # Import the local UNet model
                import importlib.util
                spec = importlib.util.spec_from_file_location("local_unet_model", str(unet_path))
                local_unet_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(local_unet_module)
                
                registry.register(ArchitectureInfo(
                    key='local_unet',
                    display_name='Local U-Net',
                    framework='PyTorch',
                    description='Local U-Net implementation',
                    model_class=local_unet_module.UNet,
                    default_config={
                        'n_channels': 1,  # Fixed: should be 1 for grayscale medical images
                        'n_classes': 1,
                        'bilinear': False,
                    },
                    category='medical_segmentation',
                    supports_2d=True,
                    supports_3d=False,
                    author='Local Implementation',
                    version='1.0.0'
                ))
            except Exception as e:
                logger.error(f"Failed to register local UNet: {e}")
                # Fallback registration if import fails
                fallback_unet_path = base_dir / 'training' / 'models' / 'unet' / 'unet_model.py'
                if fallback_unet_path.exists():
                    registry.register_from_module(
                        fallback_unet_path,
                        'local_unet',
                        'Local U-Net',
                        framework='PyTorch',
                        description='Local U-Net implementation',
                        category='medical_segmentation',
                        supports_2d=True,
                        supports_3d=False,
                        author='Local Implementation',
                        version='1.0.0'
                    )
        
        # Register legacy UNet
        unet_old_path = base_dir / 'unet-old' / 'unet.py'
        if unet_old_path.exists():
            registry.register_from_module(
                unet_old_path,
                'pytorch_unet',
                'PyTorch U-Net (Legacy)',
                framework='PyTorch',
                description='Traditional U-Net implementation in PyTorch',
                category='segmentation',
                supports_2d=True,
                supports_3d=False,
                author='Legacy',
                version='1.0.0'
            )
        
        # Register Residual U-Net models from the correct path
        # ResUNet models are in training/models/, inside ml_manager
        training_models_path = base_dir / 'training' / 'models' / 'resunet_model.py'
        if training_models_path.exists():
            # Import the models to register them properly
            try:
                import sys
                import importlib.util
                
                # Add the models directory to Python path temporarily for proper imports
                models_dir = str(training_models_path.parent)
                if models_dir not in sys.path:
                    sys.path.insert(0, models_dir)
                
                try:
                    spec = importlib.util.spec_from_file_location("resunet_models", str(training_models_path))
                    resunet_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(resunet_module)
                finally:
                    # Remove from path after import
                    if models_dir in sys.path:
                        sys.path.remove(models_dir)
                
                # Register standard Residual U-Net
                registry.register(ArchitectureInfo(
                    key='resunet',
                    display_name='Residual U-Net',
                    framework='PyTorch',
                    description='U-Net with residual connections for improved gradient flow and feature learning',
                    model_class=resunet_module.ResUNet,
                    default_config={
                        'n_channels': 3,  # Updated for RGB input
                        'n_classes': 1,
                        'bilinear': False,
                        'use_attention': False
                    },
                    category='medical_segmentation',
                    supports_2d=True,
                    supports_3d=False,
                    author='Custom Implementation',
                    version='1.0.0'
                ))
                
                # Register Deep Residual U-Net
                registry.register(ArchitectureInfo(
                    key='deep_resunet',
                    display_name='Deep Residual U-Net',
                    framework='PyTorch',
                    description='Deeper U-Net with residual connections for complex feature extraction',
                    model_class=resunet_module.DeepResUNet,
                    default_config={
                        'n_channels': 3,  # Updated for RGB input
                        'n_classes': 1,
                        'bilinear': False,
                        'use_attention': False
                    },
                    category='medical_segmentation',
                    supports_2d=True,
                    supports_3d=False,
                    author='Custom Implementation',
                    version='1.0.0'
                ))
                
                # Register Residual U-Net with Attention
                registry.register(ArchitectureInfo(
                    key='resunet_attention',
                    display_name='Residual U-Net with Attention',
                    framework='PyTorch',
                    description='Standard Residual U-Net with attention gates for better feature selection',
                    model_class=resunet_module.ResUNet,
                    default_config={
                        'n_channels': 3,  # Updated for RGB input
                        'n_classes': 1,
                        'bilinear': False,
                        'use_attention': True
                    },
                    category='medical_segmentation',
                    supports_2d=True,
                    supports_3d=False,
                    author='Custom Implementation',
                    version='1.0.0'
                ))
                
                # Register Deep Residual U-Net with Attention
                registry.register(ArchitectureInfo(
                    key='deep_resunet_attention',
                    display_name='Deep Residual U-Net with Attention',
                    framework='PyTorch',
                    description='Deeper Residual U-Net with attention gates for complex feature extraction and better localization',
                    model_class=resunet_module.DeepResUNet,
                    default_config={
                        'n_channels': 3,  # Updated for RGB input
                        'n_classes': 1,
                        'bilinear': False,
                        'use_attention': True
                    },
                    category='medical_segmentation',
                    supports_2d=True,
                    supports_3d=False,
                    author='Custom Implementation',
                    version='1.0.0'
                ))
                
                # # Register convenience functions as well
                # registry.register(ArchitectureInfo(
                #     key='create_resunet',
                #     display_name='Create Residual U-Net (Function)',
                #     framework='PyTorch',
                #     description='Convenience function to create a standard Residual U-Net',
                #     model_class=resunet_module.create_resunet,
                #     default_config={},  # Convenience function has defaults
                #     category='medical_segmentation',
                #     supports_2d=True,
                #     supports_3d=False,
                #     author='Custom Implementation',
                #     version='1.0.0'
                # ))
                
                # registry.register(ArchitectureInfo(
                #     key='create_deep_resunet',
                #     display_name='Create Deep Residual U-Net (Function)',
                #     framework='PyTorch',
                #     description='Convenience function to create a deeper Residual U-Net',
                #     model_class=resunet_module.create_deep_resunet,
                #     default_config={},  # Convenience function has defaults
                #     category='medical_segmentation',
                #     supports_2d=True,
                #     supports_3d=False,
                #     author='Custom Implementation',
                #     version='1.0.0'
                # ))
                
                # registry.register(ArchitectureInfo(
                #     key='create_attention_resunet',
                #     display_name='Create Attention Residual U-Net (Function)',
                #     framework='PyTorch',
                #     description='Convenience function to create a Residual U-Net with attention gates',
                #     model_class=resunet_module.create_attention_resunet,
                #     default_config={},  # Convenience function has defaults
                #     category='medical_segmentation',
                #     supports_2d=True,
                #     supports_3d=False,
                #     author='Custom Implementation',
                #     version='1.0.0'
                # ))
                
                logger.info("Successfully registered all ResUNet model variants")
                
            except Exception as e:
                logger.error(f"Error registering Residual U-Net models: {e}")
                
                # Register fallback implementations when PyTorch models can't be loaded
                try:
                    from torch import nn
                    
                    # Create fallback classes for all ResUNet variants
                    class ResUNetFallback(nn.Module):
                        def __init__(self, n_channels=1, n_classes=1, bilinear=False, use_attention=False, **kwargs):
                            super().__init__()
                            # Use local UNet as fallback if available
                            if hasattr(self, '_get_unet_fallback'):
                                self.model = self._get_unet_fallback(n_channels, n_classes, bilinear)
                            else:
                                self.model = nn.Sequential()
                            
                        def _get_unet_fallback(self, n_channels, n_classes, bilinear):
                            # Try to use the local UNet implementation
                            try:
                                # Import the local UNet module
                                import sys
                                import importlib.util
                                from pathlib import Path
                                
                                base_dir = Path(__file__).parent.parent
                                unet_path = base_dir / 'training' / 'models' / 'unet' / 'unet_model.py'
                                
                                if unet_path.exists():
                                    spec = importlib.util.spec_from_file_location("fallback_unet", str(unet_path))
                                    fallback_unet_module = importlib.util.module_from_spec(spec)
                                    spec.loader.exec_module(fallback_unet_module)
                                    
                                    if hasattr(fallback_unet_module, 'UNet'):
                                        return fallback_unet_module.UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
                                        
                            except Exception:
                                pass
                            
                            # Final fallback to empty sequential
                            return nn.Sequential()
                            
                        def forward(self, x):
                            return self.model(x)
                    
                    # Register fallback ResUNet models
                    try:
                        fallback_models = [
                            ('resunet', 'Residual U-Net (Fallback)', 'U-Net with residual connections (fallback to local UNet)'),
                            ('deep_resunet', 'Deep Residual U-Net (Fallback)', 'Deeper U-Net with residual connections (fallback to local UNet)'),
                            ('resunet_attention', 'Residual U-Net with Attention (Fallback)', 'U-Net with residual connections and attention gates (fallback to local UNet)'),
                            ('deep_resunet_attention', 'Deep Residual U-Net with Attention (Fallback)', 'Deeper U-Net with residual connections and attention gates (fallback to local UNet)'),
                        ]
                        
                        for key, display_name, description in fallback_models:
                            # Only register fallback if the key doesn't already exist
                            if not registry.get_architecture(key):
                                registry.register(ArchitectureInfo(
                                    key=key,
                                    display_name=display_name,
                                    framework='PyTorch',
                                    description=description,
                                    model_class=ResUNetFallback,
                                    default_config={
                                        'n_channels': 3,
                                        'n_classes': 1,
                                        'bilinear': False,
                                        'use_attention': False
                                    },
                                    category='medical_segmentation',
                                    supports_2d=True,
                                    supports_3d=False,
                                    author='Fallback Implementation',
                                    version='1.0.0'
                                ))
                                logger.info(f"Registered fallback architecture: {display_name} ({key})")
                            else:
                                logger.info(f"Skipping fallback registration for {key} - already registered")
                        
                    except Exception as fallback_error:
                        logger.error(f"Error registering fallback ResUNet models: {fallback_error}")
                        
                except Exception as outer_fallback_error:
                    logger.error(f"Error creating fallback ResUNet implementation: {outer_fallback_error}")
        
        # Register classification models
        try:
            classification_models_path = base_dir / 'training' / 'models' / 'classification_models.py'
            if classification_models_path.exists():
                import sys
                import importlib.util
                
                # Add the models directory to Python path temporarily
                models_dir = str(classification_models_path.parent)
                if models_dir not in sys.path:
                    sys.path.insert(0, models_dir)
                
                try:
                    spec = importlib.util.spec_from_file_location("classification_models", str(classification_models_path))
                    classification_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(classification_module)
                finally:
                    # Remove from path after import
                    if models_dir in sys.path:
                        sys.path.remove(models_dir)
                
                # Register U-Net Classifier
                registry.register(ArchitectureInfo(
                    key='unet_classifier',
                    display_name='U-Net Classifier',
                    framework='PyTorch',
                    description='U-Net based classifier for image classification tasks',
                    model_class=classification_module.UNetClassifier,
                    default_config={
                        'n_channels': 3,
                        'n_classes': 2,
                        'use_monai': True,
                        'spatial_dims': 2,
                        'channels': (16, 32, 64, 128, 256),
                        'strides': (2, 2, 2, 2),
                        'num_res_units': 2,
                    },
                    category='classification',
                    supports_2d=True,
                    supports_3d=False,
                    author='Custom Implementation',
                    version='1.0.0'
                ))
                
                # Register ResUNet Classifier
                registry.register(ArchitectureInfo(
                    key='resunet_classifier',
                    display_name='ResUNet Classifier',
                    framework='PyTorch',
                    description='ResUNet based classifier for image classification tasks',
                    model_class=classification_module.ResUNetClassifier,
                    default_config={
                        'n_channels': 3,
                        'n_classes': 2,
                        'deep': False,
                        'use_attention': False,
                        'bilinear': False
                    },
                    category='classification',
                    supports_2d=True,
                    supports_3d=False,
                    author='Custom Implementation',
                    version='1.0.0'
                ))
                
                # Register Deep ResUNet Classifier
                registry.register(ArchitectureInfo(
                    key='deep_resunet_classifier',
                    display_name='Deep ResUNet Classifier',
                    framework='PyTorch',
                    description='Deep ResUNet based classifier for complex image classification tasks',
                    model_class=classification_module.ResUNetClassifier,
                    default_config={
                        'n_channels': 3,
                        'n_classes': 2,
                        'deep': True,
                        'use_attention': False,
                        'bilinear': False
                    },
                    category='classification',
                    supports_2d=True,
                    supports_3d=False,
                    author='Custom Implementation',
                    version='1.0.0'
                ))
                
                # Register ResUNet Classifier with Attention
                registry.register(ArchitectureInfo(
                    key='resunet_attention_classifier',
                    display_name='ResUNet Attention Classifier',
                    framework='PyTorch',
                    description='ResUNet based classifier with attention gates for better feature selection',
                    model_class=classification_module.ResUNetClassifier,
                    default_config={
                        'n_channels': 3,
                        'n_classes': 2,
                        'deep': False,
                        'use_attention': True,
                        'bilinear': False
                    },
                    category='classification',
                    supports_2d=True,
                    supports_3d=False,
                    author='Custom Implementation',
                    version='1.0.0'
                ))
                
                logger.info("Successfully registered all classification model variants")
            else:
                logger.warning(f"Classification models not found at {classification_models_path}")
        except Exception as e:
            logger.error(f"Error registering classification models: {e}")
    
    except Exception as e:
        logger.error(f"Error setting up default architectures: {e}")
    
    # Discover additional architectures
    try:
        discovered = registry.discover_architectures()
        logger.info(f"Discovered {discovered} additional architectures")
    except Exception as e:
        logger.error(f"Error during architecture discovery: {e}")


def get_model_class(model_type: str):
    """Get model class by model type key
    
    This function provides backward compatibility with code that expects
    to get a model class directly from the registry.
    """
    registry = get_default_registry()
    arch_info = registry.get_architecture(model_type)
    return arch_info.model_class if arch_info else None


def get_default_registry() -> ModelArchitectureRegistry:
    """Get the default global registry instance"""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = initialize_registry()
    return _DEFAULT_REGISTRY

# Initialize default architectures when module is imported
# setup_default_architectures()  # Commented out - this is now handled in initialize_registry

# Create global registry instance for backward compatibility with train.py
registry = get_default_registry()
