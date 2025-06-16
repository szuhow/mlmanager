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

logger = logging.getLogger(__name__)


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


# Global registry instance
registry = ModelArchitectureRegistry()


def get_available_models() -> List[Tuple[str, str]]:
    """Get available models for Django forms (backward compatibility)"""
    return registry.get_choices()


def setup_default_architectures():
    """Set up default architectures"""
    base_dir = Path(__file__).parent
    
    # Add discovery paths
    registry.add_discovery_path(base_dir / 'unet')
    registry.add_discovery_path(base_dir / 'unet-old')
    
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
        unet_path = base_dir / 'unet' / 'unet_model.py'
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
                        'n_channels': 3,  # Updated for RGB input
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
                registry.register_from_module(
                    unet_path,
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
        # ResUNet models are in ml/training/models/, not ml/utils/resunet/
        training_models_path = base_dir.parent / 'training' / 'models' / 'resunet_model.py'
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
        else:
            logger.warning(f"ResUNet models not found at {training_models_path}")
    
    except Exception as e:
        logger.error(f"Error setting up default architectures: {e}")
    
    # Discover additional architectures
    try:
        discovered = registry.discover_architectures()
        logger.info(f"Discovered {discovered} additional architectures")
    except Exception as e:
        logger.error(f"Error during architecture discovery: {e}")


# Initialize default architectures when module is imported
setup_default_architectures()
