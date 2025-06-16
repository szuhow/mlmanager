import os
import logging
# --- Robust logging setup at the very top ---
os.makedirs('data/logs', exist_ok=True)
logging.basicConfig(
    filename='data/logs/training.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    force=True
)
logger = logging.getLogger(__name__)
logger.info('--- Training script started ---')

import sys
import time
import torch
import torch.optim as optim
import mlflow
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import uuid
from datetime import datetime
from torchvision import transforms as tv_transforms  # added for ARCADE transforms
from torch.utils.data import DataLoader as TorchDataLoader  # ARCADE DataLoader

# === Early Django Setup ===
# Setup Django early to avoid import issues with training_callback
try:
    import django
    
    # Ensure the project root is in Python path
    project_root = Path(__file__).resolve().parent.parent.parent
    core_path = str(project_root / 'core')
    ml_path = str(project_root / 'ml')
    
    if core_path not in sys.path:
        sys.path.insert(0, core_path)
    if ml_path not in sys.path:
        sys.path.insert(0, ml_path)
    
    # Set the Django settings module
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
    
    # Setup Django
    try:
        django.setup()
        logger.info("[DJANGO] Django setup completed successfully")
    except RuntimeError as e:
        if "populated" in str(e):
            logger.info("[DJANGO] Django already configured")
        else:
            logger.error(f"[DJANGO] Django setup failed: {e}")
            raise
        
except Exception as e:
    logger.warning(f"[DJANGO] Django setup failed: {e}")
    logger.warning("[DJANGO] Training callback will not be available")
from monai.networks.nets import UNet as MonaiUNet
from monai.data import CacheDataset, DataLoader as MonaiDataLoader
from monai.data import Dataset, DataLoader
from monai.transforms import (
    LoadImage, ScaleIntensity, ToTensor, Compose,
    Resize, EnsureChannelFirst, ConvertToMultiChannelBasedOnBratsClassesd,
    Lambda
)
# Import architecture registry
from ml.utils.architecture_registry import registry as architecture_registry

# Import dynamic learning rate scheduler
from ml.utils.dynamic_lr_scheduler import DynamicLearningRateScheduler

# Import ARCADE dataset integration
try:
    from ml.datasets.torch_arcade_loader import (
        create_arcade_dataloader, 
        get_arcade_dataset_info,
        ARCADEBinarySegmentation,
        ARCADESemanticSegmentation,
        ARCADEStenosisDetection,
        ARCADEArteryClassification
    )
    ARCADE_AVAILABLE = True
    logger.info("[ARCADE] ARCADE dataset integration available")
except ImportError as e:
    ARCADE_AVAILABLE = False
    logger.warning(f"[ARCADE] ARCADE dataset not available: {e}")

try:
    from monai.transforms import AddChanneld, EnsureChannelFirstd
    AddChannelTransform = AddChanneld
    EnsureChannelTransform = EnsureChannelFirstd
except ImportError:
    from monai.transforms import EnsureChannelFirstd
    AddChannelTransform = EnsureChannelFirstd
    EnsureChannelTransform = EnsureChannelFirstd
    EnsureChannelTransform = EnsureChannelFirstd
from monai.transforms import Compose, LoadImaged, ScaleIntensityd, ToTensord, RandCropByPosNegLabeld, RandFlipd, RandRotate90d, RandScaleIntensityd, Lambdad, Resized
from monai.losses import DiceLoss as MonaiDiceLoss
from monai.metrics import DiceMetric

# Global logger for architecture functions
logger = logging.getLogger(__name__)

# Utility functions moved to top to avoid NameError
def create_organized_model_directory(model_id=None, model_family="UNet-Coronary", version="1.0.0"):
    """Create an organized directory structure for model storage"""
    
    # Generate unique identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = f"{model_family.replace(' ', '_').lower()}_{timestamp}_{str(uuid.uuid4())[:8]}"
    
    # Create date-based organization
    date_str = datetime.now().strftime("%Y/%m")
    family_str = model_family.replace(" ", "_").lower()
    
    model_dir = os.path.join(
        "data",
        "models",
        "organized", 
        date_str,
        family_str,
        f"{unique_id}_v{version}"
    )
    
    # Don't create directories immediately - they will be created when needed
    # This prevents empty directories from failed/cancelled trainings
    
    return model_dir, unique_id

def ensure_model_directories(model_dir):
    """Create model subdirectories when actually needed"""
    subdirs = ["weights", "config", "artifacts", "predictions", "metrics", "logs"]
    for subdir in subdirs:
        os.makedirs(os.path.join(model_dir, subdir), exist_ok=True)

def save_enhanced_training_curves(epoch_history, model_dir, epoch):
    """Save comprehensive training curves with multiple metrics"""
    try:
        if len(epoch_history) < 2:
            return None
        
        # Ensure model directories exist when we actually need them
        ensure_model_directories(model_dir)
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Progress - Epoch {epoch+1}', fontsize=16)
        
        epochs = list(range(1, len(epoch_history) + 1))
        
        # Extract metrics from history
        train_losses = [h['train_loss'] for h in epoch_history]
        val_losses = [h['val_loss'] for h in epoch_history]
        train_dices = [h['train_dice'] for h in epoch_history]
        val_dices = [h['val_dice'] for h in epoch_history]
        
        # Loss curves
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Dice score curves
        axes[0, 1].plot(epochs, train_dices, 'b-', label='Training Dice', linewidth=2)
        axes[0, 1].plot(epochs, val_dices, 'r-', label='Validation Dice', linewidth=2)
        axes[0, 1].set_title('Dice Score Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss difference (overfitting indicator)
        loss_diff = [v - t for v, t in zip(val_losses, train_losses)]
        axes[0, 2].plot(epochs, loss_diff, 'g-', linewidth=2)
        axes[0, 2].set_title('Validation - Training Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss Difference')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Dice difference
        dice_diff = [v - t for v, t in zip(val_dices, train_dices)]
        axes[1, 0].plot(epochs, dice_diff, 'purple', linewidth=2)
        axes[1, 0].set_title('Validation - Training Dice')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice Difference')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Best metrics summary
        best_val_dice = max(val_dices)
        best_epoch = val_dices.index(best_val_dice) + 1
        
        summary_text = f"""
        Current Epoch: {epoch + 1}
        Best Val Dice: {best_val_dice:.4f} (Epoch {best_epoch})
        Current Val Dice: {val_dices[-1]:.4f}
        Current Train Dice: {train_dices[-1]:.4f}
        
        Current Val Loss: {val_losses[-1]:.4f}
        Current Train Loss: {train_losses[-1]:.4f}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
        
        # Learning progress (smoothed)
        if len(val_dices) > 5:
            # Simple moving average for trend
            window = min(5, len(val_dices))
            val_dice_smooth = []
            for i in range(len(val_dices)):
                start = max(0, i - window + 1)
                val_dice_smooth.append(np.mean(val_dices[start:i+1]))
            
            axes[1, 2].plot(epochs, val_dices, 'r--', alpha=0.5, label='Raw Validation Dice')
            axes[1, 2].plot(epochs, val_dice_smooth, 'r-', linewidth=2, label='Smoothed Validation Dice')
            axes[1, 2].set_title('Learning Progress (Smoothed)')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Dice Score')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save in artifacts subdirectory
        artifacts_dir = os.path.join(model_dir, "artifacts")
        filename = os.path.join(artifacts_dir, f'enhanced_training_curves_epoch_{epoch+1}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    except Exception as e:
        print(f"Could not save enhanced training curves: {e}")
        return None

def get_default_model_config(model_type):
    """Get default configuration for a model type"""
    # Default configurations for common architectures
    default_configs = {
        'unet': {
            'spatial_dims': 2,
            'in_channels': 1,
            'out_channels': 1,
            'channels': (16, 32, 64, 128, 256),
            'strides': (2, 2, 2, 2),
            'num_res_units': 2,
        },
        'monai_unet': {
            'spatial_dims': 2,
            'in_channels': 1,
            'out_channels': 1,
            'channels': (16, 32, 64, 128, 256),
            'strides': (2, 2, 2, 2),
            'num_res_units': 2,
        }
    }
    
    # Check if architecture has default config
    try:
        from ml.utils.architecture_registry import registry as architecture_registry
        arch_info = architecture_registry.get_architecture(model_type)
        if arch_info and arch_info.default_config:
            return arch_info.default_config
    except:
        pass
    
    # Return hardcoded default if available
    return default_configs.get(model_type, {})

def validate_architecture(model_type):
    """Validate that the specified architecture is available"""
    logger.info(f"Validating architecture: {model_type}")
    
    # Check if architecture is registered
    arch_info = architecture_registry.get_architecture(model_type)
    if not arch_info:
        logger.error(f"Architecture '{model_type}' not found in registry")
        available = [key for key in architecture_registry.get_all_architectures().keys()]
        logger.error(f"Available architectures: {available}")
        raise ValueError(f"Unknown architecture: {model_type}. Available: {available}")
    
    # Validate the architecture
    is_valid, message = architecture_registry.validate_architecture(model_type)
    if not is_valid:
        logger.error(f"Architecture validation failed: {message}")
        raise ValueError(f"Invalid architecture '{model_type}': {message}")
    
    logger.info(f"Architecture '{model_type}' validated successfully")
    return arch_info

# --- Patch: Add detailed logging around model architecture loading ---
def create_model_from_registry(model_type, device, **model_kwargs):
    """Create a model instance using the architecture registry"""
    logger.info(f"[ARCH] Attempting to load model_type: {model_type} with kwargs: {model_kwargs}")
    logger.info(f"[ARCH] Available architectures in registry: {list(architecture_registry._architectures.keys())}")

    try:
        # Handle special case for MONAI UNet (legacy compatibility) first
        if model_type in ['unet', 'monai_unet'] or 'unet' in model_type.lower():
            logger.info("[ARCH] Using MONAI UNet with default configuration (special case)")
            model = MonaiUNet(
                spatial_dims=model_kwargs.get('spatial_dims', 2),
                in_channels=model_kwargs.get('in_channels', 1),
                out_channels=model_kwargs.get('out_channels', 1),
                channels=model_kwargs.get('channels', (16, 32, 64, 128, 256)),
                strides=model_kwargs.get('strides', (2, 2, 2, 2)),
                num_res_units=model_kwargs.get('num_res_units', 2),
            )
            # Create a fake arch_info for unet
            from types import SimpleNamespace
            arch_info = SimpleNamespace(
                display_name="MONAI UNet",
                framework="PyTorch",
                key="monai_unet",
                version="1.0.0",
                description="MONAI U-Net for medical image segmentation"
            )
            logger.info(f"[ARCH] Successfully loaded model: {arch_info.display_name}")
        else:
            # Validate architecture and use from registry
            logger.info(f"[ARCH] Validating architecture: {model_type}")
            arch_info = validate_architecture(model_type)
            model_class = arch_info.model_class
            
            # Apply default config if available
            if arch_info.default_config:
                logger.info(f"[ARCH] Applying default config for {model_type}: {arch_info.default_config}")
                # Make a copy to avoid modifying the original default_config in the registry
                current_model_kwargs = arch_info.default_config.copy()
                current_model_kwargs.update(model_kwargs) # User-provided kwargs override defaults
            else:
                current_model_kwargs = model_kwargs
            
            # Create model instance
            logger.info(f"[ARCH] Instantiating {arch_info.display_name} with effective kwargs: {current_model_kwargs}")
            model = model_class(**current_model_kwargs)
            logger.info(f"[ARCH] Successfully loaded model: {arch_info.display_name}")
        
        # Move to device
        model = model.to(device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created successfully:")
        logger.info(f"  Architecture: {arch_info.display_name}")
        logger.info(f"  Framework: {arch_info.framework}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Device: {device}")
        
        return model, arch_info
        
    except Exception as e:
        logger.error(f"Failed to create model '{model_type}': {e}")
        raise ValueError(f"Model creation failed for '{model_type}': {e}")

def save_enhanced_model_metadata(model_dir, model_id, unique_id, args, model_info, training_metrics, model_family="UNet-Coronary", arch_info=None):
    """Save comprehensive model metadata"""
    # Ensure model directories exist when we actually need them
    ensure_model_directories(model_dir)
    
    # Use architecture info from registry if provided
    if arch_info is not None:
        architecture_name = arch_info.display_name
        framework = arch_info.framework
        architecture_key = arch_info.key
        architecture_version = arch_info.version
        architecture_description = arch_info.description
    else:
        architecture_name = "MonaiUNet"
        framework = "PyTorch"
        architecture_key = "monai_unet"
        architecture_version = "1.0.0"
        architecture_description = "MONAI U-Net"

    metadata = {
        "model_info": {
            "model_id": model_id,
            "unique_identifier": unique_id,
            "model_family": model_family,
            "version": architecture_version,
            "architecture": architecture_name,
            "architecture_key": architecture_key,
            "created_at": datetime.now().isoformat(),
            "framework": framework,
            "architecture_description": architecture_description
        },
        "architecture_details": model_info,
        "training_config": {
            "parameters": vars(args),
            "hyperparameters": {
                "optimizer": "Adam",
                "loss_function": "DiceLoss",
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "validation_split": args.validation_split
            }
        },
        "performance_metrics": training_metrics,
        "data_info": {
            "input_shape": "[1, 128, 128]",
            "output_shape": "[1, 128, 128]",
            "data_format": "DICOM/PNG",
            "normalization": "Scale Intensity [0, 1]"
        },
        "artifacts": {
            "weights_file": "weights/model.pth",
            "config_file": "config/model_config.json", 
            "training_curves": "artifacts/training_curves.png",
            "sample_predictions": "predictions/",
            "training_log": "logs/training.log"
        }
    }
    metadata_path = os.path.join(model_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4, default=str)
    return metadata_path

def save_model_comparison_artifacts(model_dir, model_metrics, epoch):
    """Save artifacts for model comparison and analysis"""
    try:
        # Ensure model directories exist when we actually need them
        ensure_model_directories(model_dir)
        
        artifacts_dir = os.path.join(model_dir, "artifacts")
        
        # Save detailed metrics as JSON
        metrics_file = os.path.join(artifacts_dir, f"detailed_metrics_epoch_{epoch+1}.json")
        with open(metrics_file, 'w') as f:
            json.dump(model_metrics, f, indent=4, default=str)
        
        # Create performance radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Define metrics for radar chart
        metrics_names = ['Dice Score', 'Loss (inverted)', 'Stability', 'Convergence']
        
        # Calculate normalized metrics (0-1 scale)
        dice_score = model_metrics.get('val_dice', 0)
        loss_inverted = 1 - min(model_metrics.get('val_loss', 1), 1) # Invert and cap loss
        
        # Calculate stability (lower variance in recent epochs is better)
        epoch_history = model_metrics.get('epoch_history', [])
        if len(epoch_history) > 5:
            recent_dices = [h.get('val_dice', 0) for h in epoch_history[-5:]]
            stability = 1 - (np.std(recent_dices) / np.mean(recent_dices)) if np.mean(recent_dices) > 0 else 0
        else:
            stability = 0.5
        
        # Calculate convergence (improvement trend)
        if len(epoch_history) > 3:
            early_dice = np.mean([h.get('val_dice', 0) for h in epoch_history[:3]])
            recent_dice = np.mean([h.get('val_dice', 0) for h in epoch_history[-3:]])
            convergence = min(recent_dice / early_dice if early_dice > 0 else 1, 1)
        else:
            convergence = 0.5
        
        values = [dice_score, loss_inverted, stability, convergence]
        
        # Complete the circle
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1)
        ax.set_title(f'Model Performance Radar - Epoch {epoch+1}', size=14, weight='bold')
        
        plt.tight_layout()
        radar_file = os.path.join(artifacts_dir, f"performance_radar_epoch_{epoch+1}.png")
        plt.savefig(radar_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return [metrics_file, radar_file]
    except Exception as e:
        print(f"Could not save model comparison artifacts: {e}")
        return []

def ensure_single_channel(x):
    """Ensure tensor has single channel - convert RGB to grayscale if needed"""
    if len(x.shape) == 3 and x.shape[0] == 3:  # RGB image
        # For masks/labels, check if all channels are the same (common for binary masks)
        if torch.allclose(x[0], x[1]) and torch.allclose(x[1], x[2]):
            # All channels are the same, just take the first one
            return x[0].unsqueeze(0)
        else:
            # Convert RGB to grayscale using standard weights
            return (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).unsqueeze(0)
    elif len(x.shape) == 3 and x.shape[0] == 1:  # Already single channel
        return x
    elif len(x.shape) == 2:  # No channel dimension
        return x.unsqueeze(0)
    else:
        return x

def get_monai_transforms(params, for_training=True):
    """Get MONAI transforms with configurable augmentations"""
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelTransform(keys=["image", "label"]),  # Ensure channel dimension
        # Convert images to single channel (grayscale) if they are RGB
        Lambdad(keys=["image"], func=ensure_single_channel),
        # Convert labels/masks to single channel (grayscale) if they are RGB
        Lambdad(keys=["label"], func=ensure_single_channel),
        ScaleIntensityd(keys=["image"]),
        # Normalize labels to 0-1 range for binary masks
        ScaleIntensityd(keys=["label"], minv=0.0, maxv=1.0),
    ]
    
    if for_training:
        # Add training-specific augmentations
        if params.get('use_random_scale', True):
            # Use RandScaleIntensityd for random intensity scaling
            transforms.append(RandScaleIntensityd(keys=["image"], factors=(0.8, 1.2), prob=0.5))
        
        crop_size = params.get('crop_size', 128)
        # Use RandCropByPosNegLabeld for better segmentation cropping
        # Set num_samples=1 to maintain batch size consistency
        transforms.append(RandCropByPosNegLabeld(
            keys=["image", "label"], 
            label_key="label",
            spatial_size=[crop_size, crop_size],
            pos=1,
            neg=1,
            num_samples=1,  # Changed from 4 to 1 to maintain batch size
            image_key="image",
            image_threshold=0
        ))
        
        if params.get('use_random_flip', True):
            transforms.append(RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0))
        
        if params.get('use_random_rotate', True):
            transforms.append(RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3))
        
        if params.get('use_random_intensity', True):
            # Use RandScaleIntensityd for random intensity variations
            transforms.append(RandScaleIntensityd(
                keys=["image"],
                factors=(0.9, 1.1),
                prob=0.7
            ))
    else:
        # For validation/visualization - resize to standard size but keep full images
        transforms.append(Resized(keys=["image", "label"], spatial_size=[256, 256], mode=["bilinear", "nearest"]))
    
    transforms.append(ToTensord(keys=["image", "label"]))
    return Compose(transforms)

def save_model_summary(model, model_dir=None):
    if model_dir:
        # Ensure model directories exist when we actually need them
        ensure_model_directories(model_dir)
        
    summary = []
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        summary.append(f"{name}: {list(param.shape)}, params: {param_count}")
    summary_text = "\n".join([
        "Model Summary:",
        "=" * 50,
        "\n".join(summary),
        "=" * 50,
        f"Total parameters: {total_params:,}",
        f"Trainable parameters: {trainable_params:,}",
    ])
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        out_path = os.path.join(model_dir, "model_summary.txt")
    else:
        out_path = "model_summary.txt"
    with open(out_path, "w") as f:
        f.write(summary_text)
    return out_path

def save_training_curves(epoch, metrics, logger, model_dir=None):
    """Save training curves plot"""
    try:
        if model_dir:
            # Ensure model directories exist when we actually need them
            ensure_model_directories(model_dir)
            
        # This is a simple implementation - in practice you'd want to track metrics over time
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # For now, just show current epoch metrics
        ax1.bar(['Train', 'Val'], [metrics['train_loss'], metrics['val_loss']])
        ax1.set_title('Loss')
        ax1.set_ylabel('Loss')
        
        ax2.bar(['Train', 'Val'], [metrics['train_dice'], metrics['val_dice']])
        ax2.set_title('Dice Score')
        ax2.set_ylabel('Dice')
        
        ax3.text(0.5, 0.5, f"Epoch: {epoch + 1}\nTrain Loss: {metrics['train_loss']:.4f}\nVal Loss: {metrics['val_loss']:.4f}", 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Current Metrics')
        ax3.set_xlabel('Epoch')
        ax3.axis('off')
        
        ax4.text(0.5, 0.5, f"Train Dice: {metrics['train_dice']:.4f}\nVal Dice: {metrics['val_dice']:.4f}", 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Performance')
        ax4.axis('off')
        
        plt.tight_layout()
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            filename = os.path.join(model_dir, f'training_curves_epoch_{epoch+1}.png')
        else:
            filename = f'training_curves_epoch_{epoch+1}.png'
        plt.savefig(filename)
        plt.close()
        return filename
    except Exception as e:
        logger.warning(f"Could not save training curves: {e}")
        return None

def save_sample_predictions(model, val_loader, device, epoch, model_dir=None):
    """Save sample predictions from validation set with enhanced error handling
    
    Enhanced error handling includes:
    - Safe batch format detection and validation
    - Multiple fallback paths for different batch types
    - Comprehensive error logging with traceback
    - Fallback error image creation when prediction generation fails
    - File size validation to ensure valid PNG output
    """
    logger.info(f"[PREDICTIONS] Starting to save sample predictions for epoch {epoch+1}")
    
    if model_dir:
        # Ensure model directories exist
        ensure_model_directories(model_dir)
        
    model.eval()
    
    # Ensure we have a clean matplotlib state
    plt.close('all')
    
    with torch.no_grad():
        try:
            # Safe batch retrieval
            val_batch = next(iter(val_loader))
            images, labels = None, None
            
            if isinstance(val_batch, dict):
                if "image" in val_batch and "label" in val_batch:
                    images = val_batch["image"].to(device)
                    labels = val_batch["label"].to(device)
                else:
                    logger.error(f"[PREDICTIONS] Dict batch missing required keys. Available keys: {list(val_batch.keys())}")
                    raise ValueError("Batch dict missing 'image' or 'label' keys")
            elif isinstance(val_batch, (list, tuple)) and len(val_batch) == 2:
                images, labels = val_batch[0].to(device), val_batch[1].to(device)
            else:
                logger.error(f"[PREDICTIONS] Unexpected batch format: {type(val_batch)}")
                raise ValueError(f"Unsupported batch format: {type(val_batch)}")
            
            if images is None or labels is None:
                raise ValueError("Failed to extract images and labels from batch")
                
            logger.info(f"[PREDICTIONS] Processing batch with {images.shape[0]} samples, image shape: {images.shape}")
            
            # Model inference
            outputs = model(images)
            
            # Apply appropriate post-processing based on number of output channels
            num_output_channels = outputs.shape[1]
            if num_output_channels == 1:
                # Binary segmentation
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > 0.5).float()
                logger.info(f"[PREDICTIONS] Applied binary segmentation post-processing (sigmoid + threshold)")
                use_colormap = False
            else:
                # Multi-class semantic segmentation
                outputs = torch.softmax(outputs, dim=1)
                outputs = torch.argmax(outputs, dim=1, keepdim=True).float()
                logger.info(f"[PREDICTIONS] Applied multi-class segmentation post-processing (softmax + argmax) for {num_output_channels} classes")
                use_colormap = True
            
            # Create figure with proper settings
            fig, axes = plt.subplots(3, 4, figsize=(15, 10))
            plt.suptitle(f'Sample Predictions - Epoch {epoch+1} (Full Images)', fontsize=16)
            
            # Create custom colormap for semantic segmentation
            if use_colormap:
                import matplotlib.colors as mcolors
                # Create a custom colormap with distinct colors for each class
                colors = [
                    '#000000',  # 0 - black (background)
                    '#FF0000',  # 1 - red
                    '#00FF00',  # 2 - green
                    '#0000FF',  # 3 - blue
                    '#FFFF00',  # 4 - yellow
                    '#FF00FF',  # 5 - magenta
                    '#00FFFF',  # 6 - cyan
                    '#FFA500',  # 7 - orange
                    '#800080',  # 8 - purple
                    '#FFC0CB',  # 9 - pink
                    '#ADFF2F',  # 10 - green yellow
                    '#1E90FF',  # 11 - dodger blue
                    '#FF1493',  # 12 - deep pink
                    '#00FA9A',  # 13 - medium spring green
                    '#FF4500',  # 14 - red orange
                    '#483D8B',  # 15 - dark slate blue
                    '#FFD700',  # 16 - gold
                    '#DC143C',  # 17 - crimson
                    '#7CFC00',  # 18 - lawn green
                    '#BA55D3',  # 19 - medium orchid
                    '#8A2BE2',  # 20 - blue violet
                    '#FF69B4',  # 21 - hot pink
                    '#FF8C00',  # 22 - dark orange
                    '#B8860B',  # 23 - dark goldenrod
                    '#4682B4',  # 24 - steel blue
                    '#00CED1',  # 25 - dark turquoise
                    '#FF6347'   # 26 - tomato red
                ]
                # Pad with additional colors if needed
                while len(colors) < num_output_channels:
                    colors.append('#FFFFFF')  # white for overflow
                
                cmap = mcolors.ListedColormap(colors[:num_output_channels])
            
            num_samples = min(4, images.shape[0])
            for i in range(num_samples):
                # Input image
                axes[0, i].imshow(images[i, 0].cpu().numpy(), cmap='gray')
                axes[0, i].set_title(f'Input #{i+1}')
                axes[0, i].axis('off')
                
                # Ground truth - use appropriate visualization
                if use_colormap:
                    # For multi-class segmentation, use custom color mapping
                    gt_data = labels[i, 0].cpu().numpy() if labels.shape[1] == 1 else labels[i].cpu().numpy()
                    # If gt_data is one-hot encoded, convert to class indices
                    if len(gt_data.shape) == 3:
                        gt_data = np.argmax(gt_data, axis=0)
                    elif len(gt_data.shape) == 2 and gt_data.shape[0] > 1:
                        gt_data = np.argmax(gt_data, axis=0)
                    
                    unique_classes = np.unique(gt_data)
                    logger.info(f"[PREDICTIONS] Ground truth classes for sample {i+1}: {unique_classes}")
                    axes[1, i].imshow(gt_data, cmap=cmap, vmin=0, vmax=num_output_channels-1)
                    axes[1, i].set_title(f'Ground Truth #{i+1} ({len(unique_classes)} classes)')
                else:
                    # For binary segmentation, use grayscale
                    axes[1, i].imshow(labels[i, 0].cpu().numpy(), cmap='gray')
                    axes[1, i].set_title(f'Ground Truth #{i+1}')
                axes[1, i].axis('off')
                
                # Prediction - use appropriate visualization
                if use_colormap:
                    # For multi-class segmentation, use custom color mapping
                    pred_data = outputs[i, 0].cpu().numpy()
                    unique_pred_classes = np.unique(pred_data)
                    logger.info(f"[PREDICTIONS] Predicted classes for sample {i+1}: {unique_pred_classes}")
                    axes[2, i].imshow(pred_data, cmap=cmap, vmin=0, vmax=num_output_channels-1)
                    axes[2, i].set_title(f'Prediction #{i+1} ({len(unique_pred_classes)} classes)')
                else:
                    # For binary segmentation, use grayscale
                    axes[2, i].imshow(outputs[i, 0].cpu().numpy(), cmap='gray')
                    axes[2, i].set_title(f'Prediction #{i+1}')
                axes[2, i].axis('off')
            
            # Hide unused subplots if we have fewer than 4 samples
            for i in range(num_samples, 4):
                axes[0, i].axis('off')
                axes[1, i].axis('off')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            
            # Determine output path - ensure predictions directory exists
            if model_dir and os.path.exists(model_dir):
                pred_dir = os.path.join(model_dir, f'predictions/epoch_{epoch+1:03d}')
                os.makedirs(pred_dir, exist_ok=True)
                filename = os.path.join(pred_dir, f'predictions_epoch_{epoch+1:03d}.png')
            else:
                # Fallback directory
                fallback_dir = os.path.join('data', 'models', 'artifacts', 'predictions')
                os.makedirs(fallback_dir, exist_ok=True)
                filename = os.path.join(fallback_dir, f'predictions_epoch_{epoch+1:03d}.png')
            
            # Save with high quality and ensure file is written
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            # Verify file was created and has content
            if os.path.exists(filename) and os.path.getsize(filename) > 1000:  # At least 1KB
                logger.info(f"[PREDICTIONS] Successfully saved prediction samples to: {filename}")
                return filename
            else:
                logger.error(f"[PREDICTIONS] Failed to create valid prediction file: {filename}")
                return None
                
        except Exception as e:
            logger.error(f"[PREDICTIONS] Error creating sample predictions: {e}")
            import traceback
            logger.error(f"[PREDICTIONS] Traceback: {traceback.format_exc()}")
            plt.close('all')
            
            # Try to create a simple fallback image to ensure we have something
            try:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                error_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
                ax.text(0.5, 0.5, f'Epoch {epoch+1}\nPrediction generation failed\nError: {error_msg}', 
                       ha='center', va='center', fontsize=12, transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                ax.set_title(f'Prediction Error - Epoch {epoch+1}', fontsize=14)
                
                # Determine fallback output path
                if model_dir and os.path.exists(model_dir):
                    pred_dir = os.path.join(model_dir, f'predictions/epoch_{epoch+1:03d}')
                    os.makedirs(pred_dir, exist_ok=True)
                    fallback_filename = os.path.join(pred_dir, f'predictions_epoch_{epoch+1:03d}_error.png')
                else:
                    fallback_dir = os.path.join('data', 'models', 'artifacts', 'predictions')
                    os.makedirs(fallback_dir, exist_ok=True)
                    fallback_filename = os.path.join(fallback_dir, f'predictions_epoch_{epoch+1:03d}_error.png')
                
                plt.savefig(fallback_filename, dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
                plt.close()
                
                # Verify fallback file was created
                if os.path.exists(fallback_filename) and os.path.getsize(fallback_filename) > 0:
                    logger.info(f"[PREDICTIONS] Created error fallback image: {fallback_filename}")
                    return fallback_filename
                else:
                    logger.error(f"[PREDICTIONS] Failed to create fallback image")
                    return None
                    
            except Exception as fallback_error:
                logger.error(f"[PREDICTIONS] Failed to create even fallback image: {fallback_error}")
                return None

def save_config(args, model_dir=None):
    """Save training configuration"""
    if model_dir:
        # Ensure model directories exist when we actually need them
        ensure_model_directories(model_dir)
        
    # Extract augmentation flags explicitly
    config = {
        "training_params": vars(args),
        "device": str(getattr(args, 'device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))),
        "pytorch_version": torch.__version__,
        "random_flip": getattr(args, 'random_flip', False),
        "random_rotate": getattr(args, 'random_rotate', False),
        "random_scale": getattr(args, 'random_scale', False),
        "random_intensity": getattr(args, 'random_intensity', False),
    }
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        out_path = os.path.join(model_dir, "training_config.json")
    else:
        out_path = "training_config.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=4)
    return out_path

def get_monai_datasets(data_path, val_split=0.2, transform_params=None):
    import glob
    # Use default params if none provided
    if transform_params is None:
        transform_params = {
            'use_random_flip': True,
            'use_random_rotate': True,
            'use_random_scale': True,
            'use_random_intensity': True,
            'crop_size': 128
        }
    
    # Support both images/masks and images/labels, prefer imgs/masks if present
    imgs_dir = os.path.join(data_path, "imgs")
    masks_dir = os.path.join(data_path, "masks")
    if os.path.exists(imgs_dir) and os.path.exists(masks_dir):
        image_files = sorted(glob.glob(f"{imgs_dir}/*"))
        label_files = sorted(glob.glob(f"{masks_dir}/*"))
    else:
        # Fallback to images/labels or nested data/data/images, data/data/labels
        images_dir = os.path.join(data_path, "images")
        labels_dir = os.path.join(data_path, "labels")
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            images_dir = os.path.join(data_path, "data", "images")
            labels_dir = os.path.join(data_path, "data", "labels")
        image_files = sorted(glob.glob(f"{images_dir}/*"))
        label_files = sorted(glob.glob(f"{labels_dir}/*"))
    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(image_files, label_files)]
    n_val = int(len(data_dicts) * val_split)
    train_files, val_files = data_dicts[n_val:], data_dicts[:n_val]
    
    # Enhanced dataset logging
    logger.info(f"[DATASET] Dataset directory: {data_path}")
    logger.info(f"[DATASET] Images directory: {images_dir}")
    logger.info(f"[DATASET] Labels directory: {labels_dir}")
    logger.info(f"[DATASET] Total files found: {len(image_files)} images, {len(label_files)} labels")
    logger.info(f"[DATASET] Validation split: {val_split} ({n_val} samples for validation)")
    logger.info(f"[DATASET] Training samples: {len(train_files)}, Validation samples: {len(val_files)}")
    
    # Log sample file paths for verification
    if train_files:
        logger.info(f"[DATASET] Sample training files:")
        for i, sample in enumerate(train_files[:3]):  # Show first 3 training samples
            logger.info(f"[DATASET]   Train #{i+1}: Image: {os.path.basename(sample['image'])}, Label: {os.path.basename(sample['label'])}")
    
    if val_files:
        logger.info(f"[DATASET] Sample validation files:")
        for i, sample in enumerate(val_files[:3]):  # Show first 3 validation samples
            logger.info(f"[DATASET]   Val #{i+1}: Image: {os.path.basename(sample['image'])}, Label: {os.path.basename(sample['label'])}")
    
    # Log file extension info
    if image_files:
        image_exts = set(os.path.splitext(f)[1].lower() for f in image_files)
        label_exts = set(os.path.splitext(f)[1].lower() for f in label_files)
        logger.info(f"[DATASET] Image file extensions: {sorted(image_exts)}")
        logger.info(f"[DATASET] Label file extensions: {sorted(label_exts)}")
    
    # Create separate transforms for training and validation
    train_transforms = get_monai_transforms(transform_params, for_training=True)
    val_transforms = get_monai_transforms(transform_params, for_training=False)
    
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
    return train_ds, val_ds

def detect_dataset_type(data_path):
    """
    Automatically detect dataset type (ARCADE or standard coronary)
    Returns: tuple (dataset_type, dataset_info)
    """
    logger.info(f"[DATASET] Detecting dataset type for: {data_path}")
    
    # Check for ARCADE dataset structure
    if ARCADE_AVAILABLE:
        # Look for ARCADE dataset structure
        arcade_path = os.path.join(data_path, "arcade_challenge_datasets")
        if os.path.exists(arcade_path):
            logger.info("[DATASET] ARCADE dataset detected")
            info = get_arcade_dataset_info(data_path)
            return "arcade", info
    
    # Check for standard coronary dataset structure
    imgs_path = os.path.join(data_path, "imgs")
    masks_path = os.path.join(data_path, "masks")
    
    if os.path.exists(imgs_path) and os.path.exists(masks_path):
        img_count = len([f for f in os.listdir(imgs_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        mask_count = len([f for f in os.listdir(masks_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        logger.info(f"[DATASET] Standard coronary dataset detected: {img_count} images, {mask_count} masks")
        
        info = {
            "dataset_type": "coronary_standard",
            "images": img_count,
            "masks": mask_count,
            "path": data_path
        }
        return "coronary_standard", info
    
    # Check for MONAI-style structure
    images_dir = os.path.join(data_path, "images")
    labels_dir = os.path.join(data_path, "labels")
    
    if os.path.exists(images_dir) and os.path.exists(labels_dir):
        img_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.nii', '.nii.gz'))])
        label_count = len([f for f in os.listdir(labels_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.nii', '.nii.gz'))])
        
        logger.info(f"[DATASET] MONAI-style dataset detected: {img_count} images, {label_count} labels")
        
        info = {
            "dataset_type": "monai_style",
            "images": img_count,
            "labels": label_count,
            "path": data_path
        }
        return "monai_style", info
    
    logger.warning(f"[DATASET] Unknown dataset structure in: {data_path}")
    return "unknown", {"path": data_path}

def get_datasets_with_auto_detection(data_path, validation_split, transform_params, args):
    """
    Get datasets with automatic detection of dataset type
    Supports both ARCADE and standard coronary datasets
    """
    # Check if dataset type is explicitly specified
    dataset_type_override = getattr(args, 'dataset_type', 'auto')
    
    if dataset_type_override != 'auto':
        logger.info(f"[DATASET] Using specified dataset type: {dataset_type_override}")
        
        if dataset_type_override.startswith('arcade_') and ARCADE_AVAILABLE:
            logger.info("[DATASET] Using ARCADE dataset loader (forced)")
            return get_arcade_datasets(data_path, validation_split, transform_params, args, forced_type=dataset_type_override)
        elif dataset_type_override == 'coronary':
            logger.info("[DATASET] Using MONAI dataset loader (forced)")
            return get_monai_datasets(data_path, validation_split, transform_params)
        else:
            logger.warning(f"[DATASET] Unknown dataset type '{dataset_type_override}', falling back to auto-detection")
    
    # Auto-detection
    dataset_type, dataset_info = detect_dataset_type(data_path)
    
    if dataset_type == "arcade" and ARCADE_AVAILABLE:
        logger.info("[DATASET] Using ARCADE dataset loader")
        return get_arcade_datasets(data_path, validation_split, transform_params, args)
    elif dataset_type in ["coronary_standard", "monai_style"]:
        logger.info("[DATASET] Using MONAI dataset loader")
        return get_monai_datasets(data_path, validation_split, transform_params)
    else:
        logger.error(f"[DATASET] Unsupported dataset type: {dataset_type}")
        raise ValueError(f"Unsupported dataset structure in {data_path}")

def get_arcade_datasets(data_path, validation_split, transform_params, args, forced_type=None):
    """
    Create ARCADE datasets for training
    """
    # Ensure ARCADE support
    if not ARCADE_AVAILABLE:
        raise ImportError("ARCADE dataset support not available. Install pycocotools: pip install pycocotools")
    
    # Log detailed information about dataset path and args
    logger.info(f"[ARCADE] Creating datasets with:")
    logger.info(f"[ARCADE]   data_path: {data_path}")
    logger.info(f"[ARCADE]   forced_type: {forced_type}")
    logger.info(f"[ARCADE]   args.dataset_type: {getattr(args, 'dataset_type', 'NOT_SET')}")
    logger.info(f"[ARCADE]   Directory exists: {os.path.exists(data_path)}")
    if os.path.exists(data_path):
        logger.info(f"[ARCADE]   Directory contents: {os.listdir(data_path)}")
        arcade_path = os.path.join(data_path, "arcade_challenge_datasets")
        logger.info(f"[ARCADE]   ARCADE path exists: {os.path.exists(arcade_path)}")
        if os.path.exists(arcade_path):
            logger.info(f"[ARCADE]   ARCADE contents: {os.listdir(arcade_path)}")
    
    # Determine task type
    if forced_type:
        mapping = {
            'arcade_binary': 'binary_segmentation',
            'arcade_binary_segmentation': 'binary_segmentation',
            'arcade_semantic': 'semantic_segmentation',
            'arcade_semantic_segmentation': 'semantic_segmentation',
            'arcade_stenosis': 'stenosis_detection',
            'arcade_stenosis_detection': 'stenosis_detection',
            'arcade_classification': 'artery_classification',
            'arcade_artery_classification': 'artery_classification'
        }
        task = mapping.get(forced_type, 'binary_segmentation')
    else:
        mt = getattr(args, 'model_type', '').lower()
        if 'semantic' in mt:
            task = 'semantic_segmentation'
        elif 'stenosis' in mt:
            task = 'stenosis_detection'
        elif 'artery' in mt or 'classification' in mt:
            task = 'artery_classification'
        else:
            task = 'binary_segmentation'
    logger.info(f"[ARCADE] Using task: {task}")
    
    # Common transforms - use crop_size from transform_params to match training expectations
    res = getattr(args, 'resolution', '512')
    crop_size = transform_params.get('crop_size', 128)
    
    # For ARCADE datasets, use the crop_size as the target resolution to match MONAI training expectations
    # This ensures spatial consistency between training (where model sees crop_size x crop_size) and validation
    size = crop_size  # Use crop_size instead of resolution for spatial consistency
    
    logger.info(f"[ARCADE] Using resolution {size}x{size} to match training crop size")
    
    if task in ['binary_segmentation','semantic_segmentation']:
        img_tr = tv_transforms.Compose([tv_transforms.Resize((size,size)), tv_transforms.ToTensor(), tv_transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        if task=='binary_segmentation':
            mask_tr = tv_transforms.Compose([tv_transforms.Resize((size,size)), tv_transforms.ToTensor()])
        else:
            # For semantic segmentation, add proper resizing to ensure spatial consistency
            def resize_semantic_mask(x):
                """Resize semantic mask tensor to match image dimensions"""
                # x is numpy array of shape (H, W, C)
                import torch.nn.functional as F
                import torch
                
                # Convert to tensor and permute to (C, H, W)
                tensor = torch.from_numpy(x).permute(2, 0, 1).float()
                
                # Resize to target size using nearest neighbor to preserve class labels
                resized = F.interpolate(tensor.unsqueeze(0), size=(size, size), mode='nearest')
                
                # Remove batch dimension and return
                return resized.squeeze(0)
            
            mask_tr = tv_transforms.Compose([tv_transforms.Lambda(resize_semantic_mask)])
    elif task == 'artery_classification':
        # For artery classification: input is binary mask, output is 0/1 label
        mask_tr = tv_transforms.Compose([tv_transforms.Resize((size,size)), tv_transforms.ToTensor()])
        img_tr = None  # No image transforms needed for mask input
    else:
        img_tr = tv_transforms.Compose([tv_transforms.Resize((size,size)), tv_transforms.ToTensor(), tv_transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        mask_tr = None
    
    # Log what we're about to create
    logger.info(f"[ARCADE] Creating {task} datasets...")
    
    # Instantiate datasets
    try:
        if task=='binary_segmentation':
            logger.info("[ARCADE] Instantiating ARCADEBinarySegmentation for train...")
            train_ds = ARCADEBinarySegmentation(
                root=data_path,
                image_set='train',
                side=getattr(args,'artery_side',None),
                download=False,
                transform=img_tr,
                target_transform=mask_tr
            )
            logger.info(f"[ARCADE] Train dataset created, length: {len(train_ds)}")
            
            # Enhanced ARCADE dataset logging
            logger.info(f"[ARCADE] BINARY SEGMENTATION Dataset Information:")
            logger.info(f"[ARCADE]   Root path: {data_path}")
            logger.info(f"[ARCADE]   Train samples: {len(train_ds)}")
            
            # Log sample paths and file info
            if len(train_ds) > 0:
                try:
                    sample = train_ds[0]
                    logger.info(f"[ARCADE]   Sample train data shape: {sample[0].shape if hasattr(sample[0], 'shape') else 'N/A'}")
                    logger.info(f"[ARCADE]   Sample train mask shape: {sample[1].shape if hasattr(sample[1], 'shape') else 'N/A'}")
                except Exception as e:
                    logger.warning(f"[ARCADE]   Could not get sample info: {e}")
            
            logger.info("[ARCADE] Instantiating ARCADEBinarySegmentation for val...")
            val_ds = ARCADEBinarySegmentation(
                root=data_path,
                image_set='val',
                side=getattr(args,'artery_side',None),
                download=False,
                transform=img_tr,
                target_transform=mask_tr
            )
            logger.info(f"[ARCADE] Val dataset created, length: {len(val_ds)}")
            logger.info(f"[ARCADE]   Val samples: {len(val_ds)}")
            
            # Log sample validation paths and file info
            if len(val_ds) > 0:
                try:
                    sample = val_ds[0]
                    logger.info(f"[ARCADE]   Sample val data shape: {sample[0].shape if hasattr(sample[0], 'shape') else 'N/A'}")
                    logger.info(f"[ARCADE]   Sample val mask shape: {sample[1].shape if hasattr(sample[1], 'shape') else 'N/A'}")
                except Exception as e:
                    logger.warning(f"[ARCADE]   Could not get val sample info: {e}")
            
            # Log total dataset info
            total_samples = len(train_ds) + len(val_ds)
            logger.info(f"[ARCADE] Total dataset size: {total_samples} ({len(train_ds)} train + {len(val_ds)} val)")
            logger.info(f"[ARCADE] Image resolution: {size}x{size}")
            logger.info(f"[ARCADE] Transforms applied: Resize, ToTensor, Normalize")
            
        elif task=='semantic_segmentation':
            logger.info("[ARCADE] Instantiating ARCADESemanticSegmentation for train...")
            train_ds = ARCADESemanticSegmentation(
                root=data_path,
                image_set='train',
                side=getattr(args,'artery_side',None),
                download=False,
                transform=img_tr,
                target_transform=mask_tr
            )
            logger.info(f"[ARCADE] SEMANTIC SEGMENTATION Train dataset created, length: {len(train_ds)}")
            
            logger.info("[ARCADE] Instantiating ARCADESemanticSegmentation for val...")
            val_ds = ARCADESemanticSegmentation(
                root=data_path,
                image_set='val',
                side=getattr(args,'artery_side',None),
                download=False,
                transform=img_tr,
                target_transform=mask_tr
            )
            logger.info(f"[ARCADE] SEMANTIC SEGMENTATION Val dataset created, length: {len(val_ds)}")
            
            # Enhanced logging for semantic segmentation
            logger.info(f"[ARCADE] SEMANTIC SEGMENTATION Dataset Information:")
            logger.info(f"[ARCADE]   Root path: {data_path}")
            logger.info(f"[ARCADE]   Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
            logger.info(f"[ARCADE]   Total samples: {len(train_ds) + len(val_ds)}")
            logger.info(f"[ARCADE]   Image resolution: {size}x{size}")
            logger.info(f"[ARCADE]   Task type: Semantic Segmentation (multi-class)")
            
            # Log sample info
            if len(train_ds) > 0:
                try:
                    sample = train_ds[0]
                    logger.info(f"[ARCADE]   Sample train data shape: {sample[0].shape if hasattr(sample[0], 'shape') else 'N/A'}")
                    logger.info(f"[ARCADE]   Sample train mask shape: {sample[1].shape if hasattr(sample[1], 'shape') else 'N/A'}")
                except Exception as e:
                    logger.warning(f"[ARCADE]   Could not get semantic sample info: {e}")
                    
        elif task == 'artery_classification':
            logger.info("[ARCADE] Instantiating ARCADEArteryClassification for train...")
            train_ds = ARCADEArteryClassification(
                root=data_path,
                image_set='train',
                side=getattr(args,'artery_side',None),
                download=False,
                transform=mask_tr  # Binary mask transform
            )
            logger.info(f"[ARCADE] ARTERY CLASSIFICATION Train dataset created, length: {len(train_ds)}")
            
            logger.info("[ARCADE] Instantiating ARCADEArteryClassification for val...")
            val_ds = ARCADEArteryClassification(
                root=data_path,
                image_set='val',
                side=getattr(args,'artery_side',None),
                download=False,
                transform=mask_tr  # Binary mask transform
            )
            logger.info(f"[ARCADE] ARTERY CLASSIFICATION Val dataset created, length: {len(val_ds)}")
            
            # Enhanced logging for artery classification
            logger.info(f"[ARCADE] ARTERY CLASSIFICATION Dataset Information:")
            logger.info(f"[ARCADE]   Root path: {data_path}")
            logger.info(f"[ARCADE]   Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
            logger.info(f"[ARCADE]   Total samples: {len(train_ds) + len(val_ds)}")
            logger.info(f"[ARCADE]   Image resolution: {size}x{size}")
            logger.info(f"[ARCADE]   Task type: Artery Classification (binary mask → left/right)")
            logger.info(f"[ARCADE]   Input: Binary mask (0/255)")
            logger.info(f"[ARCADE]   Output: 0=right artery, 1=left artery")
            
            # Log sample info
            if len(train_ds) > 0:
                try:
                    sample = train_ds[0]
                    logger.info(f"[ARCADE]   Sample train mask shape: {sample[0].shape if hasattr(sample[0], 'shape') else 'N/A'}")
                    logger.info(f"[ARCADE]   Sample train label: {sample[1]} ({'right' if sample[1] == 0 else 'left'})")
                except Exception as e:
                    logger.warning(f"[ARCADE]   Could not get artery classification sample info: {e}")
                    
        else:  # stenosis_detection
            logger.info("[ARCADE] Instantiating ARCADEStenosisDetection for train...")
            train_ds = ARCADEStenosisDetection(
                root=data_path,
                image_set='train',
                side=getattr(args,'artery_side',None),
                download=False,
                transform=img_tr
            )
            logger.info(f"[ARCADE] STENOSIS DETECTION Train dataset created, length: {len(train_ds)}")
            
            logger.info("[ARCADE] Instantiating ARCADEStenosisDetection for val...")
            val_ds = ARCADEStenosisDetection(
                root=data_path,
                image_set='val',
                side=getattr(args,'artery_side',None),
                download=False,
                transform=img_tr
            )
            logger.info(f"[ARCADE] STENOSIS DETECTION Val dataset created, length: {len(val_ds)}")
            
            # Enhanced logging for stenosis detection
            logger.info(f"[ARCADE] STENOSIS DETECTION Dataset Information:")
            logger.info(f"[ARCADE]   Root path: {data_path}")
            logger.info(f"[ARCADE]   Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
            logger.info(f"[ARCADE]   Total samples: {len(train_ds) + len(val_ds)}")
            logger.info(f"[ARCADE]   Image resolution: {size}x{size}")
            logger.info(f"[ARCADE]   Task type: Stenosis Detection (classification)")
            
            # Log sample info for stenosis detection
            if len(train_ds) > 0:
                try:
                    sample = train_ds[0]
                    logger.info(f"[ARCADE]   Sample train data shape: {sample[0].shape if hasattr(sample[0], 'shape') else 'N/A'}")
                    logger.info(f"[ARCADE]   Sample train label type: {type(sample[1])}")
                except Exception as e:
                    logger.warning(f"[ARCADE]   Could not get stenosis sample info: {e}")
    except Exception as e:
        logger.error(f"[ARCADE] Failed to create dataset: {e}")
        logger.error(f"[ARCADE] Exception type: {type(e)}")
        import traceback
        logger.error(f"[ARCADE] Traceback: {traceback.format_exc()}")
        raise
    
    # Wrap in DataLoaders
    train_loader = TorchDataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=getattr(args,'num_workers',2))
    val_loader   = TorchDataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=getattr(args,'num_workers',2))
    logger.info(f"[ARCADE] Loaders: {task} train={len(train_ds)}, val={len(val_ds)}")
    return train_loader, val_loader

def create_optimizer(model, args):
    """Create optimizer based on args.optimizer choice"""
    optimizer_name = getattr(args, 'optimizer', 'adam').lower()
    learning_rate = args.learning_rate
    
    logger.info(f"[OPTIMIZER] Creating {optimizer_name.upper()} optimizer with lr={learning_rate}")
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        # SGD with momentum for better convergence
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-8)
    elif optimizer_name == 'adamw':
        # AdamW with weight decay for better regularization
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    else:
        logger.warning(f"[OPTIMIZER] Unknown optimizer '{optimizer_name}', falling back to Adam")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    logger.info(f"[OPTIMIZER] Created {type(optimizer).__name__} with parameters: {optimizer.defaults}")
    return optimizer

def parse_args():
    parser = argparse.ArgumentParser(description='Train or run inference with MONAI U-Net model for coronary segmentation')
    parser.add_argument('--save-training-template', action='store_true', help='Save a training config template and exit')
    parser.add_argument('--mode', choices=['train', 'predict'], required=False, help="Mode to run in: train or predict")
    # Model parameters
    parser.add_argument('--model-family', type=str, default='UNet-Coronary', help='Model family name for registry and organization')
    parser.add_argument('--model-type', type=str, default='unet', help='Model type/architecture')
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', 
                       choices=['adam', 'sgd', 'rmsprop', 'adamw'], 
                       help='Optimizer to use for training')
    parser.add_argument('--data-path', type=str, help='Path to dataset')
    parser.add_argument('--dataset-type', type=str, default='auto', 
                       choices=['auto', 'coronary', 'arcade_binary', 'arcade_semantic', 'arcade_stenosis', 'arcade_classification'],
                       help='Type of dataset to use')
    parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--mlflow-run-id', type=str, help='MLflow run ID')
    parser.add_argument('--model-id', type=int, help='Database model ID for callback')
    # Augmentation parameters
    parser.add_argument('--random-flip', action='store_true', help='Enable random flip augmentation')
    parser.add_argument('--random-rotate', action='store_true', help='Enable random rotation augmentation')
    parser.add_argument('--random-scale', action='store_true', help='Enable random scaling augmentation')
    parser.add_argument('--random-intensity', action='store_true', help='Enable random intensity scaling')
    parser.add_argument('--crop-size', type=int, default=128, help='Size of random crop')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loading workers (reduced for Docker)')
    
    # Learning rate scheduler parameters
    parser.add_argument('--lr-scheduler', type=str, default='none', 
                       choices=['none', 'plateau', 'step', 'exponential', 'cosine', 'adaptive'],
                       help='Learning rate scheduler type')
    parser.add_argument('--lr-patience', type=int, default=5, help='Patience for plateau scheduler')
    parser.add_argument('--lr-factor', type=float, default=0.5, help='Factor to reduce learning rate')
    parser.add_argument('--lr-step-size', type=int, default=10, help='Step size for step scheduler')
    parser.add_argument('--lr-gamma', type=float, default=0.1, help='Gamma for step/exponential scheduler')
    parser.add_argument('--min-lr', type=float, default=1e-7, help='Minimum learning rate threshold')
    
    # Prediction parameters
    parser.add_argument('--model-path', type=str, help='Path to trained model weights')
    parser.add_argument('--input-path', type=str, help='Path to input image or directory')
    parser.add_argument('--output-dir', type=str, help='Directory to save predictions')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on')
    parser.add_argument('--weights-path', type=str, help='Optional path to a .pth file for inference')
    parser.add_argument('--resolution', type=str, default='original', 
                       choices=['original', '128', '256', '384', '512'],
                       help='Input image resolution for processing')
    args, unknown = parser.parse_known_args()
    if args.save_training_template:
        template = {
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "data_path": "<path>",
            "validation_split": 0.2,
            "mlflow_run_id": "<run_id>",
            "model_id": None,
            "random_flip": False,
            "random_rotate": False,
            "random_scale": False,
            "random_intensity": False,
            "crop_size": 128,
            "num_workers": 4,
            "mode": "train"
        }
        import json
        print(json.dumps(template, indent=4))
        sys.exit(0)
    if not args.mode:
        parser.print_help()
        print("\nERROR: You must specify --mode=train or --mode=predict.")
        sys.exit(2)
    return args

def detect_num_classes_from_masks(dataset_loaders, dataset_type="auto", max_samples=50):
    """
    Dynamically detect the number of classes from mask data for semantic segmentation
    
    Args:
        dataset_loaders: Either tuple of (train_loader, val_loader) or (train_ds, val_ds)
        dataset_type: Type of dataset ("auto", "arcade_semantic", "binary", etc.)
        max_samples: Maximum number of samples to check
    
    Returns:
        dict with detected class information: {
            'num_classes': int,
            'class_type': str ('binary', 'multi_class', 'semantic'),
            'unique_values': list,
            'max_channels': int (for one-hot encoded masks)
        }
    """
    logger.info(f"[CLASS DETECTION] Detecting number of classes from mask data...")
    
    try:
        # Handle different dataset loader types
        if hasattr(dataset_loaders[0], '__iter__') and hasattr(dataset_loaders[0], 'dataset'):
            # DataLoader objects (ARCADE)
            train_loader, val_loader = dataset_loaders
            train_dataset = train_loader.dataset
        else:
            # Dataset objects (MONAI)
            train_dataset, val_dataset = dataset_loaders
        
        # Special handling for ARCADE Artery Classification
        if hasattr(train_dataset, '__class__') and 'ARCADEArteryClassification' in str(train_dataset.__class__):
            logger.info(f"[CLASS DETECTION] ARCADEArteryClassification dataset detected")
            logger.info(f"[CLASS DETECTION] This is a classification task: binary mask → left/right artery")
            logger.info(f"[CLASS DETECTION] Output should be 2 classes (0=right, 1=left)")
            return {
                'num_classes': 2,  # Classification task: 2 output classes
                'class_type': 'classification',
                'unique_values': [0, 1],
                'max_channels': 1,
                'task_type': 'artery_classification'
            }
        
        # Collect unique values and shapes from masks
        all_unique_values = set()
        mask_shapes = []
        mask_channels = []
        samples_checked = 0
        
        logger.info(f"[CLASS DETECTION] Checking up to {max_samples} samples from training dataset...")
        
        # Check training dataset samples
        for i in range(min(len(train_dataset), max_samples)):
            try:
                if hasattr(train_dataset, '__getitem__'):
                    image, mask = train_dataset[i]
                else:
                    # For some dataset implementations
                    sample = train_dataset[i]
                    if isinstance(sample, dict):
                        image = sample.get('image', sample.get('img'))
                        mask = sample.get('label', sample.get('mask'))
                    else:
                        image, mask = sample
                
                # Convert to numpy for analysis
                if hasattr(mask, 'numpy'):
                    mask_array = mask.numpy()
                elif hasattr(mask, 'cpu'):
                    mask_array = mask.cpu().numpy()
                else:
                    mask_array = np.array(mask)
                
                # Record shape and channels
                mask_shapes.append(mask_array.shape)
                
                # Determine number of channels based on shape
                if len(mask_array.shape) == 4:
                    # Batch dimension included (B, C, H, W) or (B, H, W, C)
                    if mask_array.shape[1] < mask_array.shape[3]:  # Likely (B, C, H, W)
                        mask_channels.append(mask_array.shape[1])
                    else:  # Likely (B, H, W, C)
                        mask_channels.append(mask_array.shape[3])
                elif len(mask_array.shape) == 3:
                    # Either (C, H, W) or (H, W, C)
                    if mask_array.shape[0] < min(mask_array.shape[1], mask_array.shape[2]):
                        # Likely (C, H, W) format
                        mask_channels.append(mask_array.shape[0])
                        mask_array = mask_array.transpose(1, 2, 0)  # Convert to (H, W, C)
                    else:
                        # Likely (H, W, C) format
                        mask_channels.append(mask_array.shape[2])
                elif len(mask_array.shape) == 2:
                    # Single channel (H, W)
                    mask_channels.append(1)
                else:
                    # Unknown format, default to 1 channel
                    mask_channels.append(1)
                
                # Debug logging for channel detection
                logger.info(f"[CLASS DETECTION] Sample {i}: shape={mask_array.shape}, detected_channels={mask_channels[-1]}")
                
                # For multi-channel masks, check each channel
                if len(mask_array.shape) == 3 and mask_array.shape[2] > 1:
                    # Multi-channel format (H, W, C) - check each channel
                    logger.info(f"[CLASS DETECTION] Processing multi-channel mask with {mask_array.shape[2]} channels")
                    for c in range(mask_array.shape[2]):
                        channel_data = mask_array[:, :, c]
                        if np.any(channel_data > 0):
                            all_unique_values.update(np.unique(channel_data))
                else:
                    # Single channel or already processed
                    if len(mask_array.shape) > 2:
                        mask_array = mask_array.squeeze()
                    all_unique_values.update(np.unique(mask_array))
                
                samples_checked += 1
                
                # Early termination for clear cases
                if samples_checked >= 10 and len(all_unique_values) > 0:
                    break
                    
            except Exception as e:
                logger.warning(f"[CLASS DETECTION] Error processing sample {i}: {e}")
                continue
        
        # Analyze collected data
        unique_values = sorted(list(all_unique_values))
        max_channels = max(mask_channels) if mask_channels else 1
        
        logger.info(f"[CLASS DETECTION] Analyzed {samples_checked} samples")
        logger.info(f"[CLASS DETECTION] Unique mask values: {unique_values}")
        logger.info(f"[CLASS DETECTION] Max channels found: {max_channels}")
        logger.info(f"[CLASS DETECTION] Typical mask shape: {mask_shapes[0] if mask_shapes else 'Unknown'}")
        
        # Determine class type and count
        class_info = _analyze_class_distribution(unique_values, max_channels, dataset_type)
        
        logger.info(f"[CLASS DETECTION] Detection result: {class_info}")
        return class_info
        
    except Exception as e:
        logger.error(f"[CLASS DETECTION] Failed to detect classes: {e}")
        # Return safe defaults
        return {
            'num_classes': 1,
            'class_type': 'binary',
            'unique_values': [0, 1],
            'max_channels': 1
        }

def _analyze_class_distribution(unique_values, max_channels, dataset_type):
    """Analyze unique values and channels to determine class configuration"""
    
    num_unique = len(unique_values)
    
    # Handle one-hot encoded semantic segmentation (ARCADE style)
    if max_channels > 2:
        logger.info(f"[CLASS DETECTION] One-hot encoded semantic segmentation detected with {max_channels} channels")
        return {
            'num_classes': max_channels,
            'class_type': 'semantic_onehot',
            'unique_values': unique_values,
            'max_channels': max_channels
        }
    
    # Handle binary segmentation
    elif num_unique == 2 and set(unique_values) <= {0, 1, 255}:
        logger.info(f"[CLASS DETECTION] Binary segmentation detected")
        return {
            'num_classes': 1,
            'class_type': 'binary',
            'unique_values': unique_values,
            'max_channels': 1
        }
    
    # Handle grayscale binary masks that will be auto-thresholded during training
    elif num_unique > 2:
        min_val = min(unique_values)
        max_val = max(unique_values)
        
        # Check if this looks like grayscale binary masks that need thresholding
        if (min_val == 0 and max_val == 255 and num_unique >= 50) or \
           (min_val >= 0 and max_val <= 1.0 and num_unique >= 10):
            # This is a grayscale mask that will be auto-thresholded to binary during training
            logger.info(f"[CLASS DETECTION] Grayscale binary mask detected ({num_unique} values) - will be auto-thresholded to binary")
            logger.info(f"[CLASS DETECTION] Range: [{min_val}, {max_val}] → will become [0, 1] during training")
            return {
                'num_classes': 1,
                'class_type': 'binary',
                'unique_values': [0, 1],  # What it will become after thresholding
                'max_channels': 1
            }
        else:
            # True multi-class semantic segmentation
            num_classes = num_unique if 0 in unique_values else num_unique + 1
            logger.info(f"[CLASS DETECTION] Multi-class semantic segmentation detected with {num_classes} classes")
            return {
                'num_classes': num_classes,
                'class_type': 'semantic_single',
                'unique_values': unique_values,
                'max_channels': 1
            }
    
    # Default to binary
    else:
        logger.info(f"[CLASS DETECTION] Defaulting to binary segmentation")
        return {
            'num_classes': 1,
            'class_type': 'binary',
            'unique_values': unique_values,
            'max_channels': 1
        }

def train_model(args):
    # Set up logging first thing
    import sys
    
    # Create model directory for this run first to get proper paths
    model_family = getattr(args, 'model_family', 'UNet-Coronary')
    model_dir, unique_id = create_organized_model_directory(
        model_id=args.model_id, 
        model_family=model_family, 
        version="1.0.0"
    )
    
    # Set up logging to both model-specific and global locations
    model_log_path = os.path.join(model_dir, 'logs', 'training.log')
    global_log_path = os.path.join('data', 'models', 'artifacts', 'training.log')
    
    # Ensure both log directories exist
    os.makedirs(os.path.dirname(model_log_path), exist_ok=True)
    os.makedirs(os.path.dirname(global_log_path), exist_ok=True)
    
    # Create a training-specific logger instead of overriding the root logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # Only add handlers if they don't already exist
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        
        # Add file handlers for training logs (append mode to preserve logs)
        model_handler = logging.FileHandler(model_log_path, mode='a', encoding='utf-8', delay=False)
        model_handler.setFormatter(formatter)
        model_handler.setLevel(logging.INFO)  # Only log INFO and above to files
        logger.addHandler(model_handler)
        
        global_handler = logging.FileHandler(global_log_path, mode='a', encoding='utf-8', delay=False)
        global_handler.setFormatter(formatter) 
        global_handler.setLevel(logging.INFO)  # Only log INFO and above to files
        logger.addHandler(global_handler)
        
        # Add console handler for training logs
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)  # Avoid DEBUG spam in console
        logger.addHandler(console_handler)
        
        # Prevent propagation to avoid double logging
        logger.propagate = False
    
    # Log the paths being used
    logger.info(f"[SETUP] Model-specific log: {model_log_path}")
    logger.info(f"[SETUP] Global log: {global_log_path}")
    logger.info(f"[SETUP] Model directory: {model_dir}")
    
    # Setup MLflow experiment before handling run
    try:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        core_dir = os.path.abspath(os.path.join(current_script_dir, '..', '..', 'core', 'apps'))
        if core_dir not in sys.path:
            sys.path.append(core_dir)
        from ml_manager.utils.mlflow_utils import setup_mlflow
        setup_mlflow()
        logger.info("[MLFLOW] MLflow experiment setup completed")
        
        # Verify MLflow connection by listing experiments
        try:
            experiments = mlflow.search_experiments()
            logger.info(f"[MLFLOW] Connection verified - found {len(experiments)} experiments")
            current_experiment = mlflow.get_experiment_by_name(mlflow.get_experiment(mlflow.active_run().info.experiment_id if mlflow.active_run() else "0").name)
            if current_experiment:
                logger.info(f"[MLFLOW] Using experiment: {current_experiment.name}")
                logger.info(f"[MLFLOW] Artifact location: {current_experiment.artifact_location}")
        except Exception as verify_error:
            logger.warning(f"[MLFLOW] Connection verification failed: {verify_error}")
            
    except Exception as e:
        logger.warning(f"[MLFLOW] Failed to setup MLflow experiment: {e}")
    
    # Handle MLflow run in subprocess context
    # The run should already be active from Django, but check to be safe
    current_run = mlflow.active_run()
    if current_run and current_run.info.run_id == args.mlflow_run_id:
        logger.info(f"[MLFLOW] Continuing active run {args.mlflow_run_id}")
    else:
        # If no active run or different run ID, start the specified run
        if current_run:
            mlflow.end_run()  # End any existing run first
        mlflow.start_run(run_id=args.mlflow_run_id)
        logger.info(f"[MLFLOW] Started run {args.mlflow_run_id} in training subprocess")
    
    training_start_time = time.time()  # Track total training duration

    # Initialize system monitoring for MLflow
    system_monitor = None
    try:
        from ml.utils.utils.system_monitor import SystemMonitor
        system_monitor = SystemMonitor(log_interval=30, enable_gpu=True)  # Log every 30 seconds
        system_monitor.start_monitoring()
        logger.info("[MONITORING] System monitoring started - logging to MLflow every 30 seconds")
    except Exception as e:
        logger.warning(f"[MONITORING] Failed to start system monitoring: {e}")
        system_monitor = None

    # Log MLflow parameters and tags for model metadata
    mlflow.log_param("model_family", args.model_family)
    mlflow.log_param("model_type", args.model_type)
    mlflow.log_param("architecture", args.model_type)
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("crop_size", args.crop_size)
    mlflow.log_param("validation_split", args.validation_split)
    mlflow.log_param("num_workers", getattr(args, 'num_workers', 2))
    
    # Log device and resolution parameters
    mlflow.log_param("device", getattr(args, 'device', 'auto'))
    mlflow.log_param("resolution", getattr(args, 'resolution', '256'))
    
    # Log augmentation parameters
    mlflow.log_param("random_flip", getattr(args, 'random_flip', False))
    mlflow.log_param("random_rotate", getattr(args, 'random_rotate', False))
    mlflow.log_param("random_scale", getattr(args, 'random_scale', False))
    mlflow.log_param("random_intensity", getattr(args, 'random_intensity', False))
    
    # Set MLflow tags for better organization
    mlflow.set_tag("model_family", args.model_family)
    mlflow.set_tag("architecture", args.model_type)
    mlflow.set_tag("task", "coronary_segmentation")
    if hasattr(args, 'model_id') and args.model_id is not None:
        mlflow.set_tag("model_id", str(args.model_id))
    
    callback = None
    if hasattr(args, 'model_id') and args.model_id is not None:
        from ml.utils.utils.training_callback import TrainingCallback
        callback = TrainingCallback(args.model_id, args.mlflow_run_id)
        # Store the model directory path in Django model
        callback.set_model_directory(model_dir)
        # Set status to 'loading' when training script starts
        callback.on_training_start()

    try:
        logger.info("[TRAINING] Starting training with parameters: %s", vars(args))
        
        # Handle device selection based on args.device parameter
        if hasattr(args, 'device') and args.device:
            if args.device == 'auto':
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif args.device == 'cuda':
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    device = torch.device("cpu")
            else:
                device = torch.device(args.device)  # cpu or specific device
        else:
            # Fallback to auto-detection if no device specified
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        logger.info(f"[TRAINING] Using device: {device}")
        
        # Log dataset path verification with auto-detection
        logger.info(f"[DATASET] Data path: {args.data_path}")
        if not os.path.exists(args.data_path):
            logger.error(f"[DATASET] Data path does not exist: {args.data_path}")
            raise FileNotFoundError(f"Dataset path not found: {args.data_path}")

        # Get transforms with augmentation parameters
        transform_params = {
            'use_random_flip': getattr(args, 'random_flip', True),
            'use_random_rotate': getattr(args, 'random_rotate', True),
            'use_random_scale': getattr(args, 'random_scale', True),
            'use_random_intensity': getattr(args, 'random_intensity', True),
            'crop_size': getattr(args, 'crop_size', 128)
        }
        
        logger.info("[DATASET] Loading datasets with auto-detection...")
        
        # Use auto-detection to determine dataset type and create appropriate loaders
        try:
            dataset_loaders = get_datasets_with_auto_detection(
                args.data_path, 
                args.validation_split, 
                transform_params, 
                args
            )
            
            # Handle different return types (ARCADE returns DataLoaders, MONAI returns Datasets)
            if isinstance(dataset_loaders[0], MonaiDataLoader) or hasattr(dataset_loaders[0], '__iter__'):
                # ARCADE or pre-built DataLoaders
                train_loader, val_loader = dataset_loaders
                train_samples = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else len(train_loader) * train_loader.batch_size
                val_samples = len(val_loader.dataset) if hasattr(val_loader, 'dataset') else len(val_loader) * val_loader.batch_size
            else:
                # MONAI Datasets - need to create DataLoaders
                train_ds, val_ds = dataset_loaders
                num_workers = min(getattr(args, 'num_workers', 2), 2)
                train_loader = MonaiDataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
                val_loader = MonaiDataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
                train_samples = len(train_ds)
                val_samples = len(val_ds)
                
        except Exception as e:
            logger.error(f"[DATASET] Failed to load datasets: {e}")
            # Fallback to original MONAI method
            logger.info("[DATASET] Falling back to MONAI dataset loader...")
            train_ds, val_ds = get_monai_datasets(args.data_path, args.validation_split, transform_params)
            num_workers = min(getattr(args, 'num_workers', 2), 2)
            train_loader = MonaiDataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
            val_loader = MonaiDataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
            train_samples = len(train_ds)
            val_samples = len(val_ds)

        # Log dataset info
        if 'train_ds' in locals() and 'val_ds' in locals():
            logger.info(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
        elif 'train_loader' in locals() and 'val_loader' in locals():
            logger.info(f"Training samples: {train_samples}, Validation samples: {val_samples}")
        else:
            logger.warning("[DATASET] Could not determine dataset sizes - dataset loading failed.")
        
        # Dataset loaded successfully, now update status to 'training'
        if callback:
            callback.on_dataset_loaded()
            logger.info("Status updated to 'training' - starting model training")
            
            # Update training data info with detailed dataset statistics  
            detailed_training_info = {
                'data_path': args.data_path,
                'total_samples': train_samples + val_samples,
                'training_samples': train_samples,
                'validation_samples': val_samples,
                'validation_split': args.validation_split,
                'transform_params': transform_params,
                'batch_size': args.batch_size,
                'resolution': getattr(args, 'resolution', '256'),
                'crop_size': getattr(args, 'crop_size', 128),
                'dataset_type': getattr(args, 'dataset_type', 'auto'),
                'num_workers': getattr(args, 'num_workers', 4)
            }
            
            # Add dataset-specific paths and file information
            try:
                # Debug logging to track available variables
                available_vars = [k for k in locals().keys() if any(x in k for x in ['ds', 'loader', 'dataset'])]
                logger.info(f"[DATASET] Available variables for metadata: {available_vars}")
                
                # Check if we have MONAI datasets (train_ds, val_ds exist)
                if 'train_ds' in locals() and 'val_ds' in locals():
                    logger.info("[DATASET] Processing MONAI dataset metadata...")
                    if hasattr(train_ds, 'data') and train_ds.data:
                        # For MONAI datasets, extract sample paths
                        train_sample_paths = []
                        val_sample_paths = []
                        
                        # Get sample paths from training dataset
                        for i, sample in enumerate(train_ds.data[:5]):  # First 5 samples
                            if isinstance(sample, dict) and 'image' in sample:
                                train_sample_paths.append({
                                    'image': os.path.basename(sample['image']) if isinstance(sample['image'], str) else f"sample_{i}",
                                    'label': os.path.basename(sample['label']) if isinstance(sample, dict) and 'label' in sample and isinstance(sample['label'], str) else f"label_{i}"
                                })
                        
                        # Get sample paths from validation dataset  
                        if hasattr(val_ds, 'data') and val_ds.data:
                            for i, sample in enumerate(val_ds.data[:5]):  # First 5 samples
                                if isinstance(sample, dict) and 'image' in sample:
                                    val_sample_paths.append({
                                        'image': os.path.basename(sample['image']) if isinstance(sample['image'], str) else f"sample_{i}",
                                        'label': os.path.basename(sample['label']) if isinstance(sample, dict) and 'label' in sample and isinstance(sample['label'], str) else f"label_{i}"
                                    })
                        
                        detailed_training_info.update({
                            'sample_train_files': train_sample_paths,
                            'sample_val_files': val_sample_paths,
                            'train_file_count': len(train_ds.data) if hasattr(train_ds, 'data') else len(train_ds),
                            'val_file_count': len(val_ds.data) if hasattr(val_ds, 'data') else len(val_ds),
                            'dataset_framework': 'MONAI'
                        })
                        
                        logger.info(f"[DATASET] MONAI sample training files: {train_sample_paths[:3]}")
                        logger.info(f"[DATASET] MONAI sample validation files: {val_sample_paths[:3]}")
                        
                # Check if we have ARCADE loaders (train_loader, val_loader exist)  
                elif 'train_loader' in locals() and 'val_loader' in locals():
                    logger.info("[DATASET] Processing ARCADE dataset metadata...")
                    # For ARCADE datasets, get info from DataLoader
                    train_dataset = getattr(train_loader, 'dataset', None)
                    val_dataset = getattr(val_loader, 'dataset', None)
                    
                    if train_dataset and val_dataset:
                        detailed_training_info.update({
                            'dataset_framework': 'ARCADE',
                            'dataset_class': str(type(train_dataset).__name__),
                            'arcade_root': getattr(train_dataset, 'root', 'Unknown'),
                            'arcade_image_set_train': getattr(train_dataset, 'image_set', 'Unknown'),
                            'arcade_image_set_val': getattr(val_dataset, 'image_set', 'Unknown'),
                            'train_file_count': len(train_dataset) if train_dataset else 0,
                            'val_file_count': len(val_dataset) if val_dataset else 0
                        })
                        
                        logger.info(f"[DATASET] ARCADE dataset detected: {type(train_dataset).__name__}")
                        logger.info(f"[DATASET] ARCADE root: {getattr(train_dataset, 'root', 'Unknown')}")
                        logger.info(f"[DATASET] ARCADE train count: {len(train_dataset) if train_dataset else 0}")
                        logger.info(f"[DATASET] ARCADE val count: {len(val_dataset) if val_dataset else 0}")
                else:
                    logger.info("[DATASET] Neither MONAI datasets nor ARCADE loaders found for detailed logging")
                    
            except Exception as e:
                logger.warning(f"[DATASET] Could not extract detailed file information: {e}")
                logger.warning(f"[DATASET] Available variables: {[k for k in locals().keys() if 'ds' in k or 'loader' in k]}")
                
            callback.update_model_metadata(training_data_info=detailed_training_info)
        
        # Get sample batch for model signature - with safe handling
        sample_batch = None
        sample_images = None
        try:
            sample_batch = next(iter(train_loader))
            if isinstance(sample_batch, dict):
                logger.info(f"Sample batch shapes - Image: {sample_batch['image'].shape}, Label: {sample_batch['label'].shape}")
                images = sample_batch['image']
                labels = sample_batch['label']
                sample_images = images
            elif isinstance(sample_batch, (list, tuple)) and len(sample_batch) == 2:
                logger.info(f"Sample batch shapes - Image: {sample_batch[0].shape}, Label: {sample_batch[1].shape}")
                images = sample_batch[0]
                labels = sample_batch[1]
                sample_images = images
                # Create dict-like structure for MLflow signature
                sample_batch = {"image": images, "label": labels}
            else:
                logger.warning(f"Unexpected sample batch type: {type(sample_batch)}, content: {sample_batch}")
                images, labels = None, None
                sample_images = None
                sample_batch = None
        except Exception as e:
            logger.error(f"Failed to get sample batch for model signature: {e}")
            sample_batch = None
            sample_images = None
            images, labels = None, None
        # --- Wizualizacja przykładowych danych wejściowych i masek ---
        try:
            import matplotlib.pyplot as plt
            if images is not None and labels is not None:
                # Log data ranges for debugging normalization issues
                img_min, img_max = images.min().item(), images.max().item()
                label_min, label_max = labels.min().item(), labels.max().item()
                logger.info(f"[DATASET] Sample batch data ranges:")
                logger.info(f"[DATASET]   Images: min={img_min:.4f}, max={img_max:.4f} (dtype: {images.dtype})")
                logger.info(f"[DATASET]   Labels: min={label_min:.4f}, max={label_max:.4f} (dtype: {labels.dtype})")
                
                # Determine and log data range types
                if img_max <= 1.0 and img_min >= 0.0:
                    logger.info(f"[DATASET]   Images appear to be normalized to [0-1] range")
                elif img_max <= 255 and img_min >= 0:
                    logger.info(f"[DATASET]   Images appear to be in [0-255] range")
                else:
                    logger.info(f"[DATASET]   Images in custom range [{img_min:.2f}-{img_max:.2f}]")
                
                if label_max <= 1.0 and label_min >= 0.0:
                    logger.info(f"[DATASET]   Binary masks in [0-1] range (normalized)")
                elif label_max <= 255 and label_min >= 0:
                    logger.info(f"[DATASET]   Binary masks in [0-255] range (needs normalization)")
                else:
                    logger.info(f"[DATASET]   Binary masks in custom range [{label_min:.2f}-{label_max:.2f}]")
                
                # Check for unique values in masks to confirm binary nature
                unique_labels = torch.unique(labels)
                logger.info(f"[DATASET]   Unique mask values: {unique_labels.cpu().numpy()}")
                
                fig, axes = plt.subplots(2, min(4, images.shape[0]), figsize=(12, 6))
                for i in range(min(4, images.shape[0])):
                    axes[0, i].imshow(images[i, 0].cpu().numpy(), cmap='gray')
                    axes[0, i].set_title(f'Input #{i+1}')
                    axes[0, i].axis('off')
                    axes[1, i].imshow(labels[i, 0].cpu().numpy(), cmap='gray')
                    axes[1, i].set_title(f'Segmentacja #{i+1}')
                    axes[1, i].axis('off')
                plt.tight_layout()
                sample_vis_dir = os.path.join(model_dir, 'artifacts')
                os.makedirs(sample_vis_dir, exist_ok=True)
                sample_vis_path = os.path.join(sample_vis_dir, 'sample_inputs_and_masks.png')
                plt.savefig(sample_vis_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"[DATASET] Saved sample input/mask visualization: {sample_vis_path}")
        except Exception as e:
            logger.warning(f"[DATASET] Could not create sample input/mask visualization: {e}")
        
        # --- DYNAMIC CLASS AND CHANNEL DETECTION ---
        # Detect input channels from batch data
        input_channels = images.shape[1] if images is not None else 1
        logger.info(f"[MODEL CONFIG] Detected input channels: {input_channels}")
        
        # Detect output classes from mask data
        class_info = None
        if 'train_loader' in locals() and 'val_loader' in locals():
            # ARCADE dataset loaders
            logger.info("[MODEL CONFIG] Detecting classes from ARCADE dataset...")
            class_info = detect_num_classes_from_masks((train_loader, val_loader), dataset_type="arcade")
        elif 'train_ds' in locals() and 'val_ds' in locals():
            # MONAI datasets
            logger.info("[MODEL CONFIG] Detecting classes from MONAI dataset...")
            class_info = detect_num_classes_from_masks((train_ds, val_ds), dataset_type="monai")
        else:
            logger.warning("[MODEL CONFIG] No dataset available for class detection, using defaults")
        
        # Configure model based on detected parameters
        model_config = get_default_model_config(args.model_type)
        model_config["in_channels"] = input_channels  # Set input channels
        
        # Set output channels based on class detection
        if class_info:
            output_channels = class_info['num_classes']
            logger.info(f"[MODEL CONFIG] Detected {output_channels} output classes ({class_info['class_type']})")
            logger.info(f"[MODEL CONFIG] Class values found: {class_info['unique_values']}")
            logger.info(f"[MODEL CONFIG] Max channels in masks: {class_info['max_channels']}")
            
            # For one-hot encoded semantic segmentation, use the channel count
            if class_info['class_type'] == 'semantic_onehot':
                model_config["out_channels"] = class_info['max_channels']
                logger.info(f"[MODEL CONFIG] Using {class_info['max_channels']} output channels for one-hot semantic segmentation")
            else:
                model_config["out_channels"] = output_channels
                logger.info(f"[MODEL CONFIG] Using {output_channels} output channels for {class_info['class_type']} segmentation")
        else:
            # Fallback to default
            default_out_channels = 1
            model_config["out_channels"] = default_out_channels
            logger.warning(f"[MODEL CONFIG] Using default {default_out_channels} output channels")
        
        # Create model with dynamically configured parameters
        logger.info(f"[MODEL CONFIG] Final model configuration: {model_config}")
        model, arch_info = create_model_from_registry(
            args.model_type, 
            device,
            **model_config
        )

        # Update model metadata if callback is available
        if callback:
            callback.update_model_metadata(
                model_family=args.model_family,
                model_type=args.model_type,
                architecture_info=arch_info
            )
            
            # Prepare training config with class detection info
            training_config = {
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'optimizer': getattr(args, 'optimizer', 'adam'),
                'crop_size': args.crop_size,
                'validation_split': args.validation_split,
                'num_workers': getattr(args, 'num_workers', 2),
                'random_flip': getattr(args, 'random_flip', False),
                'random_rotate': getattr(args, 'random_rotate', False),
                'random_scale': getattr(args, 'random_scale', False),
                'random_intensity': getattr(args, 'random_intensity', False)
            }
            
            # Add class detection information to training config
            if class_info:
                training_config.update({
                    'detected_num_classes': class_info['num_classes'],
                    'detected_class_type': class_info['class_type'],
                    'detected_unique_values': class_info['unique_values'],
                    'detected_max_channels': class_info['max_channels']
                })
                logger.info(f"[MODEL CONFIG] Added class detection info to training config")
            
            callback.update_training_config(training_config)
            callback.update_architecture_info(model, model_config)

        # Model directory was already created during logging setup
        # Save and log model architecture summary
        summary_file = save_model_summary(model, model_dir=model_dir)
        mlflow.log_artifact(summary_file)
        # Save and log training configuration
        config_file = save_config(args, model_dir=model_dir)
        mlflow.log_artifact(config_file)

        # --- VALIDATE MODEL CONFIGURATION WITH DETECTED CLASSES ---
        # Get model output channels to validate against detected classes
        model_output_channels = None
        if hasattr(model, 'outc') and hasattr(model.outc, 'conv'):
            model_output_channels = model.outc.conv.out_channels
        elif hasattr(model, 'out_conv'):
            model_output_channels = model.out_conv.out_channels
        elif hasattr(model, 'segmentation_head'):
            model_output_channels = model.segmentation_head.out_channels
        
        if model_output_channels and class_info:
            logger.info(f"[MODEL VALIDATION] Model output channels: {model_output_channels}")
            logger.info(f"[MODEL VALIDATION] Detected classes: {class_info['num_classes']} ({class_info['class_type']})")
            
            # Validate channel match
            expected_channels = class_info['max_channels'] if class_info['class_type'] == 'semantic_onehot' else class_info['num_classes']
            if model_output_channels == expected_channels:
                logger.info(f"[MODEL VALIDATION] ✅ Model output channels match detected classes")
            else:
                logger.warning(f"[MODEL VALIDATION] ⚠️  Model output channels ({model_output_channels}) don't match expected ({expected_channels})")
                logger.warning(f"[MODEL VALIDATION] This may cause training issues - check model configuration")

        # Configure loss function based on detected class information
        if class_info and class_info.get('task_type') == 'artery_classification':
            # Classification task (ARCADEArteryClassification)
            logger.info(f"[LOSS CONFIG] Using classification loss for artery classification (2 classes)")
            loss_function = torch.nn.CrossEntropyLoss()
        elif class_info and class_info['class_type'] == 'semantic_onehot' and class_info['max_channels'] > 1:
            # Multi-class semantic segmentation with one-hot encoding
            logger.info(f"[LOSS CONFIG] Using multi-class loss for {class_info['max_channels']} classes")
            loss_function = MonaiDiceLoss(sigmoid=False, softmax=True)  # Use softmax for multi-class
        else:
            # Binary segmentation (default)
            logger.info(f"[LOSS CONFIG] Using binary segmentation loss")
            loss_function = MonaiDiceLoss(sigmoid=True)
        
        # Create optimizer based on args.optimizer choice
        optimizer = create_optimizer(model, args)
        
        # Initialize dynamic learning rate scheduler
        lr_scheduler = DynamicLearningRateScheduler(
            initial_lr=args.learning_rate,
            model_id=args.model_id if hasattr(args, 'model_id') and args.model_id else None
        )
        logger.info(f"[LR_SCHEDULER] Dynamic learning rate scheduler initialized with LR: {args.learning_rate}")
        
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        scaler = torch.cuda.amp.GradScaler()
        
        best_val_dice = -1
        best_model_path = None
        
        epoch_history = []  # Track metrics for each epoch
        
        # Set total number of batches per epoch for progress tracking
        if callback:
            callback.set_epoch_batches(len(train_loader))
            
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            logger.info(f"[EPOCH] Starting epoch {epoch+1}/{args.epochs}")
            
            if epoch == 0:
                logger.info("[MODEL] Model Architecture Summary:")
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"[MODEL] Total parameters: {total_params:,}")
                logger.info(f"[MODEL] Trainable parameters: {trainable_params:,}")
                logger.info(f"[MODEL] Model size: {total_params * 4 / (1024**2):.2f} MB")
                
                # Log training configuration
                logger.info(f"[CONFIG] Batch size: {args.batch_size}")
                if 'train_ds' in locals() and 'val_ds' in locals():
                    logger.info(f"[CONFIG] Dataset: Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
                elif 'train_loader' in locals() and 'val_loader' in locals():
                    logger.info(f"[CONFIG] Dataset: Train samples: {train_samples}, Val samples: {val_samples}")
                else:
                    logger.info("[CONFIG] Dataset: Could not determine sample counts.")
                logger.info(f"[CONFIG] Optimizer: {type(optimizer).__name__}, Loss: DiceLoss, Device: {device}")
            
            # Check for stop_requested flag using callback system
            if callback and not callback.on_epoch_start(epoch, args.epochs):
                logger.info("Stop requested via callback. Exiting training loop.")
                break
            elif hasattr(args, 'model_id') and args.model_id is not None and callback is None:
                # Fallback for stop checking if callback is not available
                try:
                    import django
                    import os as _os
                    # Setup Django if not already
                    if not hasattr(django.conf.settings, 'configured') or not django.conf.settings.configured:
                        _os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
                        django.setup()
                    from ml_manager.models import MLModel
                    model_obj = MLModel.objects.get(pk=args.model_id)
                    if getattr(model_obj, 'stop_requested', False):
                        logger.info("Stop requested. Exiting training loop.")
                        break
                except Exception as e:
                    logger.warning(f"Could not check stop_requested flag: {e}")
            model.train()
            epoch_loss = 0
            train_dice = 0
            
            for batch_idx, batch_data in enumerate(train_loader):
                # Call batch start callback
                if callback and not callback.on_batch_start(batch_idx, len(train_loader)):
                    logger.info("Stop requested during batch. Exiting training loop.")
                    break
                    
                if isinstance(batch_data, dict):
                    inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                else:
                    raise TypeError(f"Unsupported batch_data type: {type(batch_data)}")
                # Debug tensor shapes on first batch
                if epoch == 0 and batch_idx == 0:
                    logger.info(f"Training batch shapes - Inputs: {inputs.shape}, Labels: {labels.shape}")
                    
                    # Enhanced range logging for first training batch
                    input_min, input_max = inputs.min().item(), inputs.max().item()
                    input_mean, input_std = inputs.mean().item(), inputs.std().item()
                    label_min, label_max = labels.min().item(), labels.max().item()
                    label_mean, label_std = labels.mean().item(), labels.std().item()
                    
                    logger.info(f"[TRAIN BATCH] 📊 COMPREHENSIVE DATA ANALYSIS:")
                    logger.info(f"[TRAIN BATCH]   🖼️  INPUT IMAGES:")
                    logger.info(f"[TRAIN BATCH]     Range: [{input_min:.4f}, {input_max:.4f}] (dtype: {inputs.dtype})")
                    logger.info(f"[TRAIN BATCH]     Stats: mean={input_mean:.4f}, std={input_std:.4f}")
                    
                    # Determine and log data range types for training images
                    if input_max <= 1.0 and input_min >= 0.0:
                        if input_mean < 0.1:
                            logger.info(f"[TRAIN BATCH]     ✅ Normalized [0-1] range (likely medical images with dark background)")
                        else:
                            logger.info(f"[TRAIN BATCH]     ✅ Normalized [0-1] range (standard normalization)")
                    elif input_max <= 255 and input_min >= 0:
                        if input_max == 255:
                            logger.info(f"[TRAIN BATCH]     ⚠️  [0-255] range detected (8-bit images, consider normalization)")
                        else:
                            logger.info(f"[TRAIN BATCH]     ⚠️  [0-{input_max:.0f}] range (partial 8-bit scale)")
                    elif input_max > 1000:
                        logger.info(f"[TRAIN BATCH]     🔍 High-value range [{input_min:.0f}-{input_max:.0f}] (DICOM/medical data?)")
                    else:
                        logger.info(f"[TRAIN BATCH]     ❓ Custom range [{input_min:.2f}-{input_max:.2f}]")
                    
                    logger.info(f"[TRAIN BATCH]   🎭 MASK LABELS:")
                    logger.info(f"[TRAIN BATCH]     Range: [{label_min:.4f}, {label_max:.4f}] (dtype: {labels.dtype})")
                    logger.info(f"[TRAIN BATCH]     Stats: mean={label_mean:.4f}, std={label_std:.4f}")
                    
                    # Determine and log data range types for training masks
                    if label_max <= 1.0 and label_min >= 0.0:
                        logger.info(f"[TRAIN BATCH]     ✅ Binary normalized [0-1] range")
                    elif label_max <= 255 and label_min >= 0:
                        logger.info(f"[TRAIN BATCH]     ⚠️  [0-255] range (needs binary normalization)")
                    else:
                        logger.info(f"[TRAIN BATCH]     ❓ Custom range [{label_min:.2f}-{label_max:.2f}]")
                    
                    # Check for unique values in training masks with intelligent binary detection
                    unique_train_labels = torch.unique(labels)
                    unique_values_array = unique_train_labels.cpu().numpy()
                    logger.info(f"[TRAIN BATCH]     Raw unique values: {unique_values_array}")
                    
                    # Intelligent binary segmentation detection
                    if len(unique_train_labels) == 2:
                        # Check if it's standard binary (0,1) or 8-bit binary (0,255)
                        min_val, max_val = unique_values_array.min(), unique_values_array.max()
                        if (min_val == 0 and max_val == 1):
                            logger.info(f"[TRAIN BATCH]     ✅ Binary segmentation confirmed (0,1 format)")
                            positive_ratio = (labels == 1).float().mean().item()
                        elif (min_val == 0 and max_val == 255):
                            logger.info(f"[TRAIN BATCH]     ✅ Binary segmentation confirmed (0,255 format - will be normalized)")
                            positive_ratio = (labels == 255).float().mean().item()
                        else:
                            logger.info(f"[TRAIN BATCH]     ✅ Binary segmentation confirmed (custom {min_val},{max_val} format)")
                            positive_ratio = (labels == max_val).float().mean().item()
                        logger.info(f"[TRAIN BATCH]     📈 Class distribution: {positive_ratio:.2%} positive, {1-positive_ratio:.2%} background")
                    elif len(unique_train_labels) == 1:
                        single_val = unique_values_array[0]
                        if single_val == 0:
                            logger.info(f"[TRAIN BATCH]     ⚠️  Single class detected (all background) - check data!")
                        elif single_val == 1 or single_val == 255:
                            logger.info(f"[TRAIN BATCH]     ⚠️  Single class detected (all foreground) - check data!")
                        else:
                            logger.info(f"[TRAIN BATCH]     ⚠️  Single class detected (all {single_val}) - check data!")
                    else:
                        # Check if it's grayscale values that need thresholding
                        if len(unique_train_labels) > 2:
                            # Check if it's normalized grayscale (many values between 0-1) or 8-bit grayscale (0-255)
                            if label_max <= 1.0 and label_min >= 0 and len(unique_train_labels) >= 50:
                                # Many values in [0-1] range = normalized grayscale masks
                                logger.info(f"[TRAIN BATCH]     📝 Normalized grayscale mask detected ({len(unique_train_labels)} values)")
                                logger.info(f"[TRAIN BATCH]     💡 Recommend thresholding: values > 0.5 → 1, else → 0")
                                # Show sample distribution for debugging
                                sample_values = sorted(unique_values_array)
                                logger.info(f"[TRAIN BATCH]     🔍 Sample values: {sample_values[:5]}...{sample_values[-5:]}")
                            elif label_max <= 255 and label_min >= 0:
                                # Values in [0-255] range = 8-bit grayscale masks
                                logger.info(f"[TRAIN BATCH]     📝 8-bit grayscale mask detected ({len(unique_train_labels)} values)")
                                logger.info(f"[TRAIN BATCH]     💡 Recommend thresholding: values > 127 → 1, else → 0")
                                if len(unique_train_labels) <= 10:
                                    logger.info(f"[TRAIN BATCH]     🔍 All values: {sorted(unique_values_array)}")
                            else:
                                logger.info(f"[TRAIN BATCH]     ⚠️  Multi-class ({len(unique_train_labels)} classes) - not binary segmentation")
                        else:
                            logger.info(f"[TRAIN BATCH]     ⚠️  Multi-class ({len(unique_train_labels)} classes) - not binary segmentation")
                    
                    # Additional data quality checks
                    nan_inputs = torch.isnan(inputs).sum().item()
                    inf_inputs = torch.isinf(inputs).sum().item()
                    nan_labels = torch.isnan(labels).sum().item()
                    inf_labels = torch.isinf(labels).sum().item()
                    
                    if nan_inputs > 0 or inf_inputs > 0:
                        logger.warning(f"[TRAIN BATCH]     ❌ Data quality issues - NaN: {nan_inputs}, Inf: {inf_inputs} in inputs")
                    if nan_labels > 0 or inf_labels > 0:
                        logger.warning(f"[TRAIN BATCH]     ❌ Data quality issues - NaN: {nan_labels}, Inf: {inf_labels} in labels")
                    if nan_inputs == 0 and inf_inputs == 0 and nan_labels == 0 and inf_labels == 0:
                        logger.info(f"[TRAIN BATCH]     ✅ Data quality check passed (no NaN/Inf values)")
                
                # Ensure labels are float and binary (0 or 1) with automatic normalization
                labels = labels.float()
                
                # Auto-normalize and threshold masks if they're not binary
                if labels.max() > 1:
                    if labels.max() <= 255 and labels.min() >= 0:
                        logger.info(f"[TRAIN BATCH] 🔧 Auto-normalizing masks from [0-255] to [0-1] range")
                        labels = labels / 255.0
                        # Ensure binary values after normalization
                        labels = (labels > 0.5).float()
                        logger.info(f"[TRAIN BATCH] ✅ Masks normalized to range [{labels.min().item():.1f}-{labels.max().item():.1f}]")
                    else:
                        logger.warning(f"[TRAIN BATCH] ❌ Labels out of expected range! min: {labels.min().item()}, max: {labels.max().item()}")
                        # Try to normalize anyway for custom ranges
                        labels_max = labels.max()
                        if labels_max > 0:
                            labels = labels / labels_max
                            labels = (labels > 0.5).float()
                            logger.info(f"[TRAIN BATCH] 🔧 Normalized custom range to binary [0-1]")
                elif labels.max() <= 1 and labels.min() >= 0:
                    # Check if it's grayscale masks that need thresholding (many values between 0-1)
                    unique_for_threshold = torch.unique(labels)
                    if len(unique_for_threshold) > 10:  # More than 10 unique values = grayscale
                        logger.info(f"[TRAIN BATCH] 🔧 Auto-thresholding grayscale masks ({len(unique_for_threshold)} values → binary)")
                        labels = (labels > 0.5).float()
                        final_unique = torch.unique(labels)
                        logger.info(f"[TRAIN BATCH] ✅ Thresholded to {len(final_unique)} values: {final_unique.cpu().numpy()}")
                
                if labels.max() > 1 or labels.min() < 0:
                    logger.warning(f"[TRAIN BATCH] ❌ Labels still out of [0,1] range after normalization! min: {labels.min().item()}, max: {labels.max().item()}")
                
                # Verify normalization worked correctly for first batch
                if epoch == 0 and batch_idx == 0:
                    # Re-check unique values after normalization
                    normalized_unique_labels = torch.unique(labels)
                    logger.info(f"[TRAIN BATCH] 🔍 POST-NORMALIZATION VERIFICATION:")
                    logger.info(f"[TRAIN BATCH]     Final unique values: {normalized_unique_labels.cpu().numpy()}")
                    logger.info(f"[TRAIN BATCH]     Final range: [{labels.min().item():.4f}, {labels.max().item():.4f}]")
                    
                    if len(normalized_unique_labels) == 2 and labels.min() >= 0 and labels.max() <= 1:
                        logger.info(f"[TRAIN BATCH]     ✅ Successfully normalized to binary [0-1] format")
                        positive_ratio_final = (labels == 1).float().mean().item()
                        logger.info(f"[TRAIN BATCH]     📈 Final class distribution: {positive_ratio_final:.2%} positive, {1-positive_ratio_final:.2%} background")
                    else:
                        logger.warning(f"[TRAIN BATCH]     ⚠️  Normalization may have failed - {len(normalized_unique_labels)} unique values")
                # Use DiceLoss with sigmoid=False, since model outputs are raw logits
                loss_function = MonaiDiceLoss(sigmoid=True)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    if epoch == 0 and batch_idx == 0:
                        logger.info(f"Model output shape: {outputs.shape}")
                    loss = loss_function(outputs, labels)
                
                if loss.item() < 0:
                    logger.warning(f"Negative loss detected! outputs min/max: {outputs.min().item()}/{outputs.max().item()}, labels min/max: {labels.min().item()}/{labels.max().item()}")
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                
                # Calculate training dice score
                with torch.no_grad():
                    val_outputs = torch.sigmoid(outputs)
                    val_outputs = (val_outputs > 0.5).float()
                    dice_metric(y_pred=val_outputs, y=labels)
                    batch_dice = dice_metric.aggregate().item()
                    dice_metric.reset()
                    train_dice += batch_dice
                
                # Call batch end callback with current metrics
                if callback:
                    batch_logs = {
                        'train_loss': loss.item(),
                        'train_dice': batch_dice,
                        'batch_progress': (batch_idx + 1) / len(train_loader)
                    }
                    callback.on_batch_end(batch_idx, batch_logs)
                
                if batch_idx % 10 == 0:  # More frequent logging for GUI
                    progress_pct = (batch_idx / len(train_loader)) * 100
                    logger.info(f"[TRAIN] Epoch {epoch+1}/{args.epochs} - "
                              f"Batch {batch_idx}/{len(train_loader)} ({progress_pct:.1f}%) - "
                              f"Loss: {loss.item():.4f}, Dice: {batch_dice:.4f}")
                              
            # Check if training was stopped during batch processing
            if callback and callback.model.stop_requested:
                logger.info("Training stopped during batch processing.")
                break
            
            epoch_loss /= len(train_loader)
            train_dice /= len(train_loader)
            
            # Validation
            logger.info(f"[VAL] Starting validation for epoch {epoch+1}")
            model.eval()
            val_loss = 0
            val_dice = 0
            
            # --- Enhanced validation data analysis ---
            try:
                val_batch = next(iter(val_loader))
                if isinstance(val_batch, dict):
                    val_images = val_batch["image"]
                    val_labels = val_batch["label"]
                elif isinstance(val_batch, (list, tuple)) and len(val_batch) == 2:
                    val_images, val_labels = val_batch
                else:
                    val_images, val_labels = None, None
                    
                # Only log validation data analysis on first epoch
                if epoch == 0 and val_images is not None and val_labels is not None:
                    logger.info(f"[VAL DATA] Validation batch: images shape: {val_images.shape}, masks shape: {val_labels.shape}")
                    
                    # Enhanced validation data range logging
                    val_img_min, val_img_max = val_images.min().item(), val_images.max().item()
                    val_img_mean, val_img_std = val_images.mean().item(), val_images.std().item()
                    val_label_min, val_label_max = val_labels.min().item(), val_labels.max().item()
                    val_label_mean, val_label_std = val_labels.mean().item(), val_labels.std().item()
                    
                    logger.info(f"[VAL DATA] 📊 VALIDATION DATA ANALYSIS:")
                    logger.info(f"[VAL DATA]   🖼️  VALIDATION IMAGES:")
                    logger.info(f"[VAL DATA]     Range: [{val_img_min:.4f}, {val_img_max:.4f}] (dtype: {val_images.dtype})")
                    logger.info(f"[VAL DATA]     Stats: mean={val_img_mean:.4f}, std={val_img_std:.4f}")
                    
                    # Determine and log data range types for validation images
                    if val_img_max <= 1.0 and val_img_min >= 0.0:
                        if val_img_mean < 0.1:
                            logger.info(f"[VAL DATA]     ✅ Normalized [0-1] range (likely medical images with dark background)")
                        else:
                            logger.info(f"[VAL DATA]     ✅ Normalized [0-1] range (standard normalization)")
                    elif val_img_max <= 255 and val_img_min >= 0:
                        if val_img_max == 255:
                            logger.info(f"[VAL DATA]     ⚠️  [0-255] range detected (8-bit images, consider normalization)")
                        else:
                            logger.info(f"[VAL DATA]     ⚠️  [0-{val_img_max:.0f}] range (partial 8-bit scale)")
                    elif val_img_max > 1000:
                        logger.info(f"[VAL DATA]     🔍 High-value range [{val_img_min:.0f}-{val_img_max:.0f}] (DICOM/medical data?)")
                    else:
                        logger.info(f"[VAL DATA]     ❓ Custom range [{val_img_min:.2f}-{val_img_max:.2f}]")
                    
                    logger.info(f"[VAL DATA]   🎭 VALIDATION MASKS:")
                    logger.info(f"[VAL DATA]     Range: [{val_label_min:.4f}, {val_label_max:.4f}] (dtype: {val_labels.dtype})")
                    logger.info(f"[VAL DATA]     Stats: mean={val_label_mean:.4f}, std={val_label_std:.4f}")
                    
                    # Determine and log data range types for validation masks
                    if val_label_max <= 1.0 and val_label_min >= 0.0:
                        logger.info(f"[VAL DATA]     ✅ Binary normalized [0-1] range")
                    elif val_label_max <= 255 and val_label_min >= 0:
                        logger.info(f"[VAL DATA]     ⚠️  [0-255] range (needs binary normalization)")
                    else:
                        logger.info(f"[VAL DATA]     ❓ Custom range [{val_label_min:.2f}-{val_label_max:.2f}]")
                    
                    # Check for unique values in validation masks with intelligent binary detection
                    unique_val_labels = torch.unique(val_labels)
                    unique_val_values_array = unique_val_labels.cpu().numpy()
                    logger.info(f"[VAL DATA]     Raw unique values: {unique_val_values_array}")
                    
                    # Intelligent binary segmentation detection for validation
                    if len(unique_val_labels) == 2:
                        # Check if it's standard binary (0,1) or 8-bit binary (0,255)
                        min_val, max_val = unique_val_values_array.min(), unique_val_values_array.max()
                        if (min_val == 0 and max_val == 1):
                            logger.info(f"[VAL DATA]     ✅ Binary segmentation confirmed (0,1 format)")
                            val_positive_ratio = (val_labels == 1).float().mean().item()
                        elif (min_val == 0 and max_val == 255):
                            logger.info(f"[VAL DATA]     ✅ Binary segmentation confirmed (0,255 format - will be normalized)")
                            val_positive_ratio = (val_labels == 255).float().mean().item()
                        else:
                            logger.info(f"[VAL DATA]     ✅ Binary segmentation confirmed (custom {min_val},{max_val} format)")
                            val_positive_ratio = (val_labels == max_val).float().mean().item()
                        logger.info(f"[VAL DATA]     📈 Class distribution: {val_positive_ratio:.2%} positive, {1-val_positive_ratio:.2%} background")
                    elif len(unique_val_labels) == 1:
                        single_val = unique_val_values_array[0]
                        if single_val == 0:
                            logger.info(f"[VAL DATA]     ⚠️  Single class detected (all background) - check data!")
                        elif single_val == 1 or single_val == 255:
                            logger.info(f"[VAL DATA]     ⚠️  Single class detected (all foreground) - check data!")
                        else:
                            logger.info(f"[VAL DATA]     ⚠️  Single class detected (all {single_val}) - check data!")
                    else:
                        # Check if it's grayscale values that need thresholding
                        if len(unique_val_labels) > 2:
                            # Check if it's normalized grayscale (many values between 0-1) or 8-bit grayscale (0-255)
                            if val_label_max <= 1.0 and val_label_min >= 0 and len(unique_val_labels) >= 50:
                                # Many values in [0-1] range = normalized grayscale masks
                                logger.info(f"[VAL DATA]     📝 Normalized grayscale mask detected ({len(unique_val_labels)} values)")
                                logger.info(f"[VAL DATA]     💡 Recommend thresholding: values > 0.5 → 1, else → 0")
                                # Show sample distribution for debugging
                                sample_values = sorted(unique_val_values_array)
                                logger.info(f"[VAL DATA]     🔍 Sample values: {sample_values[:5]}...{sample_values[-5:]}")
                            elif val_label_max <= 255 and val_label_min >= 0:
                                # Values in [0-255] range = 8-bit grayscale masks
                                logger.info(f"[VAL DATA]     📝 8-bit grayscale mask detected ({len(unique_val_labels)} values)")
                                logger.info(f"[VAL DATA]     💡 Recommend thresholding: values > 127 → 1, else → 0")
                                if len(unique_val_labels) <= 10:
                                    logger.info(f"[VAL DATA]     🔍 All values: {sorted(unique_val_values_array)}")
                            else:
                                logger.info(f"[VAL DATA]     ⚠️  Multi-class ({len(unique_val_labels)} classes) - not binary segmentation")
                        else:
                            logger.info(f"[VAL DATA]     ⚠️  Multi-class ({len(unique_val_labels)} classes) - not binary segmentation")
                    
                    # Additional validation data quality checks
                    val_nan_inputs = torch.isnan(val_images).sum().item()
                    val_inf_inputs = torch.isinf(val_images).sum().item()
                    val_nan_labels = torch.isnan(val_labels).sum().item()
                    val_inf_labels = torch.isinf(val_labels).sum().item()
                    
                    if val_nan_inputs > 0 or val_inf_inputs > 0:
                        logger.warning(f"[VAL DATA]     ❌ Data quality issues - NaN: {val_nan_inputs}, Inf: {val_inf_inputs} in inputs")
                    if val_nan_labels > 0 or val_inf_labels > 0:
                        logger.warning(f"[VAL DATA]     ❌ Data quality issues - NaN: {val_nan_labels}, Inf: {val_inf_labels} in labels")
                    if val_nan_inputs == 0 and val_inf_inputs == 0 and val_nan_labels == 0 and val_inf_labels == 0:
                        logger.info(f"[VAL DATA]     ✅ Data quality check passed (no NaN/Inf values)")
                
                # Always generate visualization (moved after conditional logging)
                if val_images is not None and val_labels is not None:
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(2, min(4, val_images.shape[0]), figsize=(12, 6))
                    for i in range(min(4, val_images.shape[0])):
                        axes[0, i].imshow(val_images[i, 0].cpu().numpy(), cmap='gray')
                        axes[0, i].set_title(f'Val input #{i+1}')
                        axes[0, i].axis('off')
                        axes[1, i].imshow(val_labels[i, 0].cpu().numpy(), cmap='gray')
                        axes[1, i].set_title(f'Val mask #{i+1}')
                        axes[1, i].axis('off')
                    plt.tight_layout()
                    vis_dir = os.path.join(model_dir, 'artifacts')
                    os.makedirs(vis_dir, exist_ok=True)
                    vis_path = os.path.join(vis_dir, 'val_sample_inputs_and_masks.png')
                    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    logger.info(f"[VAL DATA] Zapisano wizualizację walidacyjnych wejść/masek: {vis_path}")
            except Exception as e:
                logger.warning(f"[VAL DATA] Nie udało się zwizualizować batcha walidacyjnego: {e}")

            with torch.no_grad():
                for val_idx, val_data in enumerate(val_loader):
                    if isinstance(val_data, dict):
                        val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    elif isinstance(val_data, (list, tuple)) and len(val_data) == 2:
                        val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
                    else:
                        raise TypeError(f"Unsupported val_data type: {type(val_data)}")
                    
                    # Apply same normalization and thresholding to validation labels as training
                    val_labels = val_labels.float()
                    if val_labels.max() > 1:
                        if val_labels.max() <= 255 and val_labels.min() >= 0:
                            val_labels = val_labels / 255.0
                            val_labels = (val_labels > 0.5).float()
                        else:
                            # Normalize custom ranges
                            labels_max = val_labels.max()
                            if labels_max > 0:
                                val_labels = val_labels / labels_max
                                val_labels = (val_labels > 0.5).float()
                    elif val_labels.max() <= 1 and val_labels.min() >= 0:
                        # Check if it's grayscale masks that need thresholding
                        unique_for_threshold = torch.unique(val_labels)
                        if len(unique_for_threshold) > 10:  # More than 10 unique values = grayscale
                            val_labels = (val_labels > 0.5).float()
                    
                    val_outputs = model(val_inputs)
                    batch_val_loss = loss_function(val_outputs, val_labels).item()
                    val_loss += batch_val_loss
                    
                    # Apply appropriate post-processing based on number of output channels
                    num_output_channels = val_outputs.shape[1]
                    if num_output_channels == 1:
                        # Binary segmentation
                        val_outputs = torch.sigmoid(val_outputs)
                        val_outputs = (val_outputs > 0.5).float()
                    else:
                        # Multi-class semantic segmentation
                        val_outputs = torch.softmax(val_outputs, dim=1)
                        val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True).float()
                    
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    if val_idx % 5 == 0:
                        val_progress_pct = (val_idx / len(val_loader)) * 100
                        logger.info(f"[VAL] Batch {val_idx}/{len(val_loader)} ({val_progress_pct:.1f}%) - Loss: {batch_val_loss:.4f}")
                
                val_loss /= len(val_loader)
                val_dice = dice_metric.aggregate().item()
                dice_metric.reset()
            
            metrics = {
                "train_loss": epoch_loss,
                "train_dice": train_dice,
                "val_loss": val_loss,
                "val_dice": val_dice,
            }
            
            logger.info(f"[EPOCH] {epoch+1}/{args.epochs} COMPLETED - "
                       f"Train Loss: {epoch_loss:.4f}, Train Dice: {train_dice:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
            
            # Log learning rate and other training details
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"[METRICS] Learning Rate: {current_lr:.6f}, "
                       f"Best Val Dice: {best_val_dice:.4f}")
            
            # Log batch statistics
            logger.info(f"[STATS] Total batches processed: {len(train_loader)} train, {len(val_loader)} val")
            
            mlflow.log_metrics(metrics, step=epoch)
            epoch_history.append(metrics)  # Save metrics for this epoch
            
            # Call epoch end callback with metrics
            if callback:
                callback.on_epoch_end(epoch, metrics)
            
            # Enhanced MLflow artifact logging using the new artifact manager
            try:
                from ml.utils.utils.mlflow_artifact_manager import log_epoch_artifacts
                
                # Prepare artifacts dictionary for this epoch
                epoch_artifacts = {}
                
                # Save sample predictions every epoch (not just every 5 epochs)
                try:
                    pred_file = save_sample_predictions(model, val_loader, device, epoch, model_dir=model_dir)
                    if pred_file and os.path.exists(pred_file):
                        epoch_artifacts['predictions'] = pred_file
                        mlflow.log_artifact(pred_file, artifact_path=f"predictions/epoch_{epoch+1:03d}")
                        logger.info(f"[MLFLOW] Successfully logged prediction samples: {pred_file}")
                    else:
                        logger.warning(f"[MLFLOW] Prediction file was not created or does not exist: {pred_file}")
                except Exception as pred_error:
                    logger.error(f"[MLFLOW] Failed to save/log sample predictions: {pred_error}")
                
                # Save enhanced training curves if we have enough data
                if len(epoch_history) >= 2:
                    # Use the existing model directory instead of creating a new one
                    
                    enhanced_curves_file = save_enhanced_training_curves(
                        epoch_history, model_dir, epoch
                    )
                    if enhanced_curves_file:
                        epoch_artifacts['training_curves'] = enhanced_curves_file
                        # Log training curves with organized path
                        mlflow.log_artifact(enhanced_curves_file, artifact_path=f"visualizations/training_curves/epoch_{epoch+1:03d}")
                    
                    # Save model comparison artifacts
                    comparison_artifacts = save_model_comparison_artifacts(
                        model_dir, 
                        {**metrics, 'epoch_history': epoch_history}, 
                        epoch
                    )
                    if comparison_artifacts:
                        for i, artifact in enumerate(comparison_artifacts):
                            epoch_artifacts[f'comparison_{i}'] = artifact
                            # Log comparison artifacts with organized paths
                            mlflow.log_artifact(artifact, artifact_path=f"visualizations/comparisons/epoch_{epoch+1:03d}")
                
                # Save current epoch configuration and log it
                epoch_config = {
                    'epoch': epoch + 1,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'batch_size': args.batch_size,
                    'device': str(device),
                    'metrics': metrics
                }
                config_file = os.path.join(model_dir, 'logs', f'epoch_{epoch+1:03d}_config.json')
                os.makedirs(os.path.dirname(config_file), exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(epoch_config, f, indent=2)
                epoch_artifacts['config'] = config_file
                mlflow.log_artifact(config_file, artifact_path=f"config/epoch_{epoch+1:03d}")
                
                # Prepare metadata for this epoch
                epoch_metadata = {
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'batch_size': args.batch_size,
                    'total_batches': len(train_loader),
                    'model_family': getattr(args, 'model_family', 'UNet-Coronary'),
                    'device': str(device),
                    'optimizer': 'Adam',
                    'loss_function': 'DiceLoss',
                    'epoch_duration': time.time() - epoch_start_time if 'epoch_start_time' in locals() else 0
                }
                
                # Log using enhanced artifact manager
                logged_paths = log_epoch_artifacts(
                    epoch=epoch + 1,  # 1-based epoch numbering
                    model_state=model.state_dict() if val_dice > best_val_dice else None,
                    metrics=metrics,
                    artifacts=epoch_artifacts,
                    metadata=epoch_metadata
                )
                
                logger.info(f"[MLFLOW] Enhanced artifact logging completed for epoch {epoch+1}: {len(logged_paths)} artifacts")
                logger.info(f"[MLFLOW] Artifact paths: {list(logged_paths.keys())}")
                
            except Exception as e:
                logger.warning(f"[MLFLOW] Enhanced artifact logging failed, falling back to basic logging: {e}")
                
                # Fallback to original artifact logging - Generate predictions every epoch
                pred_file = save_sample_predictions(model, val_loader, device, epoch, model_dir=model_dir)
                if pred_file:
                    mlflow.log_artifact(pred_file, artifact_path=f"predictions/epoch_{epoch+1:03d}")
                    logger.info(f"[MLFLOW] Fallback: Successfully logged prediction samples for epoch {epoch+1}")
                
                # Save training curves when we have enough data
                if len(epoch_history) >= 2:
                    # Use the existing model directory instead of creating a new one
                    
                    enhanced_curves_file = save_enhanced_training_curves(
                        epoch_history, model_dir, epoch
                    )
                    if enhanced_curves_file:
                        mlflow.log_artifact(enhanced_curves_file)
                    
                    comparison_artifacts = save_model_comparison_artifacts(
                        model_dir, 
                        {**metrics, 'epoch_history': epoch_history}, 
                        epoch
                    )
                    for artifact in comparison_artifacts:
                        mlflow.log_artifact(artifact)
                else:
                    # Fallback to original training curves
                    training_curves_file = save_training_curves(epoch, metrics, logger, model_dir=model_dir)
                    if training_curves_file:
                        mlflow.log_artifact(training_curves_file)
            
            # Dynamic Learning Rate Adjustment
            if 'lr_scheduler' in locals() and lr_scheduler:
                try:
                    # Provide current metrics to the scheduler
                    lr_adjustment = lr_scheduler.check_and_adjust(
                        epoch=epoch + 1,  # 1-based epoch numbering
                        metrics={
                            'val_dice': val_dice,
                            'val_loss': val_loss,
                            'train_dice': train_dice,
                            'train_loss': epoch_loss
                        }
                    )
                    
                    if lr_adjustment['adjusted']:
                        # Apply the new learning rate to the optimizer
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_adjustment['new_lr']
                        
                        logger.info(f"[LR_SCHEDULER] Learning rate adjusted from {lr_adjustment['old_lr']:.6f} to {lr_adjustment['new_lr']:.6f}")
                        logger.info(f"[LR_SCHEDULER] Reason: {lr_adjustment['reason']}")
                        
                        # Log the adjustment to MLflow
                        mlflow.log_metric('lr_adjustment_epoch', epoch + 1, step=epoch)
                        mlflow.log_metric('lr_adjustment_reason', hash(lr_adjustment['reason']), step=epoch)
                    
                except Exception as lr_error:
                    logger.warning(f"[LR_SCHEDULER] Error in learning rate adjustment: {lr_error}")
            
            # Track learning rate
            current_lr = optimizer.param_groups[0]['lr']
            mlflow.log_metric('learning_rate', current_lr, step=epoch)
            
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                
                # Use existing model directory instead of creating a new one
                model_name = f"model_{args.model_id if args.model_id else 'unknown'}_epoch_{epoch+1}_dice_{val_dice:.3f}"
                
                best_model_path = os.path.join(model_dir, "weights", "model.pth")
                torch.save(model.state_dict(), best_model_path)
                
                # Save model metadata
                metadata_path = save_enhanced_model_metadata(
                    model_dir=model_dir,
                    model_id=args.model_id,
                    unique_id=unique_id,
                    args=args,
                    model_info=model,
                    training_metrics=metrics,
                    model_family=model_family,
                    arch_info=arch_info
                )
                
                # Enhanced MLflow artifact logging for best model
                mlflow.log_artifact(best_model_path, artifact_path=f"checkpoints/best_model/epoch_{epoch+1:03d}")
                mlflow.log_artifact(metadata_path, artifact_path=f"checkpoints/best_model/metadata")
                
                # Log current epoch metrics as best model context
                best_model_context = {
                    'epoch': epoch +  1,
                    'validation_dice': val_dice,
                    'validation_loss': val_loss,
                    'train_dice': train_dice,
                    'train_loss': epoch_loss,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'improvement': val_dice - best_val_dice if best_val_dice != -1 else val_dice,
                    'model_path': best_model_path,
                    'timestamp': time.time()
                }
                
                context_file = os.path.join(model_dir, "weights", "best_model_context.json")
                with open(context_file, 'w') as f:
                    json.dump(best_model_context, f, indent=2, default=str)
                
                mlflow.log_artifact(context_file, artifact_path=f"checkpoints/best_model/context")
                
                logger.info(f"Saved new best model with val_dice: {val_dice:.4f} at {best_model_path}")
                logger.info(f"[MLFLOW] Best model artifacts logged to checkpoints/best_model/epoch_{epoch+1:03d}")
            
            # --- ZAPIS ŚCIEŻEK DO WIZUALIZACJI DLA GUI ---
            try:
                vis_json_path = os.path.join(model_dir, 'artifacts', 'visualizations.json')
                # Zbierz ścieżki do wizualizacji
                vis_data = {}
                # Wejścia i maski z treningu
                sample_vis_path = os.path.join(model_dir, 'artifacts', 'sample_inputs_and_masks.png')
                if os.path.exists(sample_vis_path):
                    vis_data['train_sample_inputs_and_masks'] = sample_vis_path
                # Wejścia i maski z walidacji
                val_vis_path = os.path.join(model_dir, 'artifacts', 'val_sample_inputs_and_masks.png')
                if os.path.exists(val_vis_path):
                    vis_data['val_sample_inputs_and_masks'] = val_vis_path
                # Predykcje z tej epoki
                pred_dir = os.path.join(model_dir, f'predictions/epoch_{epoch+1:03d}')
                if os.path.exists(pred_dir):
                    pred_files = [os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.endswith('.png')]
                    if pred_files:
                        vis_data['predictions'] = pred_files
                # Zapisz lub zaktualizuj plik JSON
                if os.path.exists(vis_json_path):
                    with open(vis_json_path, 'r') as f:
                        old_data = json.load(f)
                    old_data.update({f'epoch_{epoch+1}': vis_data})
                    with open(vis_json_path, 'w') as f:
                        json.dump(old_data, f, indent=2)
                else:
                    with open(vis_json_path, 'w') as f:
                        json.dump({f'epoch_{epoch+1}': vis_data}, f, indent=2)
                logger.info(f"[GUI] Zaktualizowano artifacts/visualizations.json dla GUI")
            except Exception as e:
                logger.warning(f"[GUI] Nie udało się zaktualizować visualizations.json: {e}")
            # ...existing code...

        logger.info(f"Training completed. Best validation Dice score: {best_val_dice:.4f}")
        
        # Enhanced final model logging using the new artifact manager
        try:
            from ml.utils.utils.mlflow_artifact_manager import log_final_model
            
            # Prepare comprehensive model information
            model_info = {
                'architecture': getattr(arch_info, 'display_name', 'MONAI UNet') if arch_info else 'MONAI UNet',
                'framework': getattr(arch_info, 'framework', 'PyTorch') if arch_info else 'PyTorch',
                'model_family': getattr(args, 'model_family', 'UNet-Coronary'),
                'architecture_key': getattr(arch_info, 'key', 'monai_unet') if arch_info else 'monai_unet',
                'version': getattr(arch_info, 'version', '1.0.0') if arch_info else '1.0.0',
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
                'training_config': {
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'optimizer': 'Adam',
                    'loss_function': 'DiceLoss',
                    'device': str(device),
                    'validation_split': args.validation_split
                },
                'data_info': {
                    'total_training_batches': len(train_loader),
                    'total_validation_batches': len(val_loader),
                    'crop_size': args.crop_size,
                    'augmentations': {
                        'random_flip': getattr(args, 'random_flip', False),
                        'random_rotate': getattr(args, 'random_rotate', False),
                        'random_scale': getattr(args, 'random_scale', False),
                        'random_intensity': getattr(args, 'random_intensity', False)
                    }
                }
            }
            
            # Best metrics summary
            best_metrics = {
                'best_val_dice': best_val_dice,
                'final_train_loss': epoch_history[-1]['train_loss'] if epoch_history else 0.0,
                'final_val_loss': epoch_history[-1]['val_loss'] if epoch_history else 0.0,
                'final_train_dice': epoch_history[-1]['train_dice'] if epoch_history else 0.0,
                'final_val_dice': epoch_history[-1]['val_dice'] if epoch_history else 0.0,
                'total_epochs_trained': len(epoch_history),
                'convergence_epoch': next((i+1 for i, h in enumerate(epoch_history) if h['val_dice'] == best_val_dice), len(epoch_history))
            }
            
            # Use the model directory that was created during training
            final_model_dir = model_dir if 'model_dir' in locals() else None
            
            if final_model_dir:
                # Log using enhanced artifact manager
                final_logged_paths = log_final_model(
                    model_info=model_info,
                    model_directory=final_model_dir,
                    best_metrics=best_metrics
                )
                
                # Additional comprehensive artifact logging using MLflow APIs
                logger.info("[MLFLOW] Logging additional comprehensive artifacts...")
                
                # Log final training history as JSON
                history_file = os.path.join(final_model_dir, "training_history.json")
                with open(history_file, 'w') as f:
                    json.dump({
                        'epoch_history': epoch_history,
                        'best_metrics': best_metrics,
                        'training_summary': {
                            'total_epochs': len(epoch_history),
                            'best_epoch': next((i+1 for i, h in enumerate(epoch_history) if h['val_dice'] == best_val_dice), len(epoch_history)),
                            'final_lr': optimizer.param_groups[0]['lr'],
                            'model_parameters': sum(p.numel() for p in model.parameters())
                        }
                    }, f, indent=2)
                mlflow.log_artifact(history_file, artifact_path="final_model/training_history")
                
                # Log final model state with comprehensive metadata
                if best_model_path and os.path.exists(best_model_path):
                    mlflow.log_artifact(best_model_path, artifact_path="final_model/weights")
                
                # Log complete training configuration
                final_config = {
                    'model_info': model_info,
                    'training_args': vars(args),
                    'final_metrics': best_metrics,
                    'device_info': {
                        'device': str(device),
                        'cuda_available': torch.cuda.is_available(),
                        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
                    }
                }
                config_file = os.path.join(final_model_dir, "complete_config.json")
                with open(config_file, 'w') as f:
                    json.dump(final_config, f, indent=2, default=str)
                mlflow.log_artifact(config_file, artifact_path="final_model/configuration")
                
                # Log PyTorch model using MLflow's model logging - with safe signature handling
                try:
                    if sample_batch is not None and "image" in sample_batch and sample_images is not None:
                        input_example = sample_images[:1].detach().cpu().numpy()
                        try:
                            # Try to create signature with model inference
                            signature = mlflow.models.infer_signature(
                                input_example,
                                torch.sigmoid(model(sample_images[:1].to(device))).detach().cpu().numpy()
                            )
                            mlflow.pytorch.log_model(
                                model,
                                "pytorch_model",
                                input_example=input_example,
                                signature=signature
                            )
                            logger.info("[MLFLOW] PyTorch model logged with signature and input example")
                        except Exception as sig_error:
                            logger.warning(f"[MLFLOW] Failed to create model signature: {sig_error}")
                            # Fallback without signature but with input example
                            mlflow.pytorch.log_model(
                                model,
                                "pytorch_model",
                                input_example=input_example
                            )
                            logger.info("[MLFLOW] PyTorch model logged with input example only (no signature)")
                    else:
                        logger.warning("[MLFLOW] No valid sample batch available for model signature")
                        # Final fallback without any examples
                        mlflow.pytorch.log_model(model, "pytorch_model_fallback")
                        logger.info("[MLFLOW] PyTorch model logged without signature or input example")
                except Exception as model_log_error:
                    logger.warning(f"[MLFLOW] Failed to log PyTorch model: {model_log_error}")
                    # Fallback without signature
                    mlflow.pytorch.log_model(model, "pytorch_model_fallback")
                    logger.info("[MLFLOW] PyTorch model logged with fallback method")
                
                logger.info(f"[MLFLOW] Enhanced final model logging completed: {len(final_logged_paths)} artifacts")
            else:
                logger.warning("[MLFLOW] No model directory available for enhanced logging, using fallback")
                raise Exception("No model directory for enhanced logging")
                
        except Exception as e:
            logger.warning(f"[MLFLOW] Enhanced final model logging failed, using fallback: {e}")
            
            # Fallback to original final model logging - with safe sample batch handling
            try:
                if sample_batch is not None and "image" in sample_batch and sample_images is not None:
                    input_example = sample_images[:1].detach().cpu().numpy()
                    try:
                        # Try with signature first
                        signature = mlflow.models.infer_signature(
                            input_example,
                            torch.sigmoid(model(sample_images[:1].to(device))).detach().cpu().numpy()
                        )
                        mlflow.pytorch.log_model(
                            model,
                            "model",
                            input_example=input_example,
                            signature=signature
                        )
                        logger.info("[MLFLOW] Model logged to MLflow under 'model' artifact (fallback with signature).")
                    except Exception as fallback_signature_error:
                        logger.warning(f"[MLFLOW] Failed to log fallback model with signature: {fallback_signature_error}")
                        # Final fallback with input example but no signature
                        mlflow.pytorch.log_model(
                            model,
                            "model",
                            input_example=input_example,
                        )
                        logger.info("[MLFLOW] Model logged to MLflow under 'model' artifact (fallback without signature).")
                else:
                    logger.warning("[MLFLOW] No valid sample batch for fallback model logging")
                    # Ultimate fallback without any examples
                    mlflow.pytorch.log_model(model, "model")
                    logger.info("[MLFLOW] Model logged to MLflow under 'model' artifact (no examples).")
            except Exception as fallback_error:
                logger.error(f"[MLFLOW] All fallback model logging attempts failed: {fallback_error}")
                # Final emergency fallback
                try:
                    mlflow.pytorch.log_model(model, "model_emergency")
                    logger.info("[MLFLOW] Emergency model logging successful")
                except Exception as emergency_error:
                    logger.error(f"[MLFLOW] Emergency model logging also failed: {emergency_error}")
        
        # Enhanced training artifacts and summary logging
        try:
            # Log training logs with comprehensive organization
            from ml.utils.utils.mlflow_artifact_manager import MLflowArtifactManager
            
            with MLflowArtifactManager() as artifact_manager:
                # Collect all training logs
                log_files = [
                    os.path.join('data', 'models', 'artifacts', 'training.log'),
                    os.path.join(model_dir, 'logs', 'training.log') if 'model_dir' in locals() else None
                ]
                log_files = [f for f in log_files if f and os.path.exists(f)]
                
                if log_files:
                    logged_log_paths = artifact_manager.log_training_logs(log_files, "training")
                    logger.info(f"[MLFLOW] Logged {len(logged_log_paths)} training log files")
                    
                    # Also log individual log files with organized paths
                    for log_file in log_files:
                        mlflow.log_artifact(log_file, artifact_path="logs/training")
                
                # Log complete training summary
                training_summary = {
                    'training_completed': True,
                    'total_epochs': len(epoch_history),
                    'best_validation_dice': best_val_dice,
                    'final_model_path': best_model_path,
                    'model_directory': model_dir if 'model_dir' in locals() else None,
                    'training_duration': time.time() - training_start_time if 'training_start_time' in locals() else None,
                    'device_used': str(device),
                    'dataset_info': {
                        'training_samples': len(train_ds) if 'train_ds' in locals() else train_samples,
                        'validation_samples': len(val_ds) if 'val_ds' in locals() else val_samples,
                        'total_batches_per_epoch': len(train_loader),
                        'validation_batches_per_epoch': len(val_loader)
                    }
                }
                
                summary_file = os.path.join('data', 'models', 'artifacts', 'training_summary.json')
                os.makedirs(os.path.dirname(summary_file), exist_ok=True)
                with open(summary_file, 'w') as f:
                    json.dump(training_summary, f, indent=2, default=str)
                
                mlflow.log_artifact(summary_file, artifact_path="summaries/training")
                logger.info("[MLFLOW] Comprehensive training summary logged")
                
        except Exception as e:
            logger.warning(f"[MLFLOW] Failed to log training logs with enhanced manager: {e}")
            
            # Fallback: Log training logs using basic MLflow artifact APIs
            try:
                # Log available training logs
                fallback_logs = [
                    'artifacts/training_logs/training.log',
                    'data/logs/training.log'
                ]
                
                for log_path in fallback_logs:
                    if os.path.exists(log_path):
                        mlflow.log_artifact(log_path, artifact_path="logs/training_fallback")
                        logger.info(f"[MLFLOW] Fallback logged: {log_path}")
                        
                # Create and log basic training summary
                basic_summary = {
                    'training_completed': True,
                    'epochs': len(epoch_history),
                    'best_dice': best_val_dice,
                    'device': str(device)
                }
                
                fallback_summary_file = 'data/artifacts/basic_training_summary.json'
                os.makedirs(os.path.dirname(fallback_summary_file), exist_ok=True)
                with open(fallback_summary_file, 'w') as f:
                    json.dump(basic_summary, f, indent=2)
                
                mlflow.log_artifact(fallback_summary_file, artifact_path="summaries/basic")
                        
            except Exception as fallback_e:
                logger.warning(f"[MLFLOW] Fallback log artifact also failed: {fallback_e}")
        
        # Log interactive training plot if available
        try:
            interactive_plot_file = save_interactive_training_plot(epoch_history, model_dir)
            if interactive_plot_file:
                mlflow.log_artifact(interactive_plot_file)
        except Exception as e:
            logger.warning(f"[MLFLOW] Failed to log interactive plot: {e}")
        if hasattr(args, 'model_id') and args.model_id is not None:
            try:
                # Import registry functions using absolute import
                import sys
                current_script_dir = os.path.dirname(os.path.abspath(__file__))
                core_dir = os.path.abspath(os.path.join(current_script_dir, '..', '..', 'core', 'apps'))
                if core_dir not in sys.path:
                    sys.path.append(core_dir)
                from ml_manager.utils.mlflow_utils import register_model, transition_model_stage
                
                # Determine model family/type for registry naming and tags
                model_family = getattr(args, 'model_family', None) or getattr(args, 'model_type', None) or 'generic-model'
                registry_model_name = f"{model_family}-v{args.model_id}"
                registry_tags = {
                    "model_family": model_family,
                    "framework": "PyTorch",
                    "task": "image_segmentation",
                    "best_val_dice": str(best_val_dice),
                    "training_epochs": str(args.epochs),
                    "batch_size": str(args.batch_size),
                    "learning_rate": str(args.learning_rate)
                }
                # Register the model
                model_info = register_model(
                    run_id=args.mlflow_run_id,
                    model_name=registry_model_name,
                    model_description=f"{model_family} model for coronary segmentation trained with {args.epochs} epochs, best val dice: {best_val_dice:.4f}",
                    tags=registry_tags
                )
                
                if model_info:
                    logger.info(f"[REGISTRY] Model registered: {model_info['name']} v{model_info['version']}")
                    
                    # Update Django model with registry information
                    if callback:
                        callback.update_registry_info(
                            registry_model_name=model_info['name'],
                            registry_model_version=model_info['version'],
                            is_registered=True
                        )
                    
                    # Auto-promote to Staging if performance is good
                    if best_val_dice > 0.8:  # Threshold for auto-promotion
                        success = transition_model_stage(
                            model_name=model_info['name'],
                            version=model_info['version'],
                            stage="Staging"
                        )
                        if success:
                            logger.info(f"[REGISTRY] Model auto-promoted to Staging (val_dice: {best_val_dice:.4f})")
                            if callback:
                                callback.update_registry_stage("Staging")
                        else:
                            logger.warning("[REGISTRY] Failed to auto-promote model to Staging")
                    
                else:
                    logger.warning("[REGISTRY] Failed to register model")
                    
            except Exception as e:
                logger.warning(f"[REGISTRY] Model registration failed: {e}")
        
        if callback:
            callback.on_training_end({"best_val_dice": best_val_dice})
        
        # === COMPREHENSIVE TRAINING SUMMARY WITH DATA RANGE ANALYSIS ===
        logger.info("=" * 80)
        logger.info("🏁 TRAINING COMPLETED - COMPREHENSIVE SUMMARY REPORT")
        logger.info("=" * 80)
        
        # Training Performance Summary
        logger.info(f"📊 TRAINING PERFORMANCE:")
        logger.info(f"   📈 Best Validation Dice Score: {best_val_dice:.4f}")
        logger.info(f"   🔄 Total Epochs Completed: {len(epoch_history)}")
        logger.info(f"   ⏱️  Training Duration: {time.time() - training_start_time:.1f} seconds" if 'training_start_time' in locals() else "   ⏱️  Training Duration: Not tracked")
        logger.info(f"   🖥️  Device Used: {device}")
        
        # Dataset Summary  
        logger.info(f"📁 DATASET SUMMARY:")
        if 'train_ds' in locals() and 'val_ds' in locals():
            logger.info(f"   🏋️  Training Samples: {len(train_ds)}")
            logger.info(f"   ✅ Validation Samples: {len(val_ds)}")
            logger.info(f"   📦 Total Dataset Size: {len(train_ds) + len(val_ds)}")
        elif 'train_samples' in locals() and 'val_samples' in locals():
            logger.info(f"   🏋️  Training Samples: {train_samples}")
            logger.info(f"   ✅ Validation Samples: {val_samples}")  
            logger.info(f"   📦 Total Dataset Size: {train_samples + val_samples}")
        else:
            logger.info(f"   ❓ Dataset size information not available")
        
        # Batch Processing Summary
        logger.info(f"   🔢 Batches per Epoch: {len(train_loader)} train, {len(val_loader)} validation")
        logger.info(f"   📏 Batch Size: {args.batch_size}")
        logger.info(f"   👷 Workers: {getattr(args, 'num_workers', 'Unknown')}")
        
        # Model Architecture Summary
        if 'total_params' in locals():
            logger.info(f"🏗️  MODEL ARCHITECTURE:")
            logger.info(f"   🔧 Parameters: {total_params:,} total ({trainable_params:,} trainable)")
            logger.info(f"   💾 Model Size: {total_params * 4 / (1024**2):.2f} MB")
        
        # Training Configuration Summary
        logger.info(f"⚙️  TRAINING CONFIGURATION:")
        logger.info(f"   📚 Learning Rate: {args.learning_rate}")
        logger.info(f"   🎯 Loss Function: DiceLoss with Sigmoid")
        logger.info(f"   🔄 Optimizer: {getattr(args, 'optimizer', 'adam').upper()}")
        logger.info(f"   📐 Resolution: {getattr(args, 'resolution', '256')}px")
        logger.info(f"   ✂️  Crop Size: {getattr(args, 'crop_size', '128')}px")
        
        # Data Range Analysis Summary (if data was analyzed)
        logger.info(f"🔍 DATA CHARACTERISTICS SUMMARY:")
        logger.info(f"   💽 Dataset Type: {getattr(args, 'dataset_type', 'auto-detected')}")
        logger.info(f"   📊 Data Path: {args.data_path}")
        
        # MLflow Integration Summary
        if hasattr(args, 'mlflow_run_id') and args.mlflow_run_id:
            logger.info(f"📈 MLFLOW INTEGRATION:")
            logger.info(f"   🆔 Run ID: {args.mlflow_run_id}")
            logger.info(f"   📝 Experiment: {mlflow.get_experiment(mlflow.active_run().info.experiment_id).name if mlflow.active_run() else 'N/A'}")
        
        # File Artifacts Summary
        if 'model_dir' in locals() and model_dir:
            logger.info(f"📁 GENERATED ARTIFACTS:")
            logger.info(f"   📂 Model Directory: {model_dir}")
            if 'best_model_path' in locals():
                logger.info(f"   🏆 Best Model: {os.path.basename(best_model_path)}")
            
            # List key artifact files
            artifacts_dir = os.path.join(model_dir, 'artifacts')
            if os.path.exists(artifacts_dir):
                artifact_files = [f for f in os.listdir(artifacts_dir) if f.endswith(('.png', '.json', '.txt', '.html'))]
                if artifact_files:
                    logger.info(f"   📊 Artifacts Generated: {len(artifact_files)} files")
                    for artifact in sorted(artifact_files)[:5]:  # Show first 5
                        logger.info(f"     • {artifact}")
                    if len(artifact_files) > 5:
                        logger.info(f"     • ... and {len(artifact_files) - 5} more")
                    
                    # Enhanced MLflow artifact logging with retry mechanism
                    try:
                        from ml.utils.mlflow_artifact_logger import force_log_all_artifacts
                        logged_artifacts = force_log_all_artifacts(model_dir, args.mlflow_run_id)
                        if logged_artifacts:
                            logger.info(f"   ✅ MLflow: {len(logged_artifacts)} artifacts logged successfully")
                            logger.info(f"   📊 Artifacts available in MLflow UI at experiment artifacts section")
                        else:
                            logger.warning(f"   ⚠️  No artifacts were logged to MLflow")
                    except Exception as mlflow_error:
                        logger.error(f"   ❌ MLflow artifact logging failed: {mlflow_error}")
                        logger.info(f"   💡 Check MLflow connection and experiment configuration")
                        # Continue execution even if MLflow logging fails
        
        # Performance Recommendations
        logger.info(f"💡 PERFORMANCE INSIGHTS:")
        if best_val_dice < 0.3:
            logger.info(f"   ⚠️  Low Dice Score detected - Consider:")
            logger.info(f"     • Increasing training epochs")
            logger.info(f"     • Adjusting learning rate")
            logger.info(f"     • Verifying data quality and ranges")
            logger.info(f"     • Checking mask normalization (0-1 vs 0-255)")
        elif best_val_dice < 0.7:
            logger.info(f"   🔄 Moderate performance - Potential improvements:")
            logger.info(f"     • Fine-tuning hyperparameters")
            logger.info(f"     • Adding data augmentation")
            logger.info(f"     • Increasing model complexity")
        else:
            logger.info(f"   ✅ Excellent performance achieved!")
            logger.info(f"   🚀 Model ready for production consideration")
        
        # Next Steps
        logger.info(f"🎯 NEXT STEPS:")
        logger.info(f"   • Review training curves and sample predictions")
        logger.info(f"   • Test model on independent test set")
        logger.info(f"   • Consider model deployment if performance is satisfactory")
        if hasattr(args, 'model_id') and args.model_id is not None:
            logger.info(f"   • Check MLflow for detailed metrics and artifacts")
        
        logger.info("=" * 80)
        logger.info("🎉 TRAINING SUMMARY COMPLETE")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        if callback:
            callback.on_training_failed(str(e))
        raise
    finally:
        # Stop system monitoring
        if system_monitor:
            try:
                system_monitor.stop_monitoring()
                logger.info("[MONITORING] System monitoring stopped")
            except Exception as e:
                logger.warning(f"[MONITORING] Failed to stop system monitoring: {e}")
        
        # Only call mlflow.end_run() at the very end, after all logging is complete
        if mlflow.active_run():
            mlflow.end_run()

# Placeholder for save_interactive_training_plot
def save_interactive_training_plot(epoch_history, model_dir):
    logger.info("[PLOT] `save_interactive_training_plot` called, placeholder implementation.")
    # Example: Create a dummy file to satisfy logging if needed
    # dummy_plot_path = os.path.join(model_dir, "interactive_plot_placeholder.html")
    # with open(dummy_plot_path, "w") as f:
    #     f.write("<html><body>Placeholder for interactive plot</body></html>")
    # return dummy_plot_path
    return None

def get_inference_transforms(image_size=(256, 256), use_original_size=False):
    """Get transforms for inference - uses PIL-compatible orientation
    
    Args:
        image_size: Target image size as (height, width). Ignored if use_original_size=True
        use_original_size: If True, don't resize the image, keep original dimensions
    """
    def load_pil_compatible_image(filepath):
        """Load image using PIL to maintain browser-compatible orientation"""
        from PIL import Image
        import numpy as np
        import torch
        
        img = Image.open(filepath)
        
        # Keep RGB format to match training (3 channels)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array and add channel dimension
        img_array = np.array(img)
        if len(img_array.shape) == 3:
            # Convert HWC to CHW format
            img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
        
        # Convert to torch tensor and normalize to [0, 1]
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        
        # Apply RGB normalization to match training
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor
    
    if use_original_size:
        # Don't resize, just load with RGB normalization
        return Compose([
            Lambda(load_pil_compatible_image),
        ])
    else:
        return Compose([
            Lambda(load_pil_compatible_image),
            Resize(spatial_size=image_size, mode="bilinear"),
        ])

def get_display_oriented_image(input_file, target_size=(256, 256)):
    """Load image in display orientation (same as browser shows) for comparison"""
    # Load image using PIL to maintain browser-compatible orientation
    img = Image.open(input_file)
    
    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
    
    # Resize to target size
    img = img.resize(target_size, Image.Resampling.BILINEAR)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    return img_array

def run_inference(model_path, input_path, output_dir, device="cuda", weights_path=None, model_type="unet", resolution="original"):
    """Run inference on input images using a trained model
    
    Args:
        model_path: Path to the trained model weights (MLflow or .pth)
        input_path: Path to input image or directory of images
        output_dir: Directory to save predictions
        device: Device to run inference on
        weights_path: Optional path to a .pth file to load weights from
        model_type: Type of model architecture to use
        resolution: Target resolution for input images ('original', '128', '256', '384', '512')
    """
    logger = logging.getLogger(__name__)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model using architecture registry - use 3 channels for RGB input (matching training)
    model_config = get_default_model_config(model_type)
    model_config["in_channels"] = 3  # Force 3 channels for RGB input to match training
    model, arch_info = create_model_from_registry(model_type, device, **model_config)
    
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get transforms based on resolution choice
    if resolution == "original":
        transforms = get_inference_transforms(use_original_size=True)
    else:
        # Convert string resolution to integer tuple
        res_int = int(resolution)
        transforms = get_inference_transforms(image_size=(res_int, res_int), use_original_size=False)
    
    if os.path.isdir(input_path):
        input_files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    else:
        input_files = [input_path]
    logger.info(f"Processing {len(input_files)} files...")
    with torch.no_grad():
        for input_file in input_files:
            # Load and transform image
            img = transforms(input_file)
            img = img.unsqueeze(0).to(device)  # Add batch dimension
            
            # Generate prediction
            output = model(img)
            
            # Apply appropriate post-processing based on number of output channels
            num_output_channels = output.shape[1]
            if num_output_channels == 1:
                # Binary segmentation
                output = torch.sigmoid(output)
                pred = (output > 0.5).float()
                logger.info(f"Applied binary segmentation post-processing (sigmoid + threshold)")
            else:
                # Multi-class semantic segmentation
                output = torch.softmax(output, dim=1)
                pred = torch.argmax(output, dim=1, keepdim=True).float()
                logger.info(f"Applied multi-class segmentation post-processing (softmax + argmax) for {num_output_channels} classes")
            
            # Save prediction - ensure correct tensor dimensions
            pred_np = pred.squeeze().cpu().numpy()
            # Ensure 2D array (height, width) for proper image creation
            if pred_np.ndim == 3 and pred_np.shape[0] == 1:
                pred_np = pred_np.squeeze(0)
            
            # Debug: Log prediction statistics
            logger.info(f"Prediction stats - Shape: {pred_np.shape}, Min: {pred_np.min():.3f}, Max: {pred_np.max():.3f}, Mean: {pred_np.mean():.3f}")
            
            # Convert to visible image - if prediction is all zeros, create a test pattern
            if pred_np.max() == 0:
                logger.warning("Prediction is all zeros - no segmentation detected")
                # Create a semi-transparent overlay to show the model ran
                pred_image = np.zeros_like(pred_np, dtype=np.uint8)
                # Add a small indicator that inference ran but found no segments
                pred_image[10:30, 10:30] = 128  # Small gray square as indicator
            else:
                pred_image = (pred_np * 255).astype(np.uint8)
                logger.info(f"Segmentation detected - {np.sum(pred_np > 0)} pixels")
            output_filename = os.path.join(output_dir, f"pred_{os.path.basename(input_file)}")
            
            # Create a side-by-side comparison with consistent orientation
            # Both model input and display now use the same PIL orientation
            display_input = get_display_oriented_image(input_file, (pred_image.shape[1], pred_image.shape[0]))
            
            # Create side-by-side comparison
            comparison = np.hstack([display_input, pred_image])
            comparison_img = Image.fromarray(comparison)
            comparison_img.save(output_filename)
            
            # Save the display-oriented input separately for web display consistency
            input_only_filename = os.path.join(output_dir, f"input_{os.path.basename(input_file)}")
            input_only_img = Image.fromarray(display_input)
            input_only_img.save(input_only_filename)
            
            # Save prediction only
            pred_only_filename = os.path.join(output_dir, f"pred_only_{os.path.basename(input_file)}")
            pred_only_img = Image.fromarray(pred_image)
            pred_only_img.save(pred_only_filename)
            
            logger.info(f"Saved prediction to {output_filename}")
            
            # Enhanced MLflow artifact logging for predictions
            try:
                if mlflow.active_run():
                    # Log all prediction outputs with organized structure
                    mlflow.log_artifact(output_filename, artifact_path="predictions/comparisons")
                    mlflow.log_artifact(input_only_filename, artifact_path="predictions/inputs")
                    mlflow.log_artifact(pred_only_filename, artifact_path="predictions/outputs")
                    
                    # Create and log prediction metadata
                    prediction_metadata = {
                        'input_file': os.path.basename(input_file),
                        'prediction_timestamp': time.time(),
                        'device_used': str(device),
                        'model_type': model_type,
                        'resolution': resolution,
                        'input_shape': img.shape,
                        'output_shape': pred.shape,
                        'prediction_threshold': 0.5,
                        'files_generated': {
                            'comparison': os.path.basename(output_filename),
                            'input_only': os.path.basename(input_only_filename),
                            'prediction_only': os.path.basename(pred_only_filename)
                        }
                    }
                    
                    # Save metadata file
                    metadata_filename = os.path.join(output_dir, f"metadata_{os.path.splitext(os.path.basename(input_file))[0]}.json")
                    with open(metadata_filename, 'w') as f:
                        json.dump(prediction_metadata, f, indent=2, default=str)
                    
                    mlflow.log_artifact(metadata_filename, artifact_path="predictions/metadata")
                    
                    logger.info(f"[MLFLOW] Logged prediction artifacts for {os.path.basename(input_file)}")
                    
            except Exception as e:
                logger.warning(f"Failed to log prediction to MLflow: {e}")

def inference_mode(args):
    """Run the model in inference mode with enhanced MLflow logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    
    inference_start_time = time.time()
    
    if args.mlflow_run_id:
        mlflow.start_run(run_id=args.mlflow_run_id)
        
        # Log inference parameters
        mlflow.log_param("inference_mode", True)
        mlflow.log_param("input_path", args.input_path)
        mlflow.log_param("output_dir", args.output_dir)
        mlflow.log_param("device", args.device)
        mlflow.log_param("model_type", getattr(args, 'model_type', 'unet'))
        mlflow.log_param("resolution", getattr(args, 'resolution', 'original'))
        
        # Set inference tags
        mlflow.set_tag("task", "inference")
        mlflow.set_tag("mode", "prediction")
        
    try:
        run_inference(
            model_path=args.model_path,
            input_path=args.input_path,
            output_dir=args.output_dir,
            device=args.device,
            weights_path=getattr(args, 'weights_path', None),
            model_type=getattr(args, 'model_type', 'unet'),
            resolution=getattr(args, 'resolution', 'original')
        )
        
        # Log inference summary if MLflow run is active
        if args.mlflow_run_id and mlflow.active_run():
            inference_duration = time.time() - inference_start_time
            
            # Count input files
            if os.path.isdir(args.input_path):
                input_count = len([f for f in os.listdir(args.input_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))])
            else:
                input_count = 1
            
            # Count output files
            output_count = len([f for f in os.listdir(args.output_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(args.output_dir) else 0
            
            mlflow.log_metric("inference_duration_seconds", inference_duration)
            mlflow.log_metric("input_files_processed", input_count)
            mlflow.log_metric("output_files_generated", output_count)
            mlflow.log_metric("processing_rate_files_per_second", input_count / inference_duration if inference_duration > 0 else 0)
            
            # Create and log inference summary
            inference_summary = {
                'inference_completed': True,
                'duration_seconds': inference_duration,
                'input_files_processed': input_count,
                'output_files_generated': output_count,
                'processing_rate': input_count / inference_duration if inference_duration > 0 else 0,
                'device_used': args.device,
                'model_type': getattr(args, 'model_type', 'unet'),
                'resolution': getattr(args, 'resolution', 'original')
            }
            
            summary_file = os.path.join(args.output_dir, 'inference_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(inference_summary, f, indent=2, default=str)
            
            if args.mlflow_run_id:
                mlflow.log_artifact(summary_file, artifact_path="inference/summary")
            
    finally:
        if args.mlflow_run_id:
            mlflow.end_run()

def main():
    """Main entry point"""
    args = parse_args()
    
    if args.mode == 'train':
        train_model(args)
    else:  # predict mode
        inference_mode(args)

# --- Exception handling for main ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.exception("Training failed with exception")
        print(f"[Train.py] Exception: {e}")
        raise
