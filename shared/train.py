import os
import logging
# --- Robust logging setup at the very top ---
os.makedirs('models/artifacts', exist_ok=True)
logging.basicConfig(
    filename='models/artifacts/training.log',
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
import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import uuid
from datetime import datetime
from monai.networks.nets import UNet as MonaiUNet
from monai.data import CacheDataset, DataLoader as MonaiDataLoader
from monai.data import Dataset, DataLoader
from monai.transforms import (
    LoadImage, ScaleIntensity, ToTensor, Compose,
    Resize, EnsureChannelFirst, ConvertToMultiChannelBasedOnBratsClassesd,
    Lambda
)
# Import architecture registry
from shared.architecture_registry import registry as architecture_registry
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
        "models",
        "organized", 
        date_str,
        family_str,
        f"{unique_id}_v{version}"
    )
    
    # Create subdirectories for different artifacts
    subdirs = ["weights", "config", "artifacts", "predictions", "metrics", "logs"]
    for subdir in subdirs:
        os.makedirs(os.path.join(model_dir, subdir), exist_ok=True)
    
    return model_dir, unique_id

def save_enhanced_training_curves(epoch_history, model_dir, epoch):
    """Save comprehensive training curves with multiple metrics"""
    try:
        if len(epoch_history) < 2:
            return None
            
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
        from .architecture_registry import architecture_registry
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
        ScaleIntensityd(keys=["label"]),  # Ensure labels are scaled to [0,1]
    ]
    
    if for_training:
        # Add training-specific augmentations
        if params.get('use_random_scale', True):
            # Use RandScaleIntensityd for random intensity scaling
            transforms.append(RandScaleIntensityd(keys=["image"], factors=(0.8, 1.2), prob=0.5))
        
        crop_size = params.get('crop_size', 128)
        # Use RandCropByPosNegLabeld for better segmentation cropping
        transforms.append(RandCropByPosNegLabeld(
            keys=["image", "label"], 
            label_key="label",
            spatial_size=[crop_size, crop_size],
            pos=1,
            neg=1,
            num_samples=4,
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
    """Save sample predictions from validation set with full image correspondence"""
    logger.info(f"[PREDICTIONS] Starting to save sample predictions for epoch {epoch+1}")
    model.eval()
    
    # Ensure we have a clean matplotlib state
    plt.close('all')
    
    with torch.no_grad():
        try:
            val_batch = next(iter(val_loader))
            images = val_batch["image"].to(device)
            labels = val_batch["label"].to(device)
            logger.info(f"[PREDICTIONS] Processing batch with {images.shape[0]} samples, image shape: {images.shape}")
            
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()
            
            # Create figure with proper settings
            fig, axes = plt.subplots(3, 4, figsize=(15, 10))
            plt.suptitle(f'Sample Predictions - Epoch {epoch+1} (Full Images)', fontsize=16)
            
            for i in range(min(4, images.shape[0])):
                # Input image
                axes[0, i].imshow(images[i, 0].cpu().numpy(), cmap='gray')
                axes[0, i].set_title(f'Input #{i+1}')
                axes[0, i].axis('off')
                
                # Ground truth
                axes[1, i].imshow(labels[i, 0].cpu().numpy(), cmap='gray')
                axes[1, i].set_title(f'Ground Truth #{i+1}')
                axes[1, i].axis('off')
                
                # Prediction
                axes[2, i].imshow(outputs[i, 0].cpu().numpy(), cmap='gray')
                axes[2, i].set_title(f'Prediction #{i+1}')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            
            # Determine output path - ensure predictions directory exists
            if model_dir:
                pred_dir = os.path.join(model_dir, f'predictions/epoch_{epoch+1:03d}')
                os.makedirs(pred_dir, exist_ok=True)
                filename = os.path.join(pred_dir, f'predictions_epoch_{epoch+1:03d}.png')
            else:
                # Fallback directory
                fallback_dir = os.path.join('models', 'artifacts', 'predictions')
                os.makedirs(fallback_dir, exist_ok=True)
                filename = os.path.join(fallback_dir, f'predictions_epoch_{epoch+1:03d}.png')
            
            # Save with high quality and ensure file is written
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            # Verify file was created and has content
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
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
                ax.text(0.5, 0.5, f'Epoch {epoch+1}\nPrediction generation failed\nError: {str(e)[:100]}...', 
                       ha='center', va='center', fontsize=12, transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                
                if model_dir:
                    pred_dir = os.path.join(model_dir, f'predictions/epoch_{epoch+1:03d}')
                    os.makedirs(pred_dir, exist_ok=True)
                    fallback_filename = os.path.join(pred_dir, f'predictions_epoch_{epoch+1:03d}_error.png')
                else:
                    fallback_dir = os.path.join('models', 'artifacts', 'predictions')
                    os.makedirs(fallback_dir, exist_ok=True)
                    fallback_filename = os.path.join(fallback_dir, f'predictions_epoch_{epoch+1:03d}_error.png')
                
                plt.savefig(fallback_filename, dpi=100, bbox_inches='tight')
                plt.close()
                logger.info(f"[PREDICTIONS] Created error fallback image: {fallback_filename}")
                return fallback_filename
            except:
                logger.error(f"[PREDICTIONS] Failed to create even fallback image")
                return None

def save_config(args, model_dir=None):
    """Save training configuration"""
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
    
    # Create separate transforms for training and validation
    train_transforms = get_monai_transforms(transform_params, for_training=True)
    val_transforms = get_monai_transforms(transform_params, for_training=False)
    
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
    return train_ds, val_ds

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
    parser.add_argument('--data-path', type=str, help='Path to dataset')
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
    global_log_path = os.path.join('models', 'artifacts', 'training.log')
    
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
        ml_manager_dir = os.path.abspath(os.path.join(current_script_dir, '..', 'ml_manager'))
        if ml_manager_dir not in sys.path:
            sys.path.append(ml_manager_dir)
        from mlflow_utils import setup_mlflow
        setup_mlflow()
        logger.info("[MLFLOW] MLflow experiment setup completed")
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
        from shared.utils.system_monitor import SystemMonitor
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
        from shared.utils.training_callback import TrainingCallback
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
        
        # Log dataset path verification
        logger.info(f"[DATASET] Data path: {args.data_path}")
        if os.path.exists(args.data_path):
            logger.info(f"[DATASET] Data path exists")
            images_path = os.path.join(args.data_path, 'imgs')
            masks_path = os.path.join(args.data_path, 'masks')
            if os.path.exists(images_path):
                img_count = len(os.listdir(images_path))
                logger.info(f"[DATASET] Found {img_count} images in {images_path}")
            else:
                logger.error(f"[DATASET] Images path not found: {images_path}")
            if os.path.exists(masks_path):
                mask_count = len(os.listdir(masks_path))
                logger.info(f"[DATASET] Found {mask_count} masks in {masks_path}")
            else:
                logger.error(f"[DATASET] Masks path not found: {masks_path}")
        else:
            logger.error(f"[DATASET] Data path does not exist: {args.data_path}")

        # Get transforms with augmentation parameters
        transform_params = {
            'use_random_flip': getattr(args, 'random_flip', True),
            'use_random_rotate': getattr(args, 'random_rotate', True),
            'use_random_scale': getattr(args, 'random_scale', True),
            'use_random_intensity': getattr(args, 'random_intensity', True),
            'crop_size': getattr(args, 'crop_size', 128)
        }
        
        logger.info("[DATASET] Loading datasets...")
        train_ds, val_ds = get_monai_datasets(args.data_path, args.validation_split, transform_params)
        # Use fewer workers to avoid shared memory issues in Docker
        num_workers = min(getattr(args, 'num_workers', 2), 2)
        train_loader = MonaiDataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
        val_loader = MonaiDataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

        # Log dataset info
        logger.info(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
        
        # Dataset loaded successfully, now update status to 'training'
        if callback:
            callback.on_dataset_loaded()
            logger.info("Status updated to 'training' - starting model training")
            
            # Update training data info with dataset statistics
            callback.update_model_metadata(training_data_info={
                'data_path': args.data_path,
                'total_samples': len(train_ds) + len(val_ds),
                'training_samples': len(train_ds),
                'validation_samples': len(val_ds),
                'validation_split': args.validation_split,
                'transform_params': transform_params
            })
        
        # Get sample batch for model signature
        sample_batch = next(iter(train_loader))
        logger.info(f"Sample batch shapes - Image: {sample_batch['image'].shape}, Label: {sample_batch['label'].shape}")
        
        # Create model using architecture registry
        model_config = get_default_model_config(args.model_type)
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
            callback.update_training_config({
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'crop_size': args.crop_size,
                'validation_split': args.validation_split,
                'num_workers': getattr(args, 'num_workers', 2),
                'random_flip': getattr(args, 'random_flip', False),
                'random_rotate': getattr(args, 'random_rotate', False),
                'random_scale': getattr(args, 'random_scale', False),
                'random_intensity': getattr(args, 'random_intensity', False)
            })
            callback.update_architecture_info(model, model_config)

        # Model directory was already created during logging setup
        # Save and log model architecture summary
        summary_file = save_model_summary(model, model_dir=model_dir)
        mlflow.log_artifact(summary_file)
        # Save and log training configuration
        config_file = save_config(args, model_dir=model_dir)
        mlflow.log_artifact(config_file)

        loss_function = MonaiDiceLoss(sigmoid=True)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        scaler = torch.cuda.amp.GradScaler()
        
        best_val_dice = -1
        best_model_path = None
        
        epoch_history = []  # Track metrics for each epoch
        
        for epoch in range(args.epochs):
            epoch_start_time = time.time()  # Track epoch timing
            
            # Log epoch start with detailed information
            logger.info(f"[EPOCH] Starting epoch {epoch+1}/{args.epochs}")
            logger.info(f"[EPOCH] Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Log model architecture details on first epoch
            if epoch == 0:
                logger.info("[MODEL] Model Architecture Summary:")
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"[MODEL] Total parameters: {total_params:,}")
                logger.info(f"[MODEL] Trainable parameters: {trainable_params:,}")
                logger.info(f"[MODEL] Model size: {total_params * 4 / (1024**2):.2f} MB")
                
                # Log training configuration
                logger.info(f"[CONFIG] Batch size: {args.batch_size}")
                logger.info(f"[CONFIG] Dataset: Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
                logger.info(f"[CONFIG] Optimizer: Adam, Loss: DiceLoss, Device: {device}")
            
            # Check for stop_requested flag using callback system
            if callback and not callback.on_epoch_start(epoch + 1, args.epochs):
                logger.info("Stop requested via callback. Exiting training loop.")
                break
            elif hasattr(args, 'model_id') and args.model_id is not None and callback is None:
                # Fallback for stop checking if callback is not available
                try:
                    import django
                    import os as _os
                    # Setup Django if not already
                    if not hasattr(django.conf.settings, 'configured') or not django.conf.settings.configured:
                        _os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
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
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                
                # Debug tensor shapes on first batch
                if epoch == 0 and batch_idx == 0:
                    logger.info(f"Training batch shapes - Inputs: {inputs.shape}, Labels: {labels.shape}")
                
                # Ensure labels are float and binary (0 or 1)
                labels = labels.float()
                if labels.max() > 1 or labels.min() < 0:
                    logger.warning(f"Labels out of [0,1] range! min: {labels.min().item()}, max: {labels.max().item()}")
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
                
                if batch_idx % 10 == 0:  # More frequent logging for GUI
                    progress_pct = (batch_idx / len(train_loader)) * 100
                    logger.info(f"[TRAIN] Epoch {epoch+1}/{args.epochs} - "
                              f"Batch {batch_idx}/{len(train_loader)} ({progress_pct:.1f}%) - "
                              f"Loss: {loss.item():.4f}, Dice: {batch_dice:.4f}")
            
            epoch_loss /= len(train_loader)
            train_dice /= len(train_loader)
            
            # Validation
            logger.info(f"[VAL] Starting validation for epoch {epoch+1}")
            model.eval()
            val_loss = 0
            val_dice = 0
            
            with torch.no_grad():
                for val_idx, val_data in enumerate(val_loader):
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    val_outputs = model(val_inputs)
                    batch_val_loss = loss_function(val_outputs, val_labels).item()
                    val_loss += batch_val_loss
                    
                    val_outputs = torch.sigmoid(val_outputs)
                    val_outputs = (val_outputs > 0.5).float()
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    
                    if val_idx % 5 == 0:  # Log every 5th validation batch
                        val_progress_pct = (val_idx / len(val_loader)) * 100
                        logger.info(f"[VAL] Batch {val_idx}/{len(val_loader)} ({val_progress_pct:.1f}%) - "
                                  f"Loss: {batch_val_loss:.4f}")
                
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
            
            # Enhanced MLflow artifact logging using the new artifact manager
            try:
                from shared.utils.mlflow_artifact_manager import log_epoch_artifacts
                
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
                    'epoch': epoch + 1,
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
            
            if callback:
                callback.on_epoch_end(epoch + 1, metrics)

        logger.info(f"Training completed. Best validation Dice score: {best_val_dice:.4f}")
        
        # Enhanced final model logging using the new artifact manager
        try:
            from shared.utils.mlflow_artifact_manager import log_final_model
            
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
                
                # Log PyTorch model using MLflow's model logging
                try:
                    mlflow.pytorch.log_model(
                        model,
                        "pytorch_model",
                        input_example=sample_batch["image"][:1].cpu().numpy(),  # Single sample
                        signature=mlflow.models.infer_signature(
                            sample_batch["image"][:1].cpu().numpy(),
                            torch.sigmoid(model(sample_batch["image"][:1].to(device))).cpu().numpy()
                        )
                    )
                    logger.info("[MLFLOW] PyTorch model logged with signature and input example")
                except Exception as model_log_error:
                    logger.warning(f"[MLFLOW] Failed to log PyTorch model with signature: {model_log_error}")
                    # Fallback without signature
                    mlflow.pytorch.log_model(model, "pytorch_model_fallback")
                
                logger.info(f"[MLFLOW] Enhanced final model logging completed: {len(final_logged_paths)} artifacts")
            else:
                logger.warning("[MLFLOW] No model directory available for enhanced logging, using fallback")
                raise Exception("No model directory for enhanced logging")
                
        except Exception as e:
            logger.warning(f"[MLFLOW] Enhanced final model logging failed, using fallback: {e}")
            
            # Fallback to original final model logging
            try:
                mlflow.pytorch.log_model(
                    model,
                    "model",
                    input_example=sample_batch["image"][:1].cpu().numpy(),
                    signature=mlflow.models.infer_signature(
                        sample_batch["image"][:1].cpu().numpy(),
                        torch.sigmoid(model(sample_batch["image"][:1].to(device))).cpu().numpy()
                    )
                )
                logger.info("[MLFLOW] Model logged to MLflow under 'model' artifact (fallback with signature).")
            except Exception as fallback_signature_error:
                logger.warning(f"[MLFLOW] Failed to log fallback model with signature: {fallback_signature_error}")
                # Final fallback without signature
                mlflow.pytorch.log_model(
                    model,
                    "model",
                    input_example=sample_batch["image"][:1].cpu().numpy(),
                )
                logger.info("[MLFLOW] Model logged to MLflow under 'model' artifact (fallback without signature).")
        
        # Enhanced training artifacts and summary logging
        try:
            # Log training logs with comprehensive organization
            from shared.utils.mlflow_artifact_manager import MLflowArtifactManager
            
            with MLflowArtifactManager() as artifact_manager:
                # Collect all training logs
                log_files = [
                    os.path.join('models', 'artifacts', 'training.log'),
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
                        'training_samples': len(train_ds),
                        'validation_samples': len(val_ds),
                        'total_batches_per_epoch': len(train_loader),
                        'validation_batches_per_epoch': len(val_loader)
                    }
                }
                
                summary_file = os.path.join('models', 'artifacts', 'training_summary.json')
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
                    'models/artifacts/training.log'
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
                
                fallback_summary_file = 'models/artifacts/basic_training_summary.json'
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
                # Import registry functions
                # More robust import path for mlflow_utils
                current_script_dir = os.path.dirname(os.path.abspath(__file__))
                ml_manager_dir = os.path.abspath(os.path.join(current_script_dir, '..', 'ml_manager'))
                if ml_manager_dir not in sys.path:
                    sys.path.append(ml_manager_dir)
                from mlflow_utils import register_model, transition_model_stage
                
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
        
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        # Convert to numpy array and add channel dimension
        img_array = np.array(img)
        if len(img_array.shape) == 2:
            img_array = img_array[np.newaxis, ...]  # Add channel dimension
        
        # Convert to torch tensor
        img_tensor = torch.from_numpy(img_array).float()
        
        return img_tensor
    
    if use_original_size:
        # Don't resize, just load and scale intensity
        return Compose([
            Lambda(load_pil_compatible_image),
            ScaleIntensity(),
        ])
    else:
        return Compose([
            Lambda(load_pil_compatible_image),
            Resize(spatial_size=image_size, mode="bilinear"),
            ScaleIntensity(),
            # ToTensor is not needed since we already have a tensor
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
    
    # Create model using architecture registry
    model_config = get_default_model_config(model_type)
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
            output = torch.sigmoid(output)
            pred = (output > 0.5).float()
            
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

# Architecture Management Functions

