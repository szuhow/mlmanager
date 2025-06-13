"""
ARCADE Dataset Integration for MLManager
Integrates torch-arcade data loaders with our training system
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Optional, Callable, Tuple, Union, Any
import logging

# Add torch-arcade to path if installed
try:
    from torch_arcade import (
        ARCADEBinarySegmentation,
        ARCADESemanticSegmentation, 
        ARCADEStenosisSegmentation,
        ARCADEStenosisDetection,
        ARCADEArteryClassification
    )
    ARCADE_AVAILABLE = True
except ImportError:
    ARCADE_AVAILABLE = False
    logging.warning("torch-arcade not installed. Using fallback dataset loader.")

# Fallback encoding if torch-arcade not available
FALLBACK_ENCODING = {
    "background": [1] + [0] * 26,
    "1": [0, 1] + [0] * 25,
    "2": [0, 0, 1] + [0] * 24,
    "3": [0, 0, 0, 1] + [0] * 23,
    "4": [0, 0, 0, 0, 1] + [0] * 22,
    "5": [0] * 5 + [1] + [0] * 21,
    "6": [0] * 6 + [1] + [0] * 20,
    "7": [0] * 7 + [1] + [0] * 19,
    "8": [0] * 8 + [1] + [0] * 18,
    "9": [0] * 9 + [1] + [0] * 17,
    "stenosis": [0] * 26 + [1]
}

class ARCADEDatasetAdapter(Dataset):
    """
    Adapter class that integrates ARCADE datasets with MLManager training system
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        task: str = "binary_segmentation",
        image_set: str = "train", 
        side: Optional[str] = None,
        download: bool = False,
        resolution: int = 256,
        use_augmentation: bool = True,
        **augmentation_params
    ):
        """
        Initialize ARCADE dataset adapter
        
        Args:
            root: Root directory for dataset
            task: Task type - 'binary_segmentation', 'semantic_segmentation', 'stenosis_detection', etc.
            image_set: 'train', 'val', or 'test'
            side: 'left', 'right', or None for both
            download: Whether to download dataset if not present
            resolution: Target image resolution
            use_augmentation: Whether to apply data augmentation
            **augmentation_params: Augmentation parameters
        """
        self.original_root = Path(root)
        
        # Auto-detect proper root path and task type for ARCADE structure
        self.root = get_arcade_dataset_root(str(root))
        if task == "auto" or not task:
            self.task = detect_arcade_task_type(str(root))
        else:
            self.task = task
            
        # Get specific paths for this task
        self.task_paths = get_arcade_task_paths(self.root, self.task)
        
        self.image_set = image_set
        self.side = side
        self.resolution = resolution
        self.use_augmentation = use_augmentation and image_set == 'train'
        
        # Create transforms
        self.transform = self._create_transforms(resolution, use_augmentation, augmentation_params)
        
        # Initialize appropriate dataset
        if ARCADE_AVAILABLE:
            self.dataset = self._init_arcade_dataset(download)
        else:
            self.dataset = self._init_fallback_dataset()
            
        logging.info(f"Initialized ARCADE adapter: {self.task}, {len(self.dataset)} samples")
        logging.info(f"Original path: {self.original_root}, Root path: {self.root}")
        logging.info(f"Task paths: {self.task_paths}")
    
    def _init_arcade_dataset(self, download: bool):
        """Initialize torch-arcade dataset"""
        # Use task-specific dataset root if available
        dataset_root = self.task_paths.get("dataset_root", self.root)
        
        if self.task == "binary_segmentation":
            return ARCADEBinarySegmentation(
                root=str(dataset_root),
                image_set=self.image_set,
                download=download,
                transforms=self.transform
            )
        elif self.task == "semantic_segmentation":
            return ARCADESemanticSegmentation(
                root=str(dataset_root),
                image_set=self.image_set,
                side=self.side,
                download=download,
                transforms=self.transform
            )
        elif self.task == "stenosis_segmentation":
            return ARCADEStenosisSegmentation(
                root=str(dataset_root),
                image_set=self.image_set,
                side=self.side,
                download=download,
                transforms=self.transform
            )
        elif self.task == "stenosis_detection":
            return ARCADEStenosisDetection(
                root=str(dataset_root),
                image_set=self.image_set,
                side=self.side,
                download=download,
                transforms=self.transform
            )
        elif self.task == "artery_classification":
            return ARCADEArteryClassification(
                root=str(dataset_root),
                image_set=self.image_set,
                download=download,
                transforms=self.transform
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def _init_fallback_dataset(self):
        """Initialize fallback dataset for compatibility"""
        from .coronary_dataset import CoronaryDataset
        
        # Try to use ARCADE-specific paths first
        if self.image_set == "train":
            image_path = self.task_paths.get("train_images")
        else:
            image_path = self.task_paths.get("val_images")
        
        # Fallback to standard paths
        if not image_path or not image_path.exists():
            root_path = Path(self.root)
            image_path = root_path / "imgs" if (root_path / "imgs").exists() else root_path
            mask_path = root_path / "masks" if (root_path / "masks").exists() else root_path
        else:
            # For ARCADE, masks are typically in annotations
            mask_path = image_path.parent.parent / "annotations"
            if not mask_path.exists():
                mask_path = image_path  # Fallback to same directory
        
        if not image_path.exists():
            raise FileNotFoundError(f"Dataset not found at {image_path}")
            
        # Get image and mask paths
        image_paths = sorted(list(image_path.glob("*.png")) + list(image_path.glob("*.jpg")))
        if mask_path.exists():
            mask_paths = sorted(list(mask_path.glob("*.png")) + list(mask_path.glob("*.jpg")))
        else:
            mask_paths = []
        
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {image_path}")
            
        # Create transform for fallback
        transform = transforms.Resize((self.resolution, self.resolution))
        
        return CoronaryDataset(
            image_paths=image_paths,
            target_paths=mask_paths,
            transform=transform,
            train=self.image_set == 'train'
        )
    
    def _create_transforms(self, resolution: int, use_augmentation: bool, aug_params: dict):
        """Create transforms for dataset"""
        transform_list = [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor()
        ]
        
        if use_augmentation:
            # Add augmentation transforms based on parameters
            if aug_params.get('use_random_flip', False):
                transform_list.insert(-1, transforms.RandomHorizontalFlip(
                    p=aug_params.get('flip_probability', 0.5)
                ))
            
            if aug_params.get('use_random_rotate', False):
                rotation_range = aug_params.get('rotation_range', 10)
                transform_list.insert(-1, transforms.RandomRotation(
                    degrees=rotation_range
                ))
            
            if aug_params.get('use_random_intensity', False):
                intensity_range = aug_params.get('intensity_range', 0.1)
                transform_list.insert(-1, transforms.ColorJitter(
                    brightness=intensity_range,
                    contrast=intensity_range
                ))
        
        # Normalization for medical images
        transform_list.append(transforms.Normalize(
            mean=[0.485], std=[0.229]  # Grayscale normalization
        ))
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from dataset with consistent format"""
        try:
            img, target = self.dataset[index]
            
            # Ensure consistent tensor format
            if not isinstance(img, torch.Tensor):
                img = transforms.ToTensor()(img)
            
            if not isinstance(target, torch.Tensor):
                if isinstance(target, np.ndarray):
                    target = torch.from_numpy(target).float()
                else:
                    target = transforms.ToTensor()(target)
            
            # Ensure correct dimensions
            if img.dim() == 3 and img.shape[0] == 3:
                # Convert RGB to grayscale if needed
                img = transforms.Grayscale()(img)
            
            if target.dim() == 3 and target.shape[0] > 1:
                # For semantic segmentation, take argmax or sum channels
                if self.task == "binary_segmentation":
                    target = (target.sum(dim=0, keepdim=True) > 0).float()
                else:
                    target = target[0:1]  # Take first channel
            
            return img, target
            
        except Exception as e:
            logging.error(f"Error loading item {index}: {e}")
            # Return dummy data on error
            img = torch.zeros(1, self.resolution, self.resolution)
            target = torch.zeros(1, self.resolution, self.resolution)
            return img, target

class ARCADEDataLoader:
    """
    Data loader factory for ARCADE datasets
    """
    
    @staticmethod
    def create_loaders(
        data_path: str,
        task: str = "binary_segmentation",
        batch_size: int = 32,
        validation_split: float = 0.2,
        resolution: int = 256,
        num_workers: int = 4,
        download: bool = False,
        use_augmentation: bool = True,
        **augmentation_params
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation data loaders
        
        Args:
            data_path: Path to dataset root
            task: Task type
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            resolution: Image resolution
            num_workers: Number of worker processes
            download: Whether to download dataset
            use_augmentation: Whether to use data augmentation
            **augmentation_params: Augmentation parameters
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create training dataset
        train_dataset = ARCADEDatasetAdapter(
            root=data_path,
            task=task,
            image_set="train",
            download=download,
            resolution=resolution,
            use_augmentation=use_augmentation,
            **augmentation_params
        )
        
        # Create validation dataset
        try:
            val_dataset = ARCADEDatasetAdapter(
                root=data_path,
                task=task,
                image_set="val",
                download=False,
                resolution=resolution,
                use_augmentation=False
            )
        except:
            # Split training data if no separate validation set
            dataset_size = len(train_dataset)
            val_size = int(validation_split * dataset_size)
            train_size = dataset_size - val_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader
    
    @staticmethod
    def get_available_tasks() -> list:
        """Get list of available tasks"""
        return [
            "binary_segmentation",
            "semantic_segmentation", 
            "stenosis_segmentation",
            "stenosis_detection",
            "artery_classification"
        ]
    
    @staticmethod
    def install_torch_arcade():
        """Install torch-arcade package"""
        import subprocess
        import sys
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/szuhow/torch-arcade"
            ])
            logging.info("Successfully installed torch-arcade")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install torch-arcade: {e}")
            return False

# Utility functions for integration with existing system
def get_arcade_data_loader(data_path: str, **kwargs) -> Tuple[DataLoader, DataLoader]:
    """Convenience function to get ARCADE data loaders"""
    return ARCADEDataLoader.create_loaders(data_path, **kwargs)

def is_arcade_dataset(data_path: str) -> bool:
    """Check if path contains ARCADE dataset structure"""
    path = Path(data_path)
    
    # Check for ARCADE directory structure
    arcade_indicators = [
        "arcade_challenge_datasets",
        "dataset_phase_1",
        "segmentation_dataset",
        "stenosis_dataset",
        "seg_train",
        "seg_val", 
        "sten_train",
        "sten_val"
    ]
    
    # Check if any indicator exists in the path or its subdirectories
    for indicator in arcade_indicators:
        if (path / indicator).exists():
            return True
        # Also check if the current path name contains the indicator
        if indicator in str(path):
            return True
    
    return False

def detect_arcade_task_type(data_path: str) -> str:
    """Automatically detect ARCADE task type from path structure"""
    path = Path(data_path)
    
    # Check for segmentation dataset
    if (path / "segmentation_dataset").exists() or "segmentation" in str(path):
        return "binary_segmentation"
    
    # Check for stenosis dataset
    if (path / "stenosis_dataset").exists() or "stenosis" in str(path):
        return "stenosis_detection"
    
    # Check for specific subdirectories
    if (path / "seg_train").exists() or (path / "seg_val").exists():
        return "binary_segmentation"
    
    if (path / "sten_train").exists() or (path / "sten_val").exists():
        return "stenosis_detection"
    
    # Default fallback
    return "binary_segmentation"

def get_arcade_dataset_root(data_path: str) -> str:
    """Get the correct root path for ARCADE dataset"""
    path = Path(data_path)
    
    # If path points to arcade_challenge_datasets, navigate to phase 1
    if path.name == "arcade_challenge_datasets":
        phase1_path = path / "dataset_phase_1"
        if phase1_path.exists():
            return str(phase1_path)
    
    # If path points to dataset_phase_1, return as is
    if path.name == "dataset_phase_1":
        return str(path)
    
    # If path contains arcade_challenge_datasets, try to find the right level
    if "arcade_challenge_datasets" in str(path):
        parts = path.parts
        try:
            arcade_idx = parts.index("arcade_challenge_datasets")
            # Return path up to dataset_phase_1
            phase1_path = Path(*parts[:arcade_idx+1]) / "arcade_challenge_datasets" / "dataset_phase_1"
            if phase1_path.exists():
                return str(phase1_path)
        except (ValueError, IndexError):
            pass
    
    # Return original path if no special handling needed
    return str(path)

def get_arcade_task_paths(root_path: str, task_type: str) -> dict:
    """Get specific paths for ARCADE tasks"""
    root = Path(root_path)
    
    # Check if root is already pointing to a specific dataset (segmentation_dataset or stenosis_dataset)
    if root.name == "segmentation_dataset":
        return {
            "train_images": root / "seg_train" / "images",
            "train_annotations": root / "seg_train" / "annotations",
            "val_images": root / "seg_val" / "images", 
            "val_annotations": root / "seg_val" / "annotations",
            "dataset_root": root
        }
    
    if root.name == "stenosis_dataset":
        return {
            "train_images": root / "sten_train" / "images",
            "train_annotations": root / "sten_train" / "annotations",
            "val_images": root / "sten_val" / "images",
            "val_annotations": root / "sten_val" / "annotations", 
            "dataset_root": root
        }
    
    # Check for task-specific subdirectories
    if task_type in ["binary_segmentation", "semantic_segmentation"]:
        seg_dataset = root / "segmentation_dataset"
        if seg_dataset.exists():
            return {
                "train_images": seg_dataset / "seg_train" / "images",
                "train_annotations": seg_dataset / "seg_train" / "annotations",
                "val_images": seg_dataset / "seg_val" / "images", 
                "val_annotations": seg_dataset / "seg_val" / "annotations",
                "dataset_root": seg_dataset
            }
    
    elif task_type in ["stenosis_detection", "stenosis_segmentation"]:
        sten_dataset = root / "stenosis_dataset"
        if sten_dataset.exists():
            return {
                "train_images": sten_dataset / "sten_train" / "images",
                "train_annotations": sten_dataset / "sten_train" / "annotations",
                "val_images": sten_dataset / "sten_val" / "images",
                "val_annotations": sten_dataset / "sten_val" / "annotations", 
                "dataset_root": sten_dataset
            }
    
    # Fallback to original path structure
    return {
        "train_images": root / "images",
        "train_annotations": root / "annotations",
        "val_images": root / "images",
        "val_annotations": root / "annotations",
        "dataset_root": root
    }

def detect_arcade_task_from_path(data_path: str) -> str:
    """Auto-detect ARCADE task type from path structure"""
    path = Path(data_path)
    
    # Check for segmentation dataset
    if (path / "segmentation_dataset").exists() or \
       (path / "seg_train").exists() or (path / "seg_val").exists():
        return "binary_segmentation"
    
    # Check for stenosis dataset
    if (path / "stenosis_dataset").exists() or \
       (path / "sten_train").exists() or (path / "sten_val").exists():
        return "stenosis_detection"
    
    # Check parent directories
    path_str = str(path).lower()
    if "segmentation" in path_str:
        return "binary_segmentation"
    elif "stenosis" in path_str:
        return "stenosis_detection"
    
    # Default fallback
    return "binary_segmentation"

def normalize_arcade_path(data_path: str, task: str = None) -> Tuple[Path, str]:
    """
    Normalize ARCADE dataset path to point to the correct subdirectory
    Returns (normalized_path, detected_task)
    """
    path = Path(data_path)
    
    # If path points to arcade_challenge_datasets root, navigate to dataset_phase_1
    if path.name == "arcade_challenge_datasets":
        path = path / "dataset_phase_1"
    
    # If path points to dataset_phase_1, choose appropriate subdataset
    if path.name == "dataset_phase_1":
        if task and "stenosis" in task:
            path = path / "stenosis_dataset"
        else:
            path = path / "segmentation_dataset"
    
    # Auto-detect task if not provided
    if not task:
        task = detect_arcade_task_from_path(str(path))
    
    return path, task

# Integration with existing training system
class ARCADETrainingAdapter:
    """Adapter for integrating ARCADE with existing training system"""
    
    @staticmethod
    def adapt_for_training(data_path: str, model_type: str, **training_params):
        """
        Adapt ARCADE dataset for training with existing model architectures
        
        Args:
            data_path: Path to dataset
            model_type: Type of model being trained
            **training_params: Training parameters
            
        Returns:
            Dictionary with adapted parameters
        """
        # Map model types to ARCADE tasks
        task_mapping = {
            'unet': 'binary_segmentation',
            'unet-semantic': 'semantic_segmentation',
            'stenosis-detector': 'stenosis_detection',
            'artery-classifier': 'artery_classification'
        }
        
        task = task_mapping.get(model_type, 'binary_segmentation')
        
        # Extract relevant parameters
        batch_size = training_params.get('batch_size', 32)
        resolution = training_params.get('resolution', 256)
        validation_split = training_params.get('validation_split', 0.2)
        
        # Augmentation parameters
        augmentation_params = {
            'use_random_flip': training_params.get('use_random_flip', False),
            'flip_probability': training_params.get('flip_probability', 0.5),
            'use_random_rotate': training_params.get('use_random_rotate', False),
            'rotation_range': training_params.get('rotation_range', 10),
            'use_random_intensity': training_params.get('use_random_intensity', False),
            'intensity_range': training_params.get('intensity_range', 0.1)
        }
        
        return {
            'task': task,
            'data_path': data_path,
            'batch_size': batch_size,
            'resolution': resolution,
            'validation_split': validation_split,
            'use_augmentation': True,
            **augmentation_params
        }

if __name__ == "__main__":
    # Test the adapter
    print("Testing ARCADE Dataset Adapter")
    print(f"ARCADE available: {ARCADE_AVAILABLE}")
    print(f"Available tasks: {ARCADEDataLoader.get_available_tasks()}")
    
    # Install torch-arcade if not available
    if not ARCADE_AVAILABLE:
        print("Installing torch-arcade...")
        if ARCADEDataLoader.install_torch_arcade():
            print("torch-arcade installed successfully")
        else:
            print("Failed to install torch-arcade, using fallback")
