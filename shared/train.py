import os
import sys
import numpy as np
import logging

logger = logging.getLogger(__name__)

def detect_arcade_dataset(data_path):
    """Detect if data_path contains ARCADE dataset structure"""
    # Check for ARCADE specific structure
    arcade_indicators = [
        'annotations',
        'images',
        'seg_train.json',
        'stenosis.json', 
        'test_case_segmentation'
    ]
    
    for indicator in arcade_indicators:
        indicator_path = os.path.join(data_path, indicator)
        if os.path.exists(indicator_path):
            logger.info(f"[DATASET] ARCADE indicator found: {indicator}")
            return True
    
    # Check if path contains 'arcade' or 'segmentation_dataset'
    if 'arcade' in data_path.lower() or 'segmentation_dataset' in data_path.lower():
        logger.info(f"[DATASET] ARCADE dataset detected from path: {data_path}")
        return True
    
    return False

def get_arcade_datasets(data_path, val_split=0.2, transform_params=None):
    """Get ARCADE datasets with proper loaders"""
    try:
        # Import ARCADE loaders - handle both local and container paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ml_dir = os.path.join(os.path.dirname(current_dir), 'ml')
        
        # Add ml directory to Python path if it exists
        if os.path.exists(ml_dir):
            sys.path.insert(0, os.path.dirname(current_dir))
        else:
            # Try container path
            sys.path.insert(0, '/app')
            
        from ml.datasets.torch_arcade_loader import ARCADEBinarySegmentation
        
        logger.info(f"[DATASET] Loading ARCADE dataset from: {data_path}")
        
        # Determine split (for now, use train split for both training and validation)
        # In future, we can implement proper train/val split for ARCADE
        train_dataset = ARCADEBinarySegmentation(
            dataset_dir=data_path,
            split='train',
            transforms=None  # ARCADE handles transforms internally
        )
        
        # For validation, we'll use a subset of training data
        # This is a temporary solution until ARCADE supports proper validation split
        total_size = len(train_dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        
        # Create indices for train/val split
        indices = list(range(total_size))
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create subset datasets
        try:
            from torch.utils.data import Subset
        except ImportError:
            logger.error("PyTorch not available, cannot create data subsets")
            raise
            
        train_ds = Subset(train_dataset, train_indices)
        val_ds = Subset(train_dataset, val_indices)
        
        logger.info(f"[DATASET] ARCADE dataset loaded: {len(train_ds)} train, {len(val_ds)} validation samples")
        
        return train_ds, val_ds
        
    except Exception as e:
        logger.error(f"[DATASET] Failed to load ARCADE dataset: {e}")
        raise

def get_monai_datasets(data_path, val_split=0.2, transform_params=None):
    import glob
    
    # Check if this is an ARCADE dataset
    if detect_arcade_dataset(data_path):
        logger.info("[DATASET] ARCADE dataset detected, using ARCADE loaders")
        return get_arcade_datasets(data_path, val_split, transform_params)
    
    # Use default params if none provided
    if transform_params is None:
        transform_params = {
            'use_random_flip': True,
            'use_random_rotate': True,
            'use_random_scale': True,
            'use_random_intensity': True,
            'crop_size': 128
        }