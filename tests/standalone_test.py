#!/usr/bin/env python3
"""
Standalone test for class detection - includes functions directly
"""

import numpy as np
import torch
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    # Handle multi-class semantic segmentation (single channel)
    elif num_unique > 2:
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

def detect_num_classes_from_masks(dataset_loaders, dataset_type="auto", max_samples=50):
    """
    Dynamically detect the number of classes from mask data for semantic segmentation
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

# Mock dataset for testing
class MockDataset:
    def __init__(self, mask_type='binary'):
        self.mask_type = mask_type
        self.length = 10
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Create fake image and mask data
        image = torch.randn(3, 128, 128)  # RGB image
        
        if self.mask_type == 'binary':
            # Binary mask: 0 and 1
            mask = torch.randint(0, 2, (1, 128, 128)).float()
        elif self.mask_type == 'semantic_onehot':
            # One-hot encoded semantic mask (like ARCADE)
            mask = torch.zeros(128, 128, 27)  # 27 channels as last dimension
            # Ensure we always activate at least a few channels with actual data
            for c in range(min(5, 27)):  # Activate first 5 channels consistently
                mask[:, :, c] = torch.randint(0, 2, (128, 128)).float()
            # Make sure we have some non-zero values
            mask[10:20, 10:20, 0] = 1.0  # Guarantee some activation
        elif self.mask_type == 'semantic_single':
            # Single channel with multiple classes
            mask = torch.randint(0, 5, (1, 128, 128)).float()  # 5 classes
        
        return image, mask

class MockDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset

def test_class_detection():
    print("Testing class detection functionality...")
    
    # Test 1: Binary segmentation
    print("\n=== Test 1: Binary Segmentation ===")
    binary_dataset = MockDataset('binary')
    binary_loader = MockDataLoader(binary_dataset)
    
    result = detect_num_classes_from_masks((binary_loader, binary_loader), dataset_type="binary", max_samples=5)
    print(f"Result: {result}")
    
    expected = {'class_type': 'binary', 'num_classes': 1}
    assert result['class_type'] == expected['class_type'], f"Expected {expected['class_type']}, got {result['class_type']}"
    assert result['num_classes'] == expected['num_classes'], f"Expected {expected['num_classes']}, got {result['num_classes']}"
    print("✅ Binary segmentation test passed!")
    
    # Test 2: One-hot semantic segmentation (ARCADE style)
    print("\n=== Test 2: One-hot Semantic Segmentation ===")
    semantic_dataset = MockDataset('semantic_onehot')
    semantic_loader = MockDataLoader(semantic_dataset)
    
    # Debug: Check what our mock data looks like
    sample_image, sample_mask = semantic_dataset[0]
    print(f"Debug: Sample mask shape: {sample_mask.shape}")
    print(f"Debug: Sample mask unique values: {torch.unique(sample_mask)}")
    print(f"Debug: Sample mask min/max: {sample_mask.min()}/{sample_mask.max()}")
    
    result = detect_num_classes_from_masks((semantic_loader, semantic_loader), dataset_type="arcade", max_samples=5)
    print(f"Result: {result}")
    
    expected = {'class_type': 'semantic_onehot', 'max_channels': 27}
    assert result['class_type'] == expected['class_type'], f"Expected {expected['class_type']}, got {result['class_type']}"
    assert result['max_channels'] == expected['max_channels'], f"Expected {expected['max_channels']}, got {result['max_channels']}"
    print("✅ One-hot semantic segmentation test passed!")
    
    # Test 3: Single-channel multi-class segmentation
    print("\n=== Test 3: Single-channel Multi-class Segmentation ===")
    multi_dataset = MockDataset('semantic_single')
    multi_loader = MockDataLoader(multi_dataset)
    
    result = detect_num_classes_from_masks((multi_loader, multi_loader), dataset_type="auto", max_samples=5)
    print(f"Result: {result}")
    
    expected = {'class_type': 'semantic_single'}
    assert result['class_type'] == expected['class_type'], f"Expected {expected['class_type']}, got {result['class_type']}"
    assert result['num_classes'] >= 5, f"Expected at least 5 classes, got {result['num_classes']}"
    print("✅ Multi-class segmentation test passed!")
    
    print("\n🎉 All class detection tests passed!")

if __name__ == "__main__":
    test_class_detection()
