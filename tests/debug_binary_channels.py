#!/usr/bin/env python3
"""
Debug script to understand why ARCADE binary segmentation gets 256 output channels
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set Django settings if needed
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

# Configure Django if running as standalone script
try:
    import django
    django.setup()
except:
    pass

def test_binary_class_detection():
    """Test class detection for ARCADE binary segmentation specifically"""
    print("="*60)
    print("DEBUGGING ARCADE BINARY SEGMENTATION CHANNEL ISSUE")
    print("="*60)
    
    try:
        # Import ARCADE dataset components
        from ml.datasets.torch_arcade_loader import create_arcade_dataloader
        from ml.training.train import detect_num_classes_from_masks
        
        # Create mock args for ARCADE binary segmentation
        class MockArgs:
            def __init__(self):
                self.batch_size = 8
                self.num_workers = 2
                self.data_path = "/home/rafal/Dokumenty/dataset"  # Your dataset path
                self.validation_split = 0.2
                self.crop_size = 128
        
        args = MockArgs()
        
        # Test ARCADE Binary Segmentation specifically
        print("\n[TESTING] ARCADE Binary Segmentation Class Detection...")
        
        # Create binary segmentation dataloaders
        train_loader, val_loader = create_arcade_dataloader(
            root=args.data_path,
            task_type='binary_segmentation',
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            num_workers=args.num_workers,
            transform_params={
                'crop_size': args.crop_size,
                'random_flip': True,
                'random_rotate': True,
                'random_scale': True,
                'random_intensity': True
            }
        )
        
        print(f"[DATASET] Created binary segmentation loaders:")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")
        
        # Get a sample to inspect
        sample_batch = next(iter(train_loader))
        images, masks = sample_batch
        print(f"\n[SAMPLE] Batch shapes:")
        print(f"  - Images: {images.shape}")
        print(f"  - Masks: {masks.shape}")
        print(f"  - Image data type: {images.dtype}")
        print(f"  - Mask data type: {masks.dtype}")
        print(f"  - Mask unique values: {torch.unique(masks)}")
        print(f"  - Mask value range: [{masks.min():.3f}, {masks.max():.3f}]")
        
        # Run class detection
        print(f"\n[CLASS DETECTION] Running detection on binary segmentation...")
        class_info = detect_num_classes_from_masks(
            (train_loader, val_loader), 
            dataset_type="arcade", 
            max_samples=10
        )
        
        print(f"\n[RESULTS] Class Detection Results:")
        print(f"  - Detected classes: {class_info['num_classes']}")
        print(f"  - Class type: {class_info['class_type']}")
        print(f"  - Unique values: {class_info['unique_values']}")
        print(f"  - Max channels: {class_info['max_channels']}")
        print(f"  - Sample count: {class_info['sample_count']}")
        
        # Now test model creation
        print(f"\n[MODEL CONFIG] Testing model configuration...")
        from ml.training.train import get_default_model_config, create_model_from_registry
        
        model_type = "unet"  # Default ARCADE model type
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get default config
        model_config = get_default_model_config(model_type)
        print(f"  - Default config: {model_config}")
        
        # Apply dynamic detection
        input_channels = images.shape[1] if images is not None else 1
        model_config["in_channels"] = input_channels
        
        if class_info['class_type'] == 'semantic_onehot':
            model_config["out_channels"] = class_info['max_channels']
        else:
            model_config["out_channels"] = class_info['num_classes']
        
        print(f"  - Modified config: {model_config}")
        
        # Create model
        print(f"\n[MODEL CREATION] Creating model with config...")
        model, arch_info = create_model_from_registry(
            model_type, 
            device,
            **model_config
        )
        
        # Test with sample data
        print(f"\n[MODEL TEST] Testing forward pass...")
        model.eval()
        with torch.no_grad():
            output = model(images)
            print(f"  - Input shape: {images.shape}")
            print(f"  - Output shape: {output.shape}")
            print(f"  - Expected output shape: {(images.shape[0], class_info['num_classes'], images.shape[2], images.shape[3])}")
        
        # Check if shapes match
        expected_channels = class_info['max_channels'] if class_info['class_type'] == 'semantic_onehot' else class_info['num_classes']
        actual_channels = output.shape[1]
        
        if actual_channels == expected_channels:
            print(f"\n✅ SUCCESS: Model output channels ({actual_channels}) match detected classes ({expected_channels})")
        else:
            print(f"\n❌ ISSUE: Model output channels ({actual_channels}) don't match detected classes ({expected_channels})")
            print(f"   This suggests the dynamic configuration is not being applied correctly!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_binary_class_detection()
    sys.exit(0 if success else 1)
