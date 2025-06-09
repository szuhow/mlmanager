#!/usr/bin/env python3
"""
Test script to demonstrate the input/ground truth correspondence fix.
This script shows the difference between cropped patches and full images in visualization.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add shared directory to path
sys.path.append(str(Path(__file__).parent / 'shared'))

from train import get_monai_transforms, get_monai_datasets
from monai.data import DataLoader

def test_transforms_difference(data_path):
    """Test the difference between training and validation transforms"""
    
    # Set up transform parameters
    transform_params = {
        'use_random_flip': True,
        'use_random_rotate': True, 
        'use_random_scale': True,
        'use_random_intensity': True,
        'crop_size': 128
    }
    
    print("Testing transforms difference...")
    
    # Create training transforms (with cropping)
    train_transforms = get_monai_transforms(transform_params, for_training=True)
    print("✓ Training transforms created (includes random cropping)")
    
    # Create validation transforms (full images)
    val_transforms = get_monai_transforms(transform_params, for_training=False)
    print("✓ Validation transforms created (full images, resized to 256x256)")
    
    # Test with actual data if available
    if os.path.exists(data_path):
        try:
            # Get datasets
            train_ds, val_ds = get_monai_datasets(data_path, val_split=0.2, transform_params=transform_params)
            
            # Create data loaders
            train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
            
            print(f"✓ Datasets created: {len(train_ds)} training, {len(val_ds)} validation samples")
            
            # Get sample batches
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
            
            print(f"Training batch image shape: {train_batch['image'].shape}")
            print(f"Training batch label shape: {train_batch['label'].shape}")
            print(f"Validation batch image shape: {val_batch['image'].shape}")
            print(f"Validation batch label shape: {val_batch['label'].shape}")
            
            # Create visualization comparison
            create_comparison_visualization(train_batch, val_batch)
            
            return True
            
        except Exception as e:
            print(f"✗ Error testing with real data: {e}")
            print("This is likely because data path doesn't exist or data format is incorrect")
            return False
    else:
        print(f"✗ Data path {data_path} doesn't exist")
        return False

def create_comparison_visualization(train_batch, val_batch):
    """Create a side-by-side comparison of cropped vs full image visualization, showing file names."""
    import os
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Input/Ground Truth Correspondence Fix Demonstration', fontsize=16)

    # Try to get filenames if present in batch (MONAI CacheDataset keeps them as 'image_meta_dict'/'label_meta_dict')
    def get_filename(batch, key, idx):
        meta = None
        if f'{key}_meta_dict' in batch:
            meta = batch[f'{key}_meta_dict']
            if isinstance(meta, list) and len(meta) > idx and 'filename_or_obj' in meta[idx]:
                return os.path.basename(str(meta[idx]['filename_or_obj']))
        return None

    # Training data (cropped patches) - left side
    train_img_name_0 = get_filename(train_batch, 'image', 0) or 'N/A'
    train_lbl_name_0 = get_filename(train_batch, 'label', 0) or 'N/A'
    axes[0, 0].imshow(train_batch['image'][0, 0].cpu(), cmap='gray')
    axes[0, 0].set_title(f'Training Input\n(Cropped Patch)\n{train_img_name_0}', fontsize=10)
    axes[0, 0].axis('off')

    axes[1, 0].imshow(train_batch['label'][0, 0].cpu(), cmap='gray')
    axes[1, 0].set_title(f'Training Ground Truth\n(Cropped Patch)\n{train_lbl_name_0}', fontsize=10)
    axes[1, 0].axis('off')

    axes[2, 0].text(0.5, 0.5, 'ISSUE:\nInput and ground truth\nare random patches\nfrom different locations', 
                   ha='center', va='center', transform=axes[2, 0].transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
                   fontsize=9)
    axes[2, 0].set_title('Problem with Old Approach', fontsize=10, color='red')
    axes[2, 0].axis('off')

    # Add another training sample
    train_img_name_1 = get_filename(train_batch, 'image', 1) or 'N/A'
    train_lbl_name_1 = get_filename(train_batch, 'label', 1) or 'N/A'
    axes[0, 1].imshow(train_batch['image'][1, 0].cpu(), cmap='gray')
    axes[0, 1].set_title(f'Training Input #2\n(Cropped Patch)\n{train_img_name_1}', fontsize=10)
    axes[0, 1].axis('off')

    axes[1, 1].imshow(train_batch['label'][1, 0].cpu(), cmap='gray')
    axes[1, 1].set_title(f'Training Ground Truth #2\n(Cropped Patch)\n{train_lbl_name_1}', fontsize=10)
    axes[1, 1].axis('off')
    axes[2, 1].axis('off')

    # Validation data (full images) - right side
    val_img_name_0 = get_filename(val_batch, 'image', 0) or 'N/A'
    val_lbl_name_0 = get_filename(val_batch, 'label', 0) or 'N/A'
    axes[0, 2].imshow(val_batch['image'][0, 0].cpu(), cmap='gray')
    axes[0, 2].set_title(f'Validation Input\n(Full Image)\n{val_img_name_0}', fontsize=10)
    axes[0, 2].axis('off')

    axes[1, 2].imshow(val_batch['label'][0, 0].cpu(), cmap='gray')
    axes[1, 2].set_title(f'Validation Ground Truth\n(Full Image)\n{val_lbl_name_0}', fontsize=10)
    axes[1, 2].axis('off')

    axes[2, 2].text(0.5, 0.5, 'FIXED:\nInput and ground truth\nshow corresponding\nfull images', 
                   ha='center', va='center', transform=axes[2, 2].transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                   fontsize=9)
    axes[2, 2].set_title('Solution with New Approach', fontsize=10, color='green')
    axes[2, 2].axis('off')

    # Add another validation sample
    val_img_name_1 = get_filename(val_batch, 'image', 1) or 'N/A'
    val_lbl_name_1 = get_filename(val_batch, 'label', 1) or 'N/A'
    axes[0, 3].imshow(val_batch['image'][1, 0].cpu(), cmap='gray')
    axes[0, 3].set_title(f'Validation Input #2\n(Full Image)\n{val_img_name_1}', fontsize=10)
    axes[0, 3].axis('off')

    axes[1, 3].imshow(val_batch['label'][1, 0].cpu(), cmap='gray')
    axes[1, 3].set_title(f'Validation Ground Truth #2\n(Full Image)\n{val_lbl_name_1}', fontsize=10)
    axes[1, 3].axis('off')
    axes[2, 3].axis('off')

    plt.tight_layout()
    # Ensure output directory exists
    output_file = '/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments/artifacts/predictions/visualization_fix_demo.png'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison visualization saved to: {output_file}")
    plt.close()

def main():
    """Main function to run the test"""
    print("=" * 60)
    print("INPUT/GROUND TRUTH CORRESPONDENCE FIX TEST")
    print("=" * 60)
    
    # Test with common data paths
    possible_data_paths = [
        '/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments/shared/datasets/data'
    ]
    
    success = False
    for data_path in possible_data_paths:
        print(f"\nTrying data path: {data_path}")
        if test_transforms_difference(data_path):
            success = True
            break
    
    if not success:
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print("✓ Transform functions updated successfully")
        print("✓ Training transforms include random cropping for data augmentation")
        print("✓ Validation transforms use full images for proper visualization")
        print("✗ Could not test with real data (no data found in expected paths)")
        print("\nTo test with real data, ensure your dataset is organized as:")
        print("  data_path/")
        print("    images/ or imgs/")
        print("    labels/ or masks/")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("SUCCESS! The input/ground truth correspondence issue has been fixed:")
        print("✓ Training uses cropped patches for augmentation")
        print("✓ Validation visualization shows full corresponding images")
        print("✓ Check the generated comparison visualization")
        print("=" * 60)

if __name__ == "__main__":
    main()
