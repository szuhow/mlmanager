#!/usr/bin/env python3
"""
Test script to verify that ARCADE mask scaling fix works correctly.
This test checks that masks are properly scaled from [0,1] to [0,255] for visualization.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add the project root to the Python path
sys.path.insert(0, '/Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager')

try:
    from ml.datasets.torch_arcade_loader import ARCADEBinarySegmentation, ARCADEStenosisSegmentation
    print("✅ Successfully imported ARCADE dataset classes")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_mask_scaling():
    """Test that masks are properly scaled after the fix"""
    print("\n🔍 Testing ARCADE mask scaling fix...")
    
    # Test data paths
    test_paths = [
        '/Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager/data/datasets/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset/seg_train',
        '/Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager/data/datasets/arcade_challenge_datasets/dataset_phase_1/stenosis_dataset/sten_train'
    ]
    
    for dataset_path in test_paths:
        if not os.path.exists(dataset_path):
            print(f"⚠️  Dataset path not found: {dataset_path}")
            continue
            
        print(f"\n📁 Testing dataset: {dataset_path}")
        
        try:
            # Test Binary Segmentation
            if 'segmentation_dataset' in dataset_path:
                print("   Testing ARCADEBinarySegmentation...")
                dataset = ARCADEBinarySegmentation(dataset_path)
                
                # Get a sample
                if len(dataset) > 0:
                    image, mask = dataset[0]
                    print(f"   📊 Mask shape: {mask.shape}")
                    print(f"   📊 Mask dtype: {mask.dtype}")
                    print(f"   📊 Mask min: {mask.min():.3f}")
                    print(f"   📊 Mask max: {mask.max():.3f}")
                    print(f"   📊 Mask mean: {mask.mean():.3f}")
                    
                    # Check if mask values are in [0,1] range (after ToTensor)
                    if mask.max() <= 1.0:
                        print("   ✅ Mask values are in [0,1] range (normalized by ToTensor)")
                    else:
                        print("   ❌ Mask values are NOT properly normalized")
                        
                    # Check if original cached mask is properly scaled
                    sample_img_file = dataset.images[0]
                    sample_img_id = dataset.file_to_id[sample_img_file]
                    cached_mask = dataset._get_cached_mask(sample_img_file, sample_img_id)
                    print(f"   📊 Cached mask min: {cached_mask.min()}")
                    print(f"   📊 Cached mask max: {cached_mask.max()}")
                    
                    if cached_mask.max() == 255:
                        print("   ✅ Cached mask is properly scaled to [0,255]")
                    else:
                        print("   ❌ Cached mask is NOT properly scaled")
                        
            # Test Stenosis Segmentation
            elif 'stenosis_dataset' in dataset_path:
                print("   Testing ARCADEStenosisSegmentation...")
                dataset = ARCADEStenosisSegmentation(dataset_path)
                
                # Get a sample
                if len(dataset) > 0:
                    image, mask = dataset[0]
                    print(f"   📊 Mask shape: {mask.shape}")
                    print(f"   📊 Mask dtype: {mask.dtype}")
                    print(f"   📊 Mask min: {mask.min():.3f}")
                    print(f"   📊 Mask max: {mask.max():.3f}")
                    print(f"   📊 Mask mean: {mask.mean():.3f}")
                    
                    # Check cached mask
                    sample_img_file = dataset.images[0]
                    sample_img_id = dataset.file_to_id[sample_img_file]
                    cached_mask = dataset._get_cached_stenosis_mask(sample_img_file, sample_img_id)
                    print(f"   📊 Cached mask min: {cached_mask.min()}")
                    print(f"   📊 Cached mask max: {cached_mask.max()}")
                    
                    if cached_mask.max() == 255:
                        print("   ✅ Cached stenosis mask is properly scaled to [0,255]")
                    else:
                        print("   ❌ Cached stenosis mask is NOT properly scaled")
                        
        except Exception as e:
            print(f"   ❌ Error testing dataset: {e}")
            continue

def test_mask_visualization():
    """Test mask visualization to ensure masks are visible"""
    print("\n🎨 Testing mask visualization...")
    
    dataset_path = '/Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager/data/datasets/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset/seg_train'
    
    if not os.path.exists(dataset_path):
        print("⚠️  Dataset path not found for visualization test")
        return
        
    try:
        dataset = ARCADEBinarySegmentation(dataset_path)
        
        if len(dataset) > 0:
            image, mask = dataset[0]
            
            # Convert tensors to numpy for visualization
            if hasattr(image, 'numpy'):
                image_np = image.permute(1, 2, 0).numpy()
            else:
                image_np = np.array(image)
                
            if hasattr(mask, 'numpy'):
                mask_np = mask.squeeze().numpy()
            else:
                mask_np = np.array(mask)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(image_np)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Mask
            axes[1].imshow(mask_np, cmap='gray')
            axes[1].set_title(f'Mask (min={mask_np.min():.3f}, max={mask_np.max():.3f})')
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(image_np)
            axes[2].imshow(mask_np, alpha=0.3, cmap='Reds')
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig('/Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager/arcade_mask_test_result.png', dpi=150, bbox_inches='tight')
            print("✅ Visualization saved as 'arcade_mask_test_result.png'")
            
            # Check if mask is visible (not all zeros)
            if mask_np.max() > 0:
                print("✅ Mask contains non-zero values and should be visible")
            else:
                print("❌ Mask is all zeros - not visible")
                
    except Exception as e:
        print(f"❌ Error in visualization test: {e}")

if __name__ == "__main__":
    print("🚀 Starting ARCADE mask scaling fix test...")
    test_mask_scaling()
    test_mask_visualization()
    print("\n✅ Test completed!")
