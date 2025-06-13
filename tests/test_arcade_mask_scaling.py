#!/usr/bin/env python3
"""
Test ARCADE mask scaling fixes to ensure masks display properly after cache regeneration.
This test verifies that masks are correctly scaled from [0,1] to [0,255] range.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add ml directory to path
sys.path.insert(0, '/Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager/ml')

def test_arcade_mask_scaling():
    """Test that ARCADE masks have correct scaling after cache regeneration"""
    print("🧪 Testing ARCADE mask scaling fixes...")
    
    try:
        from ml.datasets.torch_arcade_loader import ARCADEBinarySegmentation
        
        # Set dataset path
        dataset_dir = "/Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager/shared/datasets/ARCADE/segmentation_dataset"
        
        if not os.path.exists(dataset_dir):
            print(f"❌ ARCADE dataset not found at: {dataset_dir}")
            return False
        
        print(f"✅ Found ARCADE dataset at: {dataset_dir}")
        
        # Load the dataset
        print("📦 Loading ARCADE Binary Segmentation dataset...")
        dataset = ARCADEBinarySegmentation(
            dataset_dir=dataset_dir,
            split='train',
            transforms=None
        )
        
        print(f"✅ Dataset loaded with {len(dataset)} samples")
        
        # Test a few samples
        num_samples_to_test = min(5, len(dataset))
        print(f"🔍 Testing {num_samples_to_test} samples for mask scaling...")
        
        scaling_results = []
        
        for i in range(num_samples_to_test):
            try:
                sample = dataset[i]
                image = sample['image']
                label = sample['label']
                
                # Convert to numpy if tensor
                if hasattr(label, 'numpy'):
                    mask_array = label.numpy()
                else:
                    mask_array = np.array(label)
                
                # Check mask value range
                min_val = np.min(mask_array)
                max_val = np.max(mask_array)
                unique_vals = np.unique(mask_array)
                
                print(f"  Sample {i+1}:")
                print(f"    Mask shape: {mask_array.shape}")
                print(f"    Value range: [{min_val:.3f}, {max_val:.3f}]")
                print(f"    Unique values: {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}")
                
                # Check if scaling is correct
                # After fixing, masks should be in [0, 255] range, not [0, 1]
                has_proper_scaling = max_val > 1.0
                scaling_results.append({
                    'sample_idx': i,
                    'min_val': min_val,
                    'max_val': max_val,
                    'unique_count': len(unique_vals),
                    'has_proper_scaling': has_proper_scaling
                })
                
                if has_proper_scaling:
                    print(f"    ✅ Properly scaled (max > 1.0)")
                else:
                    print(f"    ❌ Still in [0,1] range - scaling fix may not be applied")
                
            except Exception as e:
                print(f"    ❌ Error loading sample {i}: {e}")
                scaling_results.append({
                    'sample_idx': i,
                    'error': str(e),
                    'has_proper_scaling': False
                })
        
        # Summary results
        successful_samples = [r for r in scaling_results if 'error' not in r]
        properly_scaled = [r for r in successful_samples if r['has_proper_scaling']]
        
        print(f"\n📊 Scaling Test Results:")
        print(f"   Total samples tested: {num_samples_to_test}")
        print(f"   Successfully loaded: {len(successful_samples)}")
        print(f"   Properly scaled masks: {len(properly_scaled)}")
        
        if len(properly_scaled) == len(successful_samples) and len(successful_samples) > 0:
            print(f"✅ SUCCESS: All masks have proper scaling!")
            create_visualization_test(dataset, scaling_results[:3])
            return True
        else:
            print(f"❌ ISSUE: Some masks still have [0,1] scaling")
            if len(successful_samples) == 0:
                print("   No samples could be loaded - check dataset integrity")
            else:
                print("   Cache may need to be regenerated or fixes not applied")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import ARCADE loaders: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def create_visualization_test(dataset, scaling_results):
    """Create a visualization to show mask scaling results"""
    print("🎨 Creating mask scaling visualization...")
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('ARCADE Mask Scaling Verification', fontsize=16)
        
        for i, result in enumerate(scaling_results):
            if 'error' in result:
                continue
                
            try:
                sample = dataset[result['sample_idx']]
                image = sample['image']
                label = sample['label']
                
                # Convert to numpy for visualization
                if hasattr(image, 'numpy'):
                    img_array = image.numpy()
                    if img_array.shape[0] <= 3:  # Channel first
                        img_array = np.transpose(img_array, (1, 2, 0))
                    if img_array.shape[2] == 1:  # Grayscale
                        img_array = img_array[:, :, 0]
                else:
                    img_array = np.array(image)
                
                if hasattr(label, 'numpy'):
                    mask_array = label.numpy()
                    if mask_array.shape[0] == 1:  # Channel first
                        mask_array = mask_array[0]
                else:
                    mask_array = np.array(label)
                
                # Plot image
                axes[0, i].imshow(img_array, cmap='gray')
                axes[0, i].set_title(f'Sample {result["sample_idx"]+1} - Image')
                axes[0, i].axis('off')
                
                # Plot mask
                axes[1, i].imshow(mask_array, cmap='gray', vmin=0, vmax=255)
                scaling_status = "✅ Scaled" if result['has_proper_scaling'] else "❌ Unscaled"
                axes[1, i].set_title(f'Mask - Range: [{result["min_val"]:.1f}, {result["max_val"]:.1f}]\n{scaling_status}')
                axes[1, i].axis('off')
                
            except Exception as e:
                axes[0, i].text(0.5, 0.5, f'Error loading\nsample {i}', 
                               ha='center', va='center', transform=axes[0, i].transAxes)
                axes[0, i].axis('off')
                axes[1, i].axis('off')
        
        # Hide unused subplots
        for i in range(len(scaling_results), 3):
            axes[0, i].axis('off')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_dir = Path(__file__).parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / 'arcade_mask_scaling_test.png'
        
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")

def test_cache_regeneration_needed():
    """Test if cache needs to be regenerated by checking file timestamps"""
    print("🔍 Checking if cache regeneration was successful...")
    
    cache_dirs = [
        "/Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager/shared/datasets/ARCADE/segmentation_dataset/cache/ARCADEBinarySegmentation_train",
        "/Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager/shared/datasets/ARCADE/segmentation_dataset/cache/ARCADEArteryClassification_train",
        "/Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager/shared/datasets/ARCADE/segmentation_dataset/cache/ARCADESemanticSegmentationBinary_train"
    ]
    
    recent_regeneration = False
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            # Check file modification times
            cache_files = list(Path(cache_dir).glob('*.pkl'))
            if cache_files:
                # Get the most recent file modification time
                latest_time = max(f.stat().st_mtime for f in cache_files)
                import time
                hours_ago = (time.time() - latest_time) / 3600
                
                print(f"  {cache_dir}:")
                print(f"    Files: {len(cache_files)}")
                print(f"    Latest modified: {hours_ago:.1f} hours ago")
                
                if hours_ago < 24:  # Modified within last 24 hours
                    recent_regeneration = True
            else:
                print(f"  {cache_dir}: No cache files found")
        else:
            print(f"  {cache_dir}: Directory not found")
    
    if recent_regeneration:
        print("✅ Cache files were recently regenerated")
    else:
        print("⚠️  Cache files may be old - consider regenerating")
    
    return recent_regeneration

def main():
    """Main test function"""
    print("=" * 70)
    print("🧪 ARCADE MASK SCALING VERIFICATION TEST")
    print("=" * 70)
    
    # Check cache status
    cache_recent = test_cache_regeneration_needed()
    print()
    
    # Test mask scaling
    scaling_success = test_arcade_mask_scaling()
    
    print("\n" + "=" * 70)
    if scaling_success:
        print("🎉 SUCCESS: ARCADE mask scaling fixes are working correctly!")
        print("✅ Masks are properly scaled from [0,1] to [0,255] range")
        print("✅ Visualization and training should work properly now")
    else:
        print("🔧 ISSUES DETECTED:")
        if not cache_recent:
            print("❌ Cache files may need regeneration")
            print("   Run: python regenerate_arcade_cache.py")
        print("❌ Mask scaling fixes may not be applied correctly")
        print("   Check that all _get_cached_*_mask functions have 'mask = mask * 255' line")
    print("=" * 70)

if __name__ == "__main__":
    main()
