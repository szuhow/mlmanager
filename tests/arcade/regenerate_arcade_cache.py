#!/usr/bin/env python3
"""
Script to regenerate all ARCADE cache files
"""
import sys
import os

# Add paths for imports
sys.path.insert(0, '/app/ml')
sys.path.insert(0, '/app')

def regenerate_arcade_cache():
    """Regenerate all ARCADE cache files"""
    try:
        from ml.datasets.torch_arcade_loader import ARCADEBinarySegmentation
        
        print("🔄 Starting ARCADE cache regeneration...")
        
        # Load dataset
        # ARCADE loader expects root to be the parent directory of "arcade_challenge_datasets"
        root_dir = '/app/data/datasets'
        dataset = ARCADEBinarySegmentation(
            root=root_dir,
            image_set='train',
            task='segmentation',
            download=False,
            transforms=None
        )
        
        print(f"📊 Dataset loaded: {len(dataset)} total samples")
        
        # Check current cache status
        cache_dir = '/app/data/datasets/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset/seg_train/masks_binary_cache'
        existing_cache = len([f for f in os.listdir(cache_dir) if f.endswith('.npz')])
        print(f"💾 Current cache files: {existing_cache}")
        
        # Force generation of all cache files by iterating through dataset
        print("🏗️  Generating cache files...")
        total_samples = len(dataset)
        
        for i in range(total_samples):
            if i % 100 == 0:
                progress = (i / total_samples) * 100
                print(f"   Progress: {i}/{total_samples} ({progress:.1f}%)")
            
            try:
                # This will trigger cache generation if file doesn't exist
                image, mask = dataset[i]
            except Exception as e:
                print(f"   ❌ Error processing sample {i}: {e}")
                continue
        
        # Check final cache status
        final_cache = len([f for f in os.listdir(cache_dir) if f.endswith('.npz')])
        print(f"✅ Cache regeneration completed!")
        print(f"   Cache files before: {existing_cache}")
        print(f"   Cache files after: {final_cache}")
        print(f"   Generated: {final_cache - existing_cache} new files")
        
        return True
        
    except Exception as e:
        print(f"❌ Error regenerating cache: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = regenerate_arcade_cache()
    sys.exit(0 if success else 1)
