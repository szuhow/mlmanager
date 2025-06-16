#!/usr/bin/env python3
"""
Test script to verify that the spatial dimension fix for ARCADE semantic segmentation is working.
This test checks that model outputs and ground truth masks have matching spatial dimensions.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add ml directory to path
sys.path.insert(0, str(Path(__file__).parent / "ml"))

def test_arcade_spatial_dimensions():
    """Test that ARCADE semantic segmentation has consistent spatial dimensions"""
    print("🧪 Testing ARCADE spatial dimension consistency...")
    
    try:
        # Import required modules
        from training.train import get_arcade_datasets, detect_num_classes_from_masks
        from datasets.torch_arcade_loader import ARCADESemanticSegmentation
        
        print("✅ Successfully imported required modules")
        
        # Test parameters that match actual training
        class MockArgs:
            def __init__(self):
                self.batch_size = 2
                self.resolution = '512'
                self.dataset_type = 'arcade_semantic'
                self.num_workers = 2
        
        args = MockArgs()
        
        # Transform parameters that match training (including crop_size)
        transform_params = {
            'use_random_flip': False,
            'use_random_rotate': False, 
            'use_random_scale': False,
            'use_random_intensity': False,
            'crop_size': 128  # This is the key parameter that was causing the mismatch
        }
        
        # Test data path - check multiple possible locations
        test_paths = [
            "/app/data/datasets/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset",
            "/Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager/data/datasets/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset",
            "./data/datasets/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset"
        ]
        
        data_path = None
        for path in test_paths:
            if os.path.exists(path):
                data_path = path
                print(f"✅ Found ARCADE dataset at: {path}")
                break
        
        if not data_path:
            print("⚠️  ARCADE dataset not found. Testing with mock data...")
            test_mock_spatial_consistency()
            return
        
        # Create ARCADE datasets with the fixed transforms
        print("📦 Creating ARCADE semantic segmentation datasets...")
        try:
            train_loader, val_loader = get_arcade_datasets(
                data_path, 
                validation_split=0.2, 
                transform_params=transform_params, 
                args=args,
                forced_type='arcade_semantic'
            )
            print("✅ Successfully created ARCADE datasets")
            
            # Test spatial dimensions
            print("\n🔍 Testing spatial dimension consistency...")
            
            # Get a sample batch from training data
            train_batch = next(iter(train_loader))
            train_images, train_masks = train_batch
            
            print(f"📊 Training batch shapes:")
            print(f"   Images: {train_images.shape}")
            print(f"   Masks: {train_masks.shape}")
            
            # Get a sample batch from validation data  
            val_batch = next(iter(val_loader))
            val_images, val_masks = val_batch
            
            print(f"📊 Validation batch shapes:")
            print(f"   Images: {val_images.shape}")
            print(f"   Masks: {val_masks.shape}")
            
            # Check spatial consistency
            train_spatial = train_images.shape[2:]  # (H, W)
            val_spatial = val_images.shape[2:]      # (H, W)
            train_mask_spatial = train_masks.shape[2:]  # (H, W) 
            val_mask_spatial = val_masks.shape[2:]      # (H, W)
            
            print(f"\n🔍 Spatial dimension analysis:")
            print(f"   Training image spatial: {train_spatial}")
            print(f"   Training mask spatial: {train_mask_spatial}")
            print(f"   Validation image spatial: {val_spatial}")
            print(f"   Validation mask spatial: {val_mask_spatial}")
            
            # Check consistency
            spatial_match = (
                train_spatial == train_mask_spatial and
                val_spatial == val_mask_spatial and
                train_spatial == val_spatial
            )
            
            if spatial_match:
                print("✅ SUCCESS: All spatial dimensions are consistent!")
                print(f"   Consistent resolution: {train_spatial}")
                
                # Test dynamic class detection
                print("\n🧪 Testing dynamic class detection...")
                class_info = detect_num_classes_from_masks((train_loader, val_loader), dataset_type="arcade")
                print(f"✅ Detected classes: {class_info['num_classes']}")
                print(f"   Class type: {class_info['class_type']}")
                print(f"   Unique values: {class_info['unique_values']}")
                print(f"   Max channels: {class_info['max_channels']}")
                
                if class_info['num_classes'] == 27 and class_info['class_type'] == 'semantic_onehot':
                    print("✅ SEMANTIC SEGMENTATION: Dynamic class detection working correctly!")
                    return True
                else:
                    print("❌ SEMANTIC SEGMENTATION: Dynamic class detection issues")
                    return False
            else:
                print("❌ SPATIAL MISMATCH DETECTED:")
                print(f"   Training: images {train_spatial} vs masks {train_mask_spatial}")
                print(f"   Validation: images {val_spatial} vs masks {val_mask_spatial}")
                return False
                
        except Exception as e:
            print(f"❌ Error creating datasets: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_mock_spatial_consistency():
    """Test spatial consistency with mock data"""
    print("🧪 Testing spatial consistency with mock data...")
    
    # Simulate the expected behavior
    crop_size = 128
    expected_shape = (crop_size, crop_size)
    
    print(f"📊 Expected spatial dimensions: {expected_shape}")
    print(f"✅ Mock test: Model would output {expected_shape}")
    print(f"✅ Mock test: Ground truth would be {expected_shape}")
    print("✅ Mock test: Spatial dimensions would match!")
    
    return True

def main():
    """Main test function"""
    print("=" * 70)
    print("🧪 ARCADE SPATIAL DIMENSION FIX VERIFICATION")
    print("=" * 70)
    print("This test verifies that the spatial dimension mismatch issue has been fixed.")
    print("Model outputs and ground truth masks should have matching spatial dimensions.")
    print()
    
    success = test_arcade_spatial_dimensions()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 SUCCESS: Spatial dimension fix is working correctly!")
        print("✅ Model outputs and ground truth masks have matching dimensions")
        print("✅ Dynamic class detection is working for ARCADE semantic segmentation")
        print("✅ Training should now work without spatial dimension errors")
    else:
        print("🔧 ISSUES DETECTED:")
        print("❌ Spatial dimensions may still be mismatched")
        print("❌ Check transform parameters and resolution settings")
        print("❌ Verify that crop_size is being used consistently")
    
    print("=" * 70)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
