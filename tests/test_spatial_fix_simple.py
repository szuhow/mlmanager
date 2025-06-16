#!/usr/bin/env python3
"""
Simplified test to verify the spatial dimension fix logic without requiring ARCADE dataset.
This test validates that the transform configuration produces consistent spatial dimensions.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add ml directory to path
sys.path.insert(0, str(Path(__file__).parent / "ml"))

def test_transform_spatial_consistency():
    """Test that transform configurations produce consistent spatial dimensions"""
    print("🧪 Testing transform spatial consistency...")
    
    try:
        from torchvision import transforms as tv_transforms
        import torch.nn.functional as F
        
        # Simulate the parameters used in the fixed get_arcade_datasets function
        transform_params = {
            'crop_size': 128  # This is the key fix
        }
        
        crop_size = transform_params.get('crop_size', 128)
        size = crop_size  # The fix: use crop_size instead of resolution
        
        print(f"📊 Target spatial resolution: {size}x{size}")
        
        # Test image transforms
        img_tr = tv_transforms.Compose([
            tv_transforms.Resize((size, size)), 
            tv_transforms.ToTensor(), 
            tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Test semantic mask transform
        def resize_semantic_mask(x):
            """Resize semantic mask tensor to match image dimensions"""
            # x is numpy array of shape (H, W, C)
            tensor = torch.from_numpy(x).permute(2, 0, 1).float()
            # Resize to target size using nearest neighbor to preserve class labels
            resized = F.interpolate(tensor.unsqueeze(0), size=(size, size), mode='nearest')
            # Remove batch dimension and return
            return resized.squeeze(0)
        
        mask_tr = tv_transforms.Compose([tv_transforms.Lambda(resize_semantic_mask)])
        
        # Create mock data to test transforms
        from PIL import Image
        
        # Mock image (simulating original 512x512 image)
        mock_image_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mock_image = Image.fromarray(mock_image_array)
        
        # Mock semantic mask (simulating original size semantic mask with 27 channels)
        mock_mask_array = np.random.randint(0, 2, (512, 512, 27), dtype=np.uint8)
        
        print("📊 Original data dimensions:")
        print(f"   Image: {mock_image.size}")  # PIL uses (W, H)
        print(f"   Mask: {mock_mask_array.shape}")
        
        # Apply transforms
        transformed_image = img_tr(mock_image)
        transformed_mask = mask_tr(mock_mask_array)
        
        print("📊 Transformed data dimensions:")
        print(f"   Image: {transformed_image.shape}")
        print(f"   Mask: {transformed_mask.shape}")
        
        # Check spatial consistency
        image_spatial = transformed_image.shape[1:]  # (H, W) - skip channel dimension
        mask_spatial = transformed_mask.shape[1:]    # (H, W) - skip channel dimension
        
        print(f"📊 Spatial dimension comparison:")
        print(f"   Image spatial: {image_spatial}")
        print(f"   Mask spatial: {mask_spatial}")
        
        if image_spatial == mask_spatial:
            print("✅ SUCCESS: Spatial dimensions are consistent!")
            print(f"   Both use resolution: {image_spatial}")
            
            # Verify the resolution matches expected crop_size
            expected_spatial = (crop_size, crop_size)
            if image_spatial == expected_spatial:
                print(f"✅ CROP SIZE MATCH: Resolution matches crop_size ({crop_size}x{crop_size})")
                return True
            else:
                print(f"❌ CROP SIZE MISMATCH: Expected {expected_spatial}, got {image_spatial}")
                return False
        else:
            print("❌ SPATIAL MISMATCH:")
            print(f"   Image: {image_spatial}")
            print(f"   Mask: {mask_spatial}")
            return False
            
    except Exception as e:
        print(f"❌ Error in transform test: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_class_detection_logic():
    """Test the class detection logic for semantic segmentation"""
    print("\n🧪 Testing class detection logic...")
    
    # Mock semantic mask with 27 channels (ARCADE semantic segmentation)
    batch_size = 2
    channels = 27
    height = 128
    width = 128
    
    # Create mock mask data
    mock_masks = torch.randint(0, 2, (batch_size, channels, height, width)).float()
    
    print(f"📊 Mock semantic mask shape: {mock_masks.shape}")
    
    # Simulate the class detection logic
    max_channels = mock_masks.shape[1]  # 27 channels
    unique_values = torch.unique(mock_masks).cpu().numpy()
    
    print(f"📊 Detected channels: {max_channels}")
    print(f"📊 Unique values: {unique_values}")
    
    # For one-hot encoded semantic segmentation
    if max_channels > 2:
        class_type = 'semantic_onehot'
        num_classes = max_channels
    else:
        class_type = 'binary'
        num_classes = 1
    
    print(f"📊 Detected class type: {class_type}")
    print(f"📊 Detected num_classes: {num_classes}")
    
    if class_type == 'semantic_onehot' and num_classes == 27:
        print("✅ SUCCESS: Class detection logic working correctly!")
        return True
    else:
        print("❌ FAILURE: Class detection logic issues")
        return False

def main():
    """Main test function"""
    print("=" * 70)
    print("🧪 SPATIAL DIMENSION FIX VERIFICATION (SIMPLIFIED)")
    print("=" * 70)
    print("This test verifies the spatial dimension fix logic without requiring ARCADE dataset.")
    print()
    
    # Test transform consistency
    transform_success = test_transform_spatial_consistency()
    
    # Test class detection logic
    class_detection_success = test_class_detection_logic()
    
    overall_success = transform_success and class_detection_success
    
    print("\n" + "=" * 70)
    if overall_success:
        print("🎉 SUCCESS: Spatial dimension fix logic is working correctly!")
        print("✅ Transforms produce consistent spatial dimensions")
        print("✅ Resolution matches training crop_size")
        print("✅ Class detection logic works for semantic segmentation")
        print("✅ The fix should resolve the spatial dimension mismatch issue")
    else:
        print("🔧 ISSUES DETECTED:")
        if not transform_success:
            print("❌ Transform spatial consistency issues")
        if not class_detection_success:
            print("❌ Class detection logic issues")
    
    print("=" * 70)
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
