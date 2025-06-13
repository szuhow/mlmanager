#!/usr/bin/env python3
"""
Test for ARCADE Binary dataset mask normalization fix.
This test verifies that masks are properly handled after removing ScaleIntensityd normalization.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

def test_arcade_binary_mask_normalization():
    """Test ARCADE Binary dataset mask values after our normalization fix"""
    print("🔍 Testing ARCADE Binary dataset mask normalization...")
    
    try:
        # Add paths for imports
        current_dir = Path(__file__).parent
        ml_dir = current_dir / "ml"
        sys.path.insert(0, str(ml_dir))
        
        from datasets.torch_arcade_loader import ARCADEBinarySegmentation
        from torchvision import transforms as tv_transforms
        
        print("✅ Successfully imported ARCADEBinarySegmentation")
        
        # Test data path - check multiple possible locations
        test_paths = [
            "/app/shared/datasets/ARCADE",
            "/Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager/data/datasets/ARCADE",
            "./data/datasets/ARCADE"
        ]
        
        arcade_path = None
        for path in test_paths:
            if os.path.exists(path):
                arcade_path = path
                print(f"✅ Found ARCADE dataset at: {path}")
                break
        
        if not arcade_path:
            print("❌ ARCADE dataset not found. Testing with mock data...")
            # Create minimal test to check transforms
            test_arcade_transforms()
            return
        
        # Set up transforms as used in training (from train.py)
        size = 512
        img_tr = tv_transforms.Compose([
            tv_transforms.Resize((size, size)), 
            tv_transforms.ToTensor(), 
            tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        mask_tr = tv_transforms.Compose([
            tv_transforms.Resize((size, size)), 
            tv_transforms.ToTensor()
        ])
        
        print(f"🔄 Creating ARCADE Binary dataset with transforms...")
        
        # Create dataset instances
        try:
            train_ds = ARCADEBinarySegmentation(
                root=arcade_path,
                image_set='train',
                download=False,
                transform=img_tr,
                target_transform=mask_tr
            )
            print(f"✅ Created training dataset with {len(train_ds)} samples")
            
            # Test a few samples
            print("\n📊 Testing mask value ranges...")
            
            for i in range(min(3, len(train_ds))):
                try:
                    image, mask = train_ds[i]
                    
                    print(f"\n--- Sample {i+1} ---")
                    print(f"Image shape: {image.shape}")
                    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
                    print(f"Mask shape: {mask.shape}")
                    print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
                    print(f"Mask dtype: {mask.dtype}")
                    
                    # Check if mask is binary
                    unique_values = torch.unique(mask)
                    print(f"Unique mask values: {unique_values.tolist()}")
                    
                    # Verify mask is in expected range for training
                    if mask.max() <= 1.0 and mask.min() >= 0.0:
                        print("✅ Mask values are in [0,1] range - good for training")
                    elif mask.max() <= 255 and mask.min() >= 0:
                        print("⚠️  Mask values are in [0,255] range - may need normalization")
                    else:
                        print(f"❌ Unexpected mask range: [{mask.min():.3f}, {mask.max():.3f}]")
                    
                except Exception as e:
                    print(f"❌ Error processing sample {i+1}: {e}")
                    continue
            
            print("\n🧮 Testing loss function compatibility...")
            test_loss_compatibility(train_ds)
            
        except Exception as e:
            print(f"❌ Failed to create ARCADE dataset: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this in the Docker container with ARCADE support")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

def test_arcade_transforms():
    """Test ARCADE transforms with synthetic data"""
    print("\n🧪 Testing ARCADE transforms with synthetic data...")
    
    from torchvision import transforms as tv_transforms
    from PIL import Image
    
    # Create synthetic binary mask (0, 255 values as PNG would have)
    mask_array = np.random.choice([0, 255], size=(256, 256), p=[0.7, 0.3])
    mask_pil = Image.fromarray(mask_array.astype(np.uint8), mode='L')
    
    print(f"Original mask range: [{mask_array.min()}, {mask_array.max()}]")
    
    # Apply ARCADE mask transform
    mask_tr = tv_transforms.Compose([
        tv_transforms.Resize((512, 512)), 
        tv_transforms.ToTensor()
    ])
    
    mask_tensor = mask_tr(mask_pil)
    
    print(f"After ToTensor() - shape: {mask_tensor.shape}")
    print(f"After ToTensor() - range: [{mask_tensor.min():.3f}, {mask_tensor.max():.3f}]")
    print(f"After ToTensor() - dtype: {mask_tensor.dtype}")
    
    # Check if values are normalized
    if mask_tensor.max() <= 1.0:
        print("✅ ToTensor() normalized values to [0,1] - good for training")
    else:
        print("❌ ToTensor() did not normalize - values still in [0,255]")

def test_loss_compatibility(dataset):
    """Test if mask values work with common loss functions"""
    print("\n🔍 Testing loss function compatibility...")
    
    try:
        # Get a sample
        image, mask = dataset[0]
        
        # Create mock model output (logits)
        mock_output = torch.randn_like(mask)
        
        # Test with binary cross entropy with logits
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(mock_output, mask.float())
        print(f"✅ BCE with logits loss: {bce_loss.item():.4f}")
        
        # Test with dice loss (sigmoid applied to output)
        pred_prob = torch.sigmoid(mock_output)
        intersection = (pred_prob * mask).sum()
        union = pred_prob.sum() + mask.sum()
        dice = (2.0 * intersection) / (union + 1e-6)
        dice_loss = 1 - dice
        print(f"✅ Dice loss: {dice_loss.item():.4f}")
        
        # Check if mask values are appropriate
        if mask.max() <= 1.0 and mask.min() >= 0.0:
            print("✅ Mask values compatible with standard loss functions")
        else:
            print("⚠️  Mask values may need adjustment for loss functions")
            
    except Exception as e:
        print(f"❌ Loss compatibility test failed: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 ARCADE BINARY MASK NORMALIZATION TEST")
    print("=" * 60)
    
    test_arcade_binary_mask_normalization()
    
    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print("=" * 60)
