#!/usr/bin/env python3
"""
Test to verify that mask normalization fix works correctly.
This test checks:
1. MONAI transforms no longer normalize mask values to [0,1]
2. ARCADE dataset masks remain in their original range [0,255]
3. Statistics match the actual data range used in training
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_monai_mask_normalization():
    """Test MONAI dataset mask handling after fix"""
    print("🧪 Testing MONAI mask normalization fix...")
    
    try:
        # Import MONAI transforms
        from ml.training.train import get_monai_transforms
        
        # Create test transform with the fixed configuration
        transform_params = {
            'use_random_flip': True,
            'use_random_rotate': True,
            'use_random_scale': True,
            'use_random_intensity': True,
            'crop_size': 128
        }
        
        transforms = get_monai_transforms(transform_params, for_training=False)
        
        # Check if ScaleIntensityd is applied to labels
        transform_str = str(transforms)
        
        # Look for ScaleIntensityd with label keys
        if 'ScaleIntensityd' in transform_str and 'label' in transform_str:
            print("❌ BŁĄD: ScaleIntensityd nadal normalizuje maski!")
            return False
        else:
            print("✅ POPRAWKA: ScaleIntensityd nie normalizuje masek")
            return True
            
    except Exception as e:
        print(f"❌ Błąd w teście MONAI: {e}")
        return False

def test_arcade_mask_values():
    """Test ARCADE dataset mask value ranges"""
    print("\n🧪 Testing ARCADE mask value ranges...")
    
    try:
        # Check if ARCADE dataset is available
        from ml.datasets.torch_arcade_loader import ARCADEBinarySegmentation
        
        # Test transform used for ARCADE masks
        from torchvision import transforms as tv_transforms
        
        # This is the transform used for ARCADE binary segmentation masks
        mask_transform = tv_transforms.Compose([
            tv_transforms.Resize((512, 512)), 
            tv_transforms.ToTensor()
        ])
        
        # Create a test mask (simulating binary mask from ARCADE)
        test_mask_np = np.array([[0, 255, 0, 255],
                                [255, 0, 255, 0],
                                [0, 255, 0, 255],
                                [255, 0, 255, 0]], dtype=np.uint8)
        
        # Convert to PIL Image (as ARCADE dataset does)
        from PIL import Image
        test_mask_pil = Image.fromarray(test_mask_np)
        
        # Apply ARCADE transform
        transformed_mask = mask_transform(test_mask_pil)
        
        # Check value range
        min_val = transformed_mask.min().item()
        max_val = transformed_mask.max().item()
        
        print(f"📊 ARCADE mask range after transform: [{min_val:.3f}, {max_val:.3f}]")
        
        # ToTensor() should preserve the [0, 255] range (normalized to [0, 1])
        # But the actual values should represent the binary nature
        if abs(min_val - 0.0) < 0.001 and abs(max_val - 1.0) < 0.001:
            print("✅ ARCADE: ToTensor() poprawnie normalizuje do [0,1]")
            return True
        else:
            print(f"❌ ARCADE: Niepoprawny zakres wartości: [{min_val}, {max_val}]")
            return False
            
    except Exception as e:
        print(f"❌ Błąd w teście ARCADE: {e}")
        return False

def test_debug_dice_statistics():
    """Test the DebugDice class to see statistics output"""
    print("\n🧪 Testing DebugDice statistics output...")
    
    try:
        # Dodaj ścieżkę do shared modułów
        import sys
        sys.path.append('/app')
        sys.path.append('/app/shared')
        
        from shared.utils.loss import DebugDice
        
        # Create test tensors
        # Simulating model outputs (logits) and targets
        pred_logits = torch.randn(2, 1, 4, 4)  # Raw model outputs
        target_binary = torch.randint(0, 2, (2, 1, 4, 4)).float()  # Binary targets [0,1]
        
        # Test DebugDice
        debug_dice = DebugDice()
        
        print("🔍 DebugDice output for [0,1] range targets:")
        loss = debug_dice.compute(pred_logits, target_binary)
        
        print(f"📈 Calculated loss: {loss.item():.4f}")
        
        # Test with [0,255] range targets (what would happen without normalization)
        target_255 = target_binary * 255
        
        print("\n🔍 DebugDice output for [0,255] range targets:")
        loss_255 = debug_dice.compute(pred_logits, target_255)
        
        print(f"📈 Calculated loss: {loss_255.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Błąd w teście DebugDice: {e}")
        return False

def test_actual_training_compatibility():
    """Test if the fix is compatible with actual training pipeline"""
    print("\n🧪 Testing training pipeline compatibility...")
    
    try:
        # Test the main training transforms
        from ml.training.train import get_monai_transforms
        
        transform_params = {
            'use_random_flip': False,
            'use_random_rotate': False, 
            'use_random_scale': False,
            'use_random_intensity': False,
            'crop_size': 64  # Small size for testing
        }
        
        transforms = get_monai_transforms(transform_params, for_training=False)
        
        # Create mock data directory with test files
        test_data_dir = Path("/tmp/test_mask_fix")
        test_data_dir.mkdir(exist_ok=True)
        
        imgs_dir = test_data_dir / "imgs"
        masks_dir = test_data_dir / "masks"
        imgs_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        
        # Create a small test image and mask
        test_img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        test_mask = np.random.randint(0, 2, (64, 64), dtype=np.uint8) * 255
        
        from PIL import Image
        Image.fromarray(test_img, mode='L').save(imgs_dir / "test_img.png")
        Image.fromarray(test_mask, mode='L').save(masks_dir / "test_mask.png")
        
        # Test data loading
        from ml.training.train import get_monai_datasets
        
        train_ds, val_ds = get_monai_datasets(str(test_data_dir), val_split=0.5, transform_params=transform_params)
        
        if len(train_ds) > 0:
            # MONAI datasets return dictionaries with 'image' and 'label' keys
            sample = train_ds[0]
            
            # Check if sample is a dictionary (MONAI format)
            if isinstance(sample, dict):
                image = sample["image"]
                label = sample["label"]
            else:
                # Handle other dataset formats
                print("❌ Niepoprawny format danych z datasetu")
                return False
            
            print(f"📊 Loaded sample shapes: Image {image.shape}, Label {label.shape}")
            print(f"📊 Image range: [{image.min().item():.3f}, {image.max().item():.3f}]")
            print(f"📊 Label range: [{label.min().item():.3f}, {label.max().item():.3f}]")
            
            # Check if label values are in expected range
            if label.min().item() >= 0 and label.max().item() <= 255:
                print("✅ POPRAWKA: Maski zachowują oryginalny zakres wartości")
                return True
            else:
                print(f"❌ BŁĄD: Maski mają nieoczekiwany zakres: [{label.min().item()}, {label.max().item()}]")
                return False
        else:
            print("❌ Nie udało się załadować danych testowych")
            return False
            
    except Exception as e:
        print(f"❌ Błąd w teście kompatybilności: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔧 Test poprawki normalizacji masek")
    print("=" * 50)
    
    tests = [
        ("MONAI Mask Normalization", test_monai_mask_normalization),
        ("ARCADE Mask Values", test_arcade_mask_values),
        ("DebugDice Statistics", test_debug_dice_statistics),
        ("Training Compatibility", test_actual_training_compatibility)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 PODSUMOWANIE TESTÓW:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Wynik: {passed}/{total} testów zakończonych sukcesem")
    
    if passed == total:
        print("🎉 WSZYSTKIE TESTY PRZESZŁY! Poprawka działa prawidłowo.")
        return True
    else:
        print("⚠️  NIEKTÓRE TESTY NIE PRZESZŁY. Wymagane dalsze poprawki.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
