#!/usr/bin/env python3
"""
Test Advanced Loss Functions for Binary Segmentation
=====================================================

This script tests the newly implemented advanced loss functions from pywick
to ensure they work correctly with binary segmentation data.
"""

import torch
import torch.nn.functional as F
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

try:
    import django
    django.setup()
except ImportError:
    print("⚠️ Django not available, testing standalone...")

def create_test_data(batch_size=4, height=64, width=64):
    """Create synthetic test data for binary segmentation."""
    # Model predictions (logits) - raw model outputs before sigmoid
    predictions = torch.randn(batch_size, 1, height, width, dtype=torch.float32)
    
    # Ground truth binary masks - values in [0, 1]
    targets = torch.randint(0, 2, (batch_size, 1, height, width), dtype=torch.float32)
    
    # Add some realistic patterns
    # Create circular regions in some targets to simulate vessels
    center_h, center_w = height // 2, width // 2
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    
    for i in range(batch_size // 2):
        # Create circular vessel-like patterns
        radius = torch.randint(10, 20, (1,)).item()
        circle_mask = ((y - center_h) ** 2 + (x - center_w) ** 2) <= radius ** 2
        targets[i, 0] = circle_mask.float()
    
    return predictions, targets

def test_standard_losses():
    """Test standard loss functions for comparison."""
    print("🧪 Testing Standard Loss Functions")
    print("-" * 40)
    
    predictions, targets = create_test_data()
    
    # Standard PyTorch losses
    standard_losses = {
        'BCE with Logits': torch.nn.BCEWithLogitsLoss(),
        'MSE': torch.nn.MSELoss(),
    }
    
    # Custom Dice loss
    def dice_loss(pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + 1e-6) / (pred_flat.sum() + target_flat.sum() + 1e-6)
        return 1 - dice
    
    standard_losses['Dice Loss'] = dice_loss
    
    for name, loss_fn in standard_losses.items():
        try:
            if name == 'Dice Loss':
                loss_value = loss_fn(predictions, targets)
            else:
                loss_value = loss_fn(predictions, targets)
            print(f"✅ {name:20s}: {loss_value.item():.4f}")
        except Exception as e:
            print(f"❌ {name:20s}: ERROR - {e}")
    
    print()

def test_advanced_losses():
    """Test the newly implemented advanced loss functions."""
    print("🚀 Testing Advanced Loss Functions from pywick")
    print("-" * 50)
    
    try:
        from ml.utils.advanced_losses import (
            TverskyLoss, FocalLoss, ComboDiceBCELoss, SoftDiceLoss,
            WeightedBCELoss, BoundaryLoss, StableBCELoss,
            create_advanced_loss, get_recommended_loss
        )
        
        predictions, targets = create_test_data()
        
        # Test individual loss functions
        advanced_losses = {
            'Tversky (Recall)': TverskyLoss(alpha=0.3, beta=0.7),
            'Tversky (Precision)': TverskyLoss(alpha=0.7, beta=0.3),
            'Focal Loss': FocalLoss(alpha=0.25, gamma=2.0),
            'Combo Dice+BCE': ComboDiceBCELoss(dice_weight=0.7, bce_weight=0.3),
            'Combo with Focal': ComboDiceBCELoss(dice_weight=0.6, bce_weight=0.4, use_focal=True),
            'Soft Dice': SoftDiceLoss(),
            'Weighted BCE': WeightedBCELoss(adaptive=True),
            'Boundary Loss': BoundaryLoss(theta0=3, theta=5),
            'Stable BCE': StableBCELoss(),
        }
        
        results = {}
        for name, loss_fn in advanced_losses.items():
            try:
                loss_value = loss_fn(predictions, targets)
                results[name] = loss_value.item()
                print(f"✅ {name:20s}: {loss_value.item():.4f}")
            except Exception as e:
                print(f"❌ {name:20s}: ERROR - {e}")
                results[name] = None
        
        # Test factory function
        print("\n🏭 Testing Factory Function")
        try:
            factory_loss = create_advanced_loss('tversky', alpha=0.3, beta=0.7)
            factory_value = factory_loss(predictions, targets)
            print(f"✅ Factory Tversky     : {factory_value.item():.4f}")
        except Exception as e:
            print(f"❌ Factory function    : ERROR - {e}")
        
        # Test recommended configurations
        print("\n🎯 Testing Recommended Configurations")
        recommended_configs = ['conservative', 'recall_focused', 'precision_focused', 'class_imbalanced']
        
        for config_name in recommended_configs:
            try:
                recommended_loss = get_recommended_loss(config_name)
                rec_value = recommended_loss(predictions, targets)
                print(f"✅ {config_name:15s}: {rec_value.item():.4f}")
            except Exception as e:
                print(f"❌ {config_name:15s}: ERROR - {e}")
        
        return results
        
    except ImportError as e:
        print(f"❌ Advanced losses not available: {e}")
        return None

def test_loss_manager_integration():
    """Test integration with the enhanced loss manager."""
    print("\n🔧 Testing Loss Manager Integration")
    print("-" * 40)
    
    try:
        from ml.utils.loss_manager import (
            LossManager, get_preset_loss_config, 
            get_coronary_optimized_loss, create_coronary_loss
        )
        
        predictions, targets = create_test_data()
        
        # Test preset configurations
        presets_to_test = [
            'tversky_recall', 'tversky_precision', 'focal_advanced', 
            'combo_dice_bce_focal', 'boundary_aware'
        ]
        
        for preset_name in presets_to_test:
            try:
                config = get_preset_loss_config(preset_name)
                loss_fn = LossManager.create_loss_function(config)
                loss_value = loss_fn(predictions, targets)
                print(f"✅ {preset_name:20s}: {loss_value.item():.4f}")
            except Exception as e:
                print(f"❌ {preset_name:20s}: ERROR - {e}")
        
        # Test coronary-optimized configurations
        print("\n🫀 Testing Coronary-Optimized Configurations")
        coronary_configs = ['balanced', 'recall', 'precision', 'class_imbalanced']
        
        for config_name in coronary_configs:
            try:
                loss_fn = create_coronary_loss(config_name)
                loss_value = loss_fn(predictions, targets)
                print(f"✅ Coronary {config_name:10s}: {loss_value.item():.4f}")
            except Exception as e:
                print(f"❌ Coronary {config_name:10s}: ERROR - {e}")
        
    except ImportError as e:
        print(f"❌ Loss manager not available: {e}")

def test_loss_properties():
    """Test mathematical properties of loss functions."""
    print("\n📊 Testing Loss Function Properties")
    print("-" * 40)
    
    try:
        from ml.utils.advanced_losses import TverskyLoss, FocalLoss, ComboDiceBCELoss
        
        # Test perfect predictions (should give low loss)
        batch_size, height, width = 2, 32, 32
        targets = torch.randint(0, 2, (batch_size, 1, height, width), dtype=torch.float32)
        
        # Perfect predictions (after sigmoid should match targets)
        perfect_logits = torch.logit(targets + 1e-7)  # Add small epsilon to avoid inf
        
        # Bad predictions (opposite of targets)
        bad_logits = torch.logit(1 - targets + 1e-7)
        
        losses_to_test = {
            'Tversky': TverskyLoss(),
            'Focal': FocalLoss(),
            'Combo': ComboDiceBCELoss()
        }
        
        print("Perfect vs Bad Predictions:")
        for name, loss_fn in losses_to_test.items():
            try:
                perfect_loss = loss_fn(perfect_logits, targets).item()
                bad_loss = loss_fn(bad_logits, targets).item()
                print(f"✅ {name:10s}: Perfect={perfect_loss:.4f}, Bad={bad_loss:.4f}, Ratio={bad_loss/perfect_loss:.2f}")
            except Exception as e:
                print(f"❌ {name:10s}: ERROR - {e}")
                
    except ImportError:
        print("❌ Advanced losses not available for property testing")

def test_gradient_flow():
    """Test that gradients flow properly through loss functions."""
    print("\n🌊 Testing Gradient Flow")
    print("-" * 30)
    
    try:
        from ml.utils.advanced_losses import TverskyLoss, FocalLoss, ComboDiceBCELoss
        
        predictions, targets = create_test_data()
        predictions.requires_grad_(True)
        
        losses_to_test = {
            'Tversky': TverskyLoss(),
            'Focal': FocalLoss(),
            'Combo': ComboDiceBCELoss()
        }
        
        for name, loss_fn in losses_to_test.items():
            try:
                loss = loss_fn(predictions, targets)
                loss.backward(retain_graph=True)
                
                if predictions.grad is not None:
                    grad_norm = predictions.grad.norm().item()
                    print(f"✅ {name:10s}: Gradient norm = {grad_norm:.4f}")
                else:
                    print(f"❌ {name:10s}: No gradients!")
                
                predictions.grad.zero_()  # Clear gradients for next test
                
            except Exception as e:
                print(f"❌ {name:10s}: ERROR - {e}")
                
    except ImportError:
        print("❌ Advanced losses not available for gradient testing")

def main():
    """Run all tests."""
    print("🔬 Advanced Loss Functions Test Suite")
    print("=" * 60)
    
    # Test data shapes
    predictions, targets = create_test_data()
    print(f"📏 Test data shapes:")
    print(f"   Predictions (logits): {predictions.shape} [{predictions.dtype}]")
    print(f"   Targets (binary):     {targets.shape} [{targets.dtype}]")
    print(f"   Predictions range:    [{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"   Targets range:        [{targets.min():.2f}, {targets.max():.2f}]")
    print(f"   Positive ratio:       {targets.mean():.2%}")
    print()
    
    # Run all tests
    test_standard_losses()
    advanced_results = test_advanced_losses()
    test_loss_manager_integration()
    test_loss_properties()
    test_gradient_flow()
    
    print("\n🎉 Test Suite Complete!")
    
    if advanced_results:
        print("\n📈 Summary of Advanced Loss Results:")
        for name, value in advanced_results.items():
            if value is not None:
                print(f"   {name:20s}: {value:.4f}")

if __name__ == "__main__":
    main()
