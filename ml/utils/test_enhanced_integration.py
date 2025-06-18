#!/usr/bin/env python3
"""
Enhanced Training Integration Script

This script demonstrates how to integrate the new checkpointing manager
and loss setup into the existing training.py script.

Run this to see the integration in action:
python ml/utils/test_enhanced_integration.py
"""

import os
import sys
import torch
import torch.optim as optim
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'ml'))
sys.path.insert(0, str(project_root / 'core'))

try:
    from ml.utils.training_manager import create_training_manager, quick_training_setup
    from ml.utils.checkpoint_manager import CheckpointManager
    from ml.utils.loss_manager import LossManager, get_preset_loss_config
    ENHANCED_AVAILABLE = True
    logger.info("✅ Enhanced training utilities loaded successfully")
except ImportError as e:
    ENHANCED_AVAILABLE = False
    logger.error(f"❌ Failed to load enhanced utilities: {e}")
    sys.exit(1)

try:
    from ml.training.models.unet.unet_model import UNet
    MODEL_AVAILABLE = True
    logger.info("✅ UNet model loaded successfully")
except ImportError as e:
    MODEL_AVAILABLE = False
    logger.warning(f"⚠️ UNet model not available: {e}")


def test_enhanced_integration():
    """Test the enhanced training integration"""
    
    print("\n🧪 Testing Enhanced Training Integration")
    print("=" * 60)
    
    # Test 1: Quick setup
    print("\n1️⃣  Testing Quick Setup")
    try:
        manager = quick_training_setup(
            model_dir='./test_enhanced_output',
            model_name='test_coronary_unet'
        )
        
        print(f"   ✅ Training manager created: {manager.model_name}")
        print(f"   ✅ Model directory: {manager.model_dir}")
        print(f"   ✅ Loss function: {manager.loss_function.__class__.__name__}")
        print(f"   ✅ Checkpoint manager: {manager.checkpoint_manager.__class__.__name__}")
        
    except Exception as e:
        print(f"   ❌ Quick setup failed: {e}")
        return False
    
    # Test 2: Loss function presets
    print("\n2️⃣  Testing Loss Function Presets")
    try:
        presets_to_test = ['default_segmentation', 'focal_segmentation', 'balanced_segmentation']
        
        for preset in presets_to_test:
            config = get_preset_loss_config(preset)
            loss_fn = LossManager.create_loss_function(config)
            print(f"   ✅ {preset}: {loss_fn.__class__.__name__}")
        
    except Exception as e:
        print(f"   ❌ Loss preset test failed: {e}")
        return False
    
    # Test 3: Checkpoint strategies
    print("\n3️⃣  Testing Checkpoint Strategies")
    try:
        strategies = ['standard', 'adaptive', 'performance']
        
        for strategy in strategies:
            test_manager = create_training_manager(
                model_dir=f'./test_checkpoint_{strategy}',
                model_name=f'test_model_{strategy}',
                checkpoint_preset=strategy
            )
            print(f"   ✅ {strategy}: {test_manager.checkpoint_manager.__class__.__name__}")
        
    except Exception as e:
        print(f"   ❌ Checkpoint strategy test failed: {e}")
        return False
    
    # Test 4: Training simulation
    print("\n4️⃣  Testing Training Simulation")
    try:
        if MODEL_AVAILABLE:
            # Create model
            model = UNet(n_channels=1, n_classes=2)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            print(f"   ✅ Model created and moved to {device}")
            
            # Test training step
            inputs = torch.randn(1, 1, 64, 64).to(device)
            targets = torch.randint(0, 2, (1, 1, 64, 64)).float().to(device)  # Match input channels
            
            step_metrics = manager.train_step(model, (inputs, targets), optimizer, device)
            print(f"   ✅ Training step executed successfully")
            print(f"      Loss: {step_metrics['total_loss']:.4f}")
            print(f"      Dice: {step_metrics['dice']:.4f}")
            
        else:
            print("   ⚠️ Model not available, skipping training simulation")
        
    except Exception as e:
        print(f"   ❌ Training simulation failed: {e}")
        return False
    
    # Test 5: Checkpoint saving
    print("\n5️⃣  Testing Checkpoint Saving")
    try:
        if MODEL_AVAILABLE:
            # Simulate metrics
            metrics = {
                'train_loss': 0.5,
                'train_dice': 0.7,
                'val_loss': 0.4,
                'val_dice': 0.8
            }
            
            checkpoint_path = manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=0,
                metrics=metrics
            )
            
            if checkpoint_path:
                print(f"   ✅ Checkpoint saved: {Path(checkpoint_path).name}")
                print(f"      Location: {checkpoint_path}")
            else:
                print("   ℹ️ Checkpoint not saved (based on strategy)")
            
        else:
            print("   ⚠️ Model not available, skipping checkpoint test")
        
    except Exception as e:
        print(f"   ❌ Checkpoint saving failed: {e}")
        return False
    
    # Test 6: Training summary
    print("\n6️⃣  Testing Training Summary")
    try:
        summary = manager.get_training_summary()
        
        print("   ✅ Training summary generated:")
        print(f"      Model: {summary['model_info']['model_name']}")
        print(f"      Epochs: {summary['training_progress']['total_epochs']}")
        print(f"      Checkpoints: {summary['checkpoint_info'].get('total_checkpoints', 0)}")
        
        # Export summary
        export_path = manager.export_training_data()
        print(f"   ✅ Summary exported to: {export_path}")
        
    except Exception as e:
        print(f"   ❌ Training summary failed: {e}")
        return False
    
    print("\n🎉 All integration tests passed!")
    return True


def demonstrate_integration_in_existing_code():
    """Show how to integrate with existing training code"""
    
    print("\n📝 Integration Example for Existing Training Code")
    print("=" * 60)
    
    integration_code = '''
# Add to your existing train.py imports:
from ml.utils.training_manager import create_training_manager

# Replace your existing checkpoint and loss setup with:
def setup_enhanced_training(model_dir, model_name):
    """Setup enhanced training with one line"""
    return create_training_manager(
        model_dir=model_dir,
        model_name=model_name,
        loss_preset='focal_segmentation',  # or 'default_segmentation'
        checkpoint_preset='adaptive'       # or 'standard'
    )

# In your training loop:
def train_epoch(manager, model, train_loader, optimizer, device, epoch):
    """Enhanced training epoch with integrated features"""
    epoch_loss = 0.0
    epoch_dice = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Use manager's train_step for enhanced logging
        step_metrics = manager.train_step(model, (inputs, targets), optimizer, device)
        
        epoch_loss += step_metrics['total_loss']
        epoch_dice += step_metrics['dice']
        
        # Optional: Log loss components every N batches
        if batch_idx % 10 == 0:
            components = step_metrics
            print(f"Batch {batch_idx}: Loss={components['total_loss']:.4f}")
    
    return {
        'train_loss': epoch_loss / len(train_loader),
        'train_dice': epoch_dice / len(train_loader)
    }

# End of epoch:
def end_training_epoch(manager, model, optimizer, scheduler, epoch, train_metrics, val_metrics):
    """End-of-epoch processing with enhanced features"""
    
    # Save checkpoint (automatic based on strategy)
    checkpoint_path = manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        metrics={**train_metrics, **val_metrics}
    )
    
    # End epoch processing (includes loss scheduling)
    epoch_summary = manager.end_epoch(epoch, train_metrics, val_metrics)
    
    return checkpoint_path, epoch_summary

# At the end of training:
def finish_training(manager):
    """Complete training with summary export"""
    final_summary = manager.get_training_summary()
    export_path = manager.export_training_data()
    
    print(f"Training completed. Summary exported to: {export_path}")
    return final_summary
'''
    
    print(integration_code)
    
    # Show file integration
    print("\n📁 File Integration Steps:")
    print("1. Add imports to your train.py:")
    print("   from ml.utils.training_manager import create_training_manager")
    print("\n2. Replace your loss function setup:")
    print("   # Old: loss_fn = DiceLoss() + BCELoss()")
    print("   # New: manager = create_training_manager(...)")
    print("   #      loss_fn = manager.loss_function")
    print("\n3. Replace your checkpoint saving:")
    print("   # Old: torch.save(model.state_dict(), 'checkpoint.pth')")
    print("   # New: manager.save_checkpoint(model, optimizer, scheduler, epoch, metrics)")
    print("\n4. Use enhanced training steps:")
    print("   # Old: loss = loss_fn(outputs, targets)")
    print("   # New: step_metrics = manager.train_step(model, (inputs, targets), optimizer, device)")


if __name__ == "__main__":
    print("🚀 Enhanced Training Integration Test")
    print("=" * 70)
    
    if not ENHANCED_AVAILABLE:
        print("❌ Enhanced utilities not available. Please check installation.")
        sys.exit(1)
    
    try:
        # Run integration test
        success = test_enhanced_integration()
        
        if success:
            # Show integration example
            demonstrate_integration_in_existing_code()
            
            print("\n" + "=" * 70)
            print("🎉 Enhanced Training Integration Ready!")
            print("   ✅ All components working correctly")
            print("   ✅ Integration examples provided")
            print("   ✅ Ready to integrate with existing training code")
            print("\n📚 Next steps:")
            print("   1. Copy the integration examples to your train.py")
            print("   2. Replace existing loss and checkpoint code")
            print("   3. Run your training with enhanced features")
            
        else:
            print("\n❌ Integration test failed. Please check the errors above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
