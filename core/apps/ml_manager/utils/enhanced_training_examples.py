"""
Integration Example: Enhanced Training with Checkpointing and Loss Management

This script demonstrates how to integrate the new checkpointing manager
and loss setup with existing training code.
"""

import os
import sys
import torch
import torch.optim as optim
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'ml'))
sys.path.insert(0, str(project_root / 'core'))

from ml.utils.training_manager import TrainingManager, create_training_manager, quick_training_setup
from ml.utils.checkpoint_manager import CheckpointManager, AutoCheckpointManager
from ml.utils.loss_manager import LossManager, get_preset_loss_config
from ml.training.models.unet.unet_model import UNet

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_integration():
    """Example 1: Basic integration with existing training code"""
    
    print("🔧 Example 1: Basic Training Manager Integration")
    print("=" * 60)
    
    # Setup training manager
    manager = quick_training_setup(
        model_dir='./enhanced_training_output',
        model_name='coronary_unet_enhanced'
    )
    
    # Create a dummy model for demonstration
    model = UNet(n_channels=1, n_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Simulate training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"✅ Training manager setup complete")
    print(f"   - Model directory: {manager.model_dir}")
    print(f"   - Loss function: {manager.loss_function.name}")
    print(f"   - Checkpoint strategy: {manager.checkpoint_manager.save_strategy}")
    
    # Simulate some training epochs
    for epoch in range(3):
        print(f"\n📊 Epoch {epoch + 1}")
        
        # Simulate training metrics
        train_metrics = {
            'train_loss': 0.5 - epoch * 0.1,
            'train_dice': 0.6 + epoch * 0.1
        }
        
        # Simulate validation metrics  
        val_metrics = {
            'val_loss': 0.4 - epoch * 0.08,
            'val_dice': 0.7 + epoch * 0.1
        }
        
        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics={**train_metrics, **val_metrics}
        )
        
        if checkpoint_path:
            print(f"   ✅ Checkpoint saved: {Path(checkpoint_path).name}")
        
        # End epoch processing
        epoch_summary = manager.end_epoch(epoch, train_metrics, val_metrics)
        print(f"   📈 Val Dice: {val_metrics['val_dice']:.4f}")
    
    # Get training summary
    summary = manager.get_training_summary()
    print(f"\n📋 Training Summary:")
    print(f"   - Total epochs: {summary['training_progress']['total_epochs']}")
    print(f"   - Best metric: {summary['model_info']['best_metric']:.4f}")
    print(f"   - Checkpoints saved: {summary['checkpoint_info']['total_checkpoints']}")
    
    return manager


def example_advanced_loss_configuration():
    """Example 2: Advanced loss function configuration"""
    
    print("\n\n🎯 Example 2: Advanced Loss Configuration")
    print("=" * 60)
    
    # Custom loss configuration
    custom_loss_config = {
        'type': 'combined',
        'dice_weight': 0.8,
        'bce_weight': 0.2,
        'dice_config': {
            'smooth': 1e-6,
            'jaccard': True  # Use Jaccard instead of Dice
        },
        'bce_config': {
            'gamma': 2.0,  # Focal loss
            'alpha': 0.25
        },
        'scheduler': {
            'type': 'adaptive'
        }
    }
    
    # Setup training manager with custom loss
    manager = TrainingManager(
        model_dir='./advanced_loss_output',
        model_name='coronary_unet_focal'
    )
    
    manager.setup_loss_function(custom_loss_config)
    manager.setup_checkpointing({
        'manager_type': 'auto',
        'auto_strategy': 'adaptive',
        'max_checkpoints': 3
    })
    
    print(f"✅ Advanced loss setup complete:")
    print(f"   - Loss type: {manager.loss_function.__class__.__name__}")
    print(f"   - Scheduler enabled: {manager.loss_scheduler is not None}")
    print(f"   - Checkpoint strategy: Auto-adaptive")
    
    # Demonstrate loss component logging
    model = UNet(n_channels=1, n_classes=2)
    dummy_pred = torch.randn(2, 2, 64, 64)
    dummy_target = torch.randint(0, 2, (2, 2, 64, 64)).float()
    
    loss_components = manager.loss_function.get_loss_components(dummy_pred, dummy_target)
    print(f"\n📊 Loss Components:")
    for component, value in loss_components.items():
        if isinstance(value, float):
            print(f"   - {component}: {value:.6f}")
        else:
            print(f"   - {component}: {value}")
    
    return manager


def example_checkpoint_strategies():
    """Example 3: Different checkpoint strategies"""
    
    print("\n\n💾 Example 3: Checkpoint Strategy Comparison")
    print("=" * 60)
    
    strategies = [
        ('best_only', {'save_strategy': 'best', 'monitor_metric': 'val_dice', 'mode': 'max'}),
        ('every_epoch', {'save_strategy': 'epoch', 'max_checkpoints': 3}),
        ('adaptive', {'manager_type': 'auto', 'auto_strategy': 'adaptive'}),
        ('performance_based', {'manager_type': 'auto', 'auto_strategy': 'performance_based'})
    ]
    
    for strategy_name, config in strategies:
        print(f"\n🔧 Strategy: {strategy_name}")
        
        # Create checkpoint manager
        if config.get('manager_type') == 'auto':
            manager = AutoCheckpointManager(
                checkpoint_dir=f'./checkpoints_{strategy_name}',
                model_name='test_model',
                auto_strategy=config['auto_strategy']
            )
        else:
            manager = CheckpointManager(
                checkpoint_dir=f'./checkpoints_{strategy_name}',
                model_name='test_model',
                **config
            )
        
        # Simulate different performance scenarios
        model = UNet(n_channels=1, n_classes=2)
        optimizer = optim.Adam(model.parameters())
        
        print(f"   Configuration: {config}")
        
        # Simulate training with varying performance
        for epoch in range(5):
            # Simulate improving then plateauing performance
            if epoch < 3:
                val_dice = 0.6 + epoch * 0.1  # Improving
            else:
                val_dice = 0.8 + (epoch - 3) * 0.01  # Plateauing
            
            metrics = {'val_dice': val_dice, 'val_loss': 1.0 - val_dice}
            
            should_save, checkpoint_type = manager.should_save_checkpoint(epoch, metrics)
            
            if should_save:
                # Don't actually save, just log
                print(f"     Epoch {epoch}: Would save {checkpoint_type} checkpoint (val_dice: {val_dice:.4f})")
            else:
                print(f"     Epoch {epoch}: No checkpoint (val_dice: {val_dice:.4f})")
    
    return strategies


def example_training_loop_integration():
    """Example 4: Complete training loop integration"""
    
    print("\n\n🔄 Example 4: Complete Training Loop Integration")
    print("=" * 60)
    
    # Setup everything
    manager = create_training_manager(
        model_dir='./complete_integration',
        model_name='coronary_segmentation_v1',
        loss_preset='focal_segmentation',
        checkpoint_preset='adaptive',
        enhanced=True
    )
    
    # Create model and training components
    model = UNet(n_channels=1, n_classes=2)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"✅ Complete setup ready:")
    print(f"   - Model: {model.__class__.__name__}")
    print(f"   - Loss: {manager.loss_function.__class__.__name__}")
    print(f"   - Checkpoint: {manager.checkpoint_manager.__class__.__name__}")
    
    # Simulate complete training loop
    num_epochs = 5
    
    for epoch in range(num_epochs):
        print(f"\n🔄 Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        num_batches = 10  # Simulate 10 batches
        
        for batch_idx in range(num_batches):
            # Simulate batch data
            inputs = torch.randn(2, 1, 64, 64).to(device)
            targets = torch.randint(0, 2, (2, 1, 64, 64)).float().to(device)
            
            # Use training manager's train_step
            step_metrics = manager.train_step(model, (inputs, targets), optimizer, device, batch_idx)
            
            train_loss += step_metrics['total_loss']
            train_dice += step_metrics['dice']
            
            if batch_idx % 5 == 0:
                print(f"   Batch {batch_idx}: Loss={step_metrics['total_loss']:.4f}, Dice={step_metrics['dice']:.4f}")
        
        # Average training metrics
        train_metrics = {
            'train_loss': train_loss / num_batches,
            'train_dice': train_dice / num_batches
        }
        
        # Validation phase (simulated)
        val_metrics = {
            'val_loss': train_metrics['train_loss'] * 0.9,  # Slightly better
            'val_dice': train_metrics['train_dice'] * 1.1   # Slightly better
        }
        
        # Step scheduler
        scheduler.step()
        
        # Save checkpoint through manager
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics={**train_metrics, **val_metrics},
            additional_data={'learning_rate': scheduler.get_last_lr()[0]}
        )
        
        # End epoch processing
        epoch_summary = manager.end_epoch(epoch, train_metrics, val_metrics)
        
        print(f"   📊 Train Loss: {train_metrics['train_loss']:.4f}, Val Dice: {val_metrics['val_dice']:.4f}")
        if checkpoint_path:
            print(f"   💾 Checkpoint: {Path(checkpoint_path).name}")
    
    # Export training data
    export_path = manager.export_training_data()
    print(f"\n📁 Training data exported to: {export_path}")
    
    # Final summary
    final_summary = manager.get_training_summary()
    print(f"\n📋 Final Training Summary:")
    print(f"   - Total epochs: {final_summary['training_progress']['total_epochs']}")
    print(f"   - Best metric: {final_summary['model_info']['best_metric']:.4f}")
    print(f"   - Total checkpoints: {final_summary['checkpoint_info']['total_checkpoints']}")
    print(f"   - Best checkpoint: {final_summary['checkpoint_info'].get('best_checkpoint', 'None')}")
    
    return manager, final_summary


def example_resume_training():
    """Example 5: Resume training from checkpoint"""
    
    print("\n\n🔄 Example 5: Resume Training from Checkpoint")
    print("=" * 60)
    
    # This would typically use a real checkpoint file
    # For demo purposes, we'll show the structure
    
    manager = quick_training_setup('./resume_training', 'resumed_model')
    
    # Create model and optimizer
    model = UNet(n_channels=1, n_classes=2)
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)
    
    print("✅ Training manager ready for resume")
    print("   To resume training from a checkpoint:")
    print("   ```python")
    print("   checkpoint_info = manager.load_checkpoint(")
    print("       checkpoint_path='path/to/checkpoint.pth',")
    print("       model=model,")
    print("       optimizer=optimizer,")
    print("       scheduler=scheduler")
    print("   )")
    print("   start_epoch = checkpoint_info['epoch']")
    print("   ```")
    
    return manager


if __name__ == "__main__":
    print("🚀 Enhanced Training Integration Examples")
    print("=" * 70)
    
    try:
        # Run all examples
        manager1 = example_basic_integration()
        manager2 = example_advanced_loss_configuration()
        strategies = example_checkpoint_strategies()
        manager4, summary = example_training_loop_integration()
        manager5 = example_resume_training()
        
        print("\n" + "=" * 70)
        print("🎉 All examples completed successfully!")
        print("   Check the output directories for generated files.")
        print("   Integration is ready for use in your training scripts.")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
