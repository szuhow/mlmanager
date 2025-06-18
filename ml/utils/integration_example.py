"""
Enhanced Training Integration Example
Shows how to integrate checkpointing manager and loss setup in training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_enhanced_training(model_dir: str, model_name: str):
    """
    Example of how to setup enhanced training with checkpointing and loss management.
    
    Args:
        model_dir: Directory for model storage
        model_name: Name of the model
        
    Returns:
        Tuple of (loss_function, checkpoint_manager, training_config)
    """
    # Create model directory
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup Loss Function
    try:
        # Try to use enhanced loss manager
        from ml.utils.loss_manager import LossManager
        
        loss_config = {
            'type': 'combined',
            'dice_weight': 0.7,
            'bce_weight': 0.3,
            'dice_config': {'smooth': 1e-6},
            'bce_config': {}
        }
        
        loss_function = LossManager.create_loss_function(loss_config)
        logger.info("✅ Enhanced loss function created")
        
    except ImportError:
        # Fallback to simple loss
        logger.warning("Enhanced loss manager not available, using simple BCE")
        loss_function = nn.BCEWithLogitsLoss()
        loss_config = {'type': 'bce'}
    
    # 2. Setup Checkpoint Manager
    try:
        from ml.utils.checkpoint_manager import CheckpointManager
        
        checkpoint_config = {
            'save_strategy': 'best',
            'max_checkpoints': 5,
            'monitor_metric': 'val_dice',
            'mode': 'max'
        }
        
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(model_path / 'checkpoints'),
            model_name=model_name,
            **checkpoint_config
        )
        logger.info("✅ Enhanced checkpoint manager created")
        
    except ImportError:
        logger.warning("Checkpoint manager not available")
        checkpoint_manager = None
        checkpoint_config = {}
    
    # 3. Create Training Configuration
    training_config = {
        'model_dir': str(model_path),
        'model_name': model_name,
        'loss_config': loss_config,
        'checkpoint_config': checkpoint_config,
        'setup_timestamp': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    }
    
    logger.info(f"✅ Enhanced training setup completed for {model_name}")
    
    return loss_function, checkpoint_manager, training_config


def enhanced_training_loop_example():
    """
    Example of how to use enhanced training components in a training loop.
    """
    # Setup
    model_dir = './test_model_output'
    model_name = 'test_unet'
    
    loss_function, checkpoint_manager, config = setup_enhanced_training(model_dir, model_name)
    
    # Create simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3, padding=1)
            
        def forward(self, x):
            return torch.sigmoid(self.conv(x))
    
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    logger.info(f"Training on device: {device}")
    
    # Simulate training loop
    num_epochs = 10
    best_val_dice = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        # Simulate training batches
        for batch_idx in range(5):  # 5 batches per epoch
            # Create dummy data
            inputs = torch.randn(2, 1, 64, 64).to(device)
            targets = torch.randint(0, 2, (2, 1, 64, 64)).float().to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                # Dice coefficient
                pred_flat = outputs.view(-1)
                target_flat = targets.view(-1)
                intersection = (pred_flat * target_flat).sum()
                dice = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
                
                train_loss += loss.item()
                train_dice += dice.item()
        
        # Average training metrics
        train_loss /= 5
        train_dice /= 5
        
        # Validation phase (simulated)
        model.eval()
        val_loss = train_loss * 0.9  # Simulate slightly better validation
        val_dice = min(train_dice * 1.1, 1.0)  # Simulate validation dice
        
        # Update scheduler
        scheduler.step()
        
        # Prepare metrics for checkpointing
        metrics = {
            'train_loss': train_loss,
            'train_dice': train_dice,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        # Save checkpoint if we have checkpoint manager
        checkpoint_saved = False
        if checkpoint_manager:
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                model_metadata={'epoch': epoch + 1}
            )
            if checkpoint_path:
                checkpoint_saved = True
                logger.info(f"Checkpoint saved: {Path(checkpoint_path).name}")
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                   f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, "
                   f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if checkpoint_saved:
            logger.info("  ✅ Checkpoint saved")
        
        # Update best validation dice
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            logger.info(f"  🎯 New best validation dice: {best_val_dice:.4f}")
    
    # Final summary
    logger.info(f"\n🎉 Training completed!")
    logger.info(f"Best validation dice: {best_val_dice:.4f}")
    
    if checkpoint_manager:
        summary = checkpoint_manager.get_checkpoint_summary()
        logger.info(f"Total checkpoints saved: {summary['total_checkpoints']}")
        
        best_checkpoint = checkpoint_manager.get_best_checkpoint_path()
        if best_checkpoint:
            logger.info(f"Best checkpoint: {Path(best_checkpoint).name}")
    
    return {
        'best_val_dice': best_val_dice,
        'model': model,
        'training_config': config
    }


def demonstrate_checkpoint_loading():
    """Demonstrate how to load and resume from checkpoints."""
    logger.info("\n🔄 Demonstrating checkpoint loading...")
    
    # Setup
    model_dir = './test_model_output'
    model_name = 'test_unet'
    
    try:
        from ml.utils.checkpoint_manager import CheckpointManager
        
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(Path(model_dir) / 'checkpoints'),
            model_name=model_name,
            save_strategy='best',
            monitor_metric='val_dice',
            mode='max'
        )
        
        # Check if we have any checkpoints
        best_checkpoint = checkpoint_manager.get_best_checkpoint_path()
        if best_checkpoint:
            logger.info(f"Found best checkpoint: {Path(best_checkpoint).name}")
            
            # Create model and load checkpoint
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(1, 1, 3, padding=1)
                    
                def forward(self, x):
                    return torch.sigmoid(self.conv(x))
            
            model = SimpleModel()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            
            # Load checkpoint
            checkpoint_info = checkpoint_manager.load_checkpoint(
                checkpoint_path=best_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler
            )
            
            logger.info(f"✅ Checkpoint loaded successfully!")
            logger.info(f"Resumed from epoch: {checkpoint_info['epoch']}")
            logger.info(f"Best metric: {checkpoint_info['best_metric']:.4f}")
            
            return checkpoint_info
        else:
            logger.info("No checkpoints found to load")
            return None
            
    except ImportError:
        logger.warning("Checkpoint manager not available for loading demo")
        return None


if __name__ == "__main__":
    print("🚀 Enhanced Training Integration Example")
    print("=" * 50)
    
    # Run training example
    result = enhanced_training_loop_example()
    
    # Demonstrate checkpoint loading
    checkpoint_info = demonstrate_checkpoint_loading()
    
    print("\n✅ Integration example completed successfully!")
    print(f"Best validation dice achieved: {result['best_val_dice']:.4f}")
    
    if checkpoint_info:
        print(f"Successfully loaded checkpoint from epoch {checkpoint_info['epoch']}")
