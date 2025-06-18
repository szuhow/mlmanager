# Enhanced Training System - Integration Guide

This document explains how to integrate the enhanced checkpointing manager and loss setup into your ML training workflows.

## 🚀 Quick Start

### Basic Enhanced Training Setup

```python
from ml.utils.checkpoint_manager import CheckpointManager
from ml.utils.loss_manager import LossManager
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Setup Enhanced Loss Function
loss_config = {
    'type': 'combined',
    'dice_weight': 0.7,
    'bce_weight': 0.3,
    'dice_config': {'smooth': 1e-6},
    'bce_config': {}
}

loss_function = LossManager.create_loss_function(loss_config)

# 2. Setup Checkpoint Manager
checkpoint_manager = CheckpointManager(
    checkpoint_dir='./model_checkpoints',
    model_name='my_model',
    save_strategy='best',
    monitor_metric='val_dice',
    mode='max',
    max_checkpoints=5
)

# 3. Use in Training Loop
for epoch in range(num_epochs):
    # Training code...
    
    # Save checkpoint
    checkpoint_path = checkpoint_manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        metrics={'val_dice': val_dice, 'val_loss': val_loss}
    )
```

## 📊 Components Overview

### 1. Enhanced Checkpoint Manager (`CheckpointManager`)

**Features:**
- Multiple saving strategies (best, epoch, interval, all)
- Automatic cleanup of old checkpoints
- Comprehensive metadata storage
- Easy loading and resuming
- Training progress tracking

**Configuration Options:**
```python
checkpoint_config = {
    'save_strategy': 'best',        # 'best', 'epoch', 'interval', 'all'
    'max_checkpoints': 5,           # Maximum checkpoints to keep
    'monitor_metric': 'val_dice',   # Metric to monitor for 'best' strategy
    'mode': 'max',                  # 'min' or 'max' for monitored metric
    'save_optimizer': True,         # Whether to save optimizer state
    'save_scheduler': True          # Whether to save scheduler state
}
```

### 2. Enhanced Loss Manager (`LossManager`)

**Features:**
- Multiple loss function types (Dice, BCE, Combined, Focal)
- Dynamic weight scheduling
- Detailed loss component tracking
- Preset configurations
- Performance evaluation tools

**Loss Function Types:**
```python
# Combined Dice + BCE Loss (Recommended for segmentation)
loss_config = {
    'type': 'combined',
    'dice_weight': 0.7,
    'bce_weight': 0.3
}

# Enhanced Dice Loss
loss_config = {
    'type': 'dice',
    'smooth': 1e-6,
    'jaccard': False,
    'squared_pred': False
}

# Enhanced BCE with Focal Loss features
loss_config = {
    'type': 'bce',
    'pos_weight': 1.0,
    'alpha': 0.25,
    'gamma': 2.0  # Focal loss gamma
}
```

### 3. Auto Checkpoint Manager (`AutoCheckpointManager`)

**Features:**
- Intelligent checkpoint strategies
- Adaptive saving based on training progress
- Performance-based checkpointing
- Plateau detection

```python
auto_checkpoint_manager = AutoCheckpointManager(
    checkpoint_dir='./checkpoints',
    model_name='adaptive_model',
    auto_strategy='adaptive',  # 'adaptive', 'performance_based', 'time_based'
    monitor_metric='val_dice',
    mode='max'
)
```

## 🔧 Advanced Usage

### 1. Using Presets

```python
from ml.utils.loss_manager import get_preset_loss_config

# Use predefined loss configurations
loss_config = get_preset_loss_config('focal_segmentation')
loss_function = LossManager.create_loss_function(loss_config)

# Available presets:
# - 'default_segmentation'
# - 'focal_segmentation'
# - 'balanced_segmentation'
# - 'dice_focused'
# - 'jaccard_based'
```

### 2. Dynamic Loss Weight Scheduling

```python
from ml.utils.loss_manager import LossScheduler

# Create combined loss function
loss_function = LossManager.create_loss_function({
    'type': 'combined',
    'dice_weight': 0.7,
    'bce_weight': 0.3
})

# Create loss scheduler
scheduler_config = {
    'type': 'adaptive'  # 'adaptive', 'cosine', 'step', 'performance'
}

loss_scheduler = LossManager.create_loss_scheduler(loss_function, scheduler_config)

# In training loop
for epoch in range(num_epochs):
    # Training...
    
    # Update loss weights based on performance
    loss_scheduler.step(epoch, {'val_dice': val_dice})
```

### 3. Comprehensive Training Integration

```python
from ml.utils.training_manager import TrainingManager

# Create training manager with all features
training_manager = TrainingManager(
    model_dir='./training_output',
    model_name='enhanced_model',
    config={
        'loss_config': {
            'type': 'combined',
            'dice_weight': 0.8,
            'bce_weight': 0.2
        },
        'checkpoint_config': {
            'save_strategy': 'best',
            'monitor_metric': 'val_dice',
            'max_checkpoints': 3
        },
        'loss_scheduler': {
            'type': 'adaptive'
        }
    }
)

# Setup training
training_manager.setup_training(model, optimizer, scheduler)

# Training loop with integrated features
for epoch in range(num_epochs):
    for batch in train_loader:
        # Single training step with automatic loss calculation
        step_metrics = training_manager.train_step(model, batch, optimizer, device)
    
    # Validation
    val_metrics = {}
    for batch in val_loader:
        batch_metrics = training_manager.validate_step(model, batch, device)
        # Accumulate metrics...
    
    # Automatic checkpointing and loss scheduling
    training_manager.log_epoch(epoch, train_metrics, val_metrics)
    checkpoint_path = training_manager.save_checkpoint(epoch, val_metrics)
```

## 📈 Monitoring and Analysis

### 1. Training Progress Tracking

```python
# Get comprehensive training summary
summary = checkpoint_manager.get_checkpoint_summary()
print(f"Total checkpoints: {summary['total_checkpoints']}")
print(f"Best checkpoint: {summary['best_checkpoint']}")
print(f"Training progress: {summary['training_progress']}")

# Export training history
checkpoint_manager.export_training_history('./training_history.json')
```

### 2. Loss Function Analysis

```python
# Evaluate loss function performance
performance = LossManager.evaluate_loss_performance(
    loss_function, predictions, targets
)

print(f"Loss value: {performance['loss_value']:.4f}")
print(f"Dice coefficient: {performance['dice_coefficient']:.4f}")
print(f"IoU: {performance['iou']:.4f}")
print(f"Loss components: {performance['loss_components']}")
```

### 3. Resume Training from Checkpoint

```python
# Load best checkpoint
best_checkpoint = checkpoint_manager.get_best_checkpoint_path()
if best_checkpoint:
    resume_info = checkpoint_manager.load_checkpoint(
        checkpoint_path=best_checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    start_epoch = resume_info['epoch']
    best_metric = resume_info['best_metric']
    print(f"Resumed from epoch {start_epoch}, best metric: {best_metric:.4f}")
```

## 🎯 Integration with Existing Training Code

### Minimal Integration (Drop-in Replacement)

```python
# Before: Simple training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Training code...
        loss = criterion(outputs, targets)
        # ...
    
    # Simple checkpoint saving
    if val_dice > best_val_dice:
        torch.save(model.state_dict(), 'best_model.pth')

# After: Enhanced training loop
from ml.utils.integration_example import setup_enhanced_training

# One-line setup
loss_function, checkpoint_manager, config = setup_enhanced_training(
    model_dir='./enhanced_output',
    model_name='my_model'
)

for epoch in range(num_epochs):
    for batch in train_loader:
        # Training code...
        loss = loss_function(outputs, targets)  # Enhanced loss
        # ...
    
    # Enhanced checkpoint saving with automatic best model detection
    checkpoint_manager.save_checkpoint(
        model, optimizer, scheduler, epoch, 
        {'val_dice': val_dice, 'val_loss': val_loss}
    )
```

## 🔍 Best Practices

### 1. Checkpoint Strategy Selection

- **`best`**: Recommended for most cases - saves only when performance improves
- **`epoch`**: For debugging or when you need every epoch saved
- **`interval`**: For long training runs where you want periodic saves
- **`all`**: Combination of best + regular interval saves

### 2. Loss Function Selection

- **Segmentation tasks**: Use `combined` with Dice + BCE
- **Imbalanced datasets**: Add focal loss features (`gamma > 0`)
- **Small objects**: Increase Dice weight (`dice_weight > 0.7`)
- **Large objects**: More balanced weights (`dice_weight ~ 0.5`)

### 3. Monitoring Metrics

- **Primary metric**: Usually validation Dice for segmentation
- **Secondary metrics**: Validation loss, IoU, accuracy
- **Loss components**: Monitor individual Dice and BCE components

### 4. Directory Organization

```
model_output/
├── checkpoints/
│   ├── weights/           # Model checkpoint files
│   ├── metadata/          # Checkpoint metadata
│   └── config/            # Training configurations
├── logs/                  # Training logs
├── training_setup.json    # Training configuration
└── training_history.json  # Complete training history
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure the ml.utils modules are in your Python path
2. **CUDA Memory**: Use checkpointing to reduce memory usage during training
3. **Disk Space**: Adjust `max_checkpoints` to control storage usage
4. **Performance**: Use `save_strategy='best'` to avoid unnecessary I/O

### Error Handling

```python
try:
    from ml.utils.checkpoint_manager import CheckpointManager
    # Enhanced features available
except ImportError:
    # Fallback to basic functionality
    logger.warning("Enhanced features not available, using basic checkpointing")
```

## 📚 Example Scripts

- `integration_example.py`: Complete training example with all features
- `test_enhanced_integration.py`: Test suite for verifying functionality
- `training_manager.py`: High-level training orchestration

## 🚀 Getting Started

1. **Try the example**: Run `python -m ml.utils.integration_example`
2. **Test integration**: Run `python -m ml.utils.test_enhanced_integration`
3. **Adapt to your code**: Use the minimal integration approach shown above

The enhanced training system is designed to be backward-compatible and easy to integrate into existing training pipelines while providing significant improvements in training management and monitoring capabilities.
