# Enhanced Training System - Checkpointing and Loss Management

This document explains the enhanced training system with advanced checkpointing and loss management features.

## 🎯 Overview

The enhanced training system provides:

1. **Advanced Checkpointing Manager** - Intelligent model saving with multiple strategies
2. **Enhanced Loss Function Manager** - Flexible loss combinations with scheduling  
3. **Training Manager** - Orchestrates all components with minimal code changes
4. **Easy Integration** - Drop-in replacement for existing training code

## 🚀 Quick Start

### Basic Setup

```python
from ml.utils.training_manager import quick_training_setup

# One-line setup with sensible defaults
manager = quick_training_setup(
    model_dir='./model_output',
    model_name='coronary_unet_v1'
)

# Use in training loop
for epoch in range(num_epochs):
    # Training step with enhanced features
    step_metrics = manager.train_step(model, (inputs, targets), optimizer, device)
    
    # Validation
    val_metrics = validate_model(model, val_loader)
    
    # Save checkpoint (automatic based on strategy)
    checkpoint_path = manager.save_checkpoint(
        model, optimizer, scheduler, epoch, 
        {**step_metrics, **val_metrics}
    )
    
    # End epoch processing
    epoch_summary = manager.end_epoch(epoch, step_metrics, val_metrics)
```

### Advanced Setup

```python
from ml.utils.training_manager import create_training_manager

# Advanced setup with custom configurations
manager = create_training_manager(
    model_dir='./advanced_output',
    model_name='coronary_unet_focal',
    loss_preset='focal_segmentation',    # Loss function preset
    checkpoint_preset='adaptive',        # Checkpoint strategy
    enhanced=True                        # Enable MLflow integration
)
```

## 📦 Components

### 1. Checkpoint Manager

The checkpoint manager provides intelligent model saving with multiple strategies:

#### Basic Usage

```python
from ml.utils.checkpoint_manager import CheckpointManager

# Standard checkpoint manager
checkpoint_manager = CheckpointManager(
    checkpoint_dir='./checkpoints',
    model_name='my_model',
    save_strategy='best',           # 'best', 'epoch', 'interval', 'all'
    max_checkpoints=5,              # Maximum checkpoints to keep
    monitor_metric='val_dice',      # Metric to monitor
    mode='max'                      # 'min' or 'max'
)

# Save checkpoint
checkpoint_path = checkpoint_manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=epoch,
    metrics={'val_dice': 0.85, 'val_loss': 0.23}
)

# Load checkpoint
checkpoint_info = checkpoint_manager.load_checkpoint(
    checkpoint_path='path/to/checkpoint.pth',
    model=model,
    optimizer=optimizer,
    scheduler=scheduler
)
```

#### Auto Checkpoint Manager

```python
from ml.utils.checkpoint_manager import AutoCheckpointManager

# Automatic checkpoint manager with intelligent strategies
auto_manager = AutoCheckpointManager(
    checkpoint_dir='./auto_checkpoints',
    model_name='my_model',
    auto_strategy='adaptive'        # 'adaptive', 'performance_based', 'time_based'
)
```

#### Checkpoint Strategies

1. **Best Model Strategy** - Save only when metric improves
2. **Every Epoch** - Save after every epoch (with cleanup)
3. **Interval Based** - Save every N epochs
4. **All Models** - Save both best and regular checkpoints
5. **Adaptive** - Intelligent saving based on training progress
6. **Performance Based** - Save models within performance threshold
7. **Time Based** - Save at specific training milestones

### 2. Loss Function Manager

Advanced loss function combinations with dynamic scheduling:

#### Basic Usage

```python
from ml.utils.loss_manager import LossManager, get_preset_loss_config

# Use preset configuration
loss_config = get_preset_loss_config('default_segmentation')
loss_function = LossManager.create_loss_function(loss_config)

# Custom configuration
custom_config = {
    'type': 'combined',
    'dice_weight': 0.7,
    'bce_weight': 0.3,
    'dice_config': {
        'smooth': 1e-6,
        'jaccard': True  # Use Jaccard instead of Dice
    },
    'bce_config': {
        'gamma': 2.0,    # Focal loss
        'alpha': 0.25
    }
}
loss_function = LossManager.create_loss_function(custom_config)

# Get loss components for monitoring
loss_components = loss_function.get_loss_components(predictions, targets)
print(f"Dice Loss: {loss_components['dice_loss']:.4f}")
print(f"BCE Loss: {loss_components['bce_loss']:.4f}")
```

#### Loss Scheduling

```python
from ml.utils.loss_manager import LossScheduler

# Create loss scheduler for dynamic weight adjustment
scheduler_config = {
    'type': 'adaptive',  # 'adaptive', 'cosine', 'step', 'performance'
    'params': {}
}

loss_scheduler = LossManager.create_loss_scheduler(loss_function, scheduler_config)

# In training loop
for epoch in range(num_epochs):
    # ... training code ...
    
    # Update loss weights based on performance
    loss_scheduler.step(epoch, val_metrics)
```

#### Available Loss Functions

1. **Enhanced Dice Loss** - With Jaccard option and smoothing
2. **Enhanced BCE Loss** - With focal loss capabilities  
3. **Combined Loss** - Flexible combination of multiple losses
4. **Preset Configurations** - Ready-to-use combinations

#### Loss Presets

- `default_segmentation` - 70% Dice + 30% BCE
- `focal_segmentation` - 60% Dice + 40% Focal BCE
- `balanced_segmentation` - 50% Dice + 50% BCE
- `dice_focused` - 80% Dice + 20% BCE
- `jaccard_based` - 70% Jaccard + 30% BCE

### 3. Training Manager

Orchestrates all components with minimal code changes:

#### Full Featured Setup

```python
from ml.utils.training_manager import TrainingManager

# Create training manager
manager = TrainingManager(
    model_dir='./training_output',
    model_name='coronary_unet'
)

# Setup loss function
loss_config = {
    'type': 'combined',
    'dice_weight': 0.8,
    'bce_weight': 0.2,
    'scheduler': {
        'type': 'adaptive'
    }
}
manager.setup_loss_function(loss_config)

# Setup checkpointing
checkpoint_config = {
    'manager_type': 'auto',
    'auto_strategy': 'adaptive',
    'max_checkpoints': 3
}
manager.setup_checkpointing(checkpoint_config)

# Training loop integration
for epoch in range(num_epochs):
    # Training phase
    for inputs, targets in train_loader:
        step_metrics = manager.train_step(model, (inputs, targets), optimizer, device)
    
    # Validation phase
    val_metrics = manager.validation_step(model, val_loader, device)
    
    # End epoch
    checkpoint_path = manager.save_checkpoint(model, optimizer, scheduler, epoch, val_metrics)
    epoch_summary = manager.end_epoch(epoch, step_metrics, val_metrics)

# Export training data
summary = manager.get_training_summary()
export_path = manager.export_training_data()
```

## 🔧 Integration with Existing Code

### Step 1: Add Imports

```python
# Add to your train.py
from ml.utils.training_manager import create_training_manager
```

### Step 2: Setup Training Manager

```python
# Replace your existing loss and checkpoint setup
def setup_training(model_dir, model_name):
    return create_training_manager(
        model_dir=model_dir,
        model_name=model_name,
        loss_preset='focal_segmentation',
        checkpoint_preset='adaptive'
    )

# In your main training function
manager = setup_training('./output', 'my_model')
```

### Step 3: Replace Training Loop Code

```python
# Old training step
def old_train_step(model, inputs, targets, optimizer, loss_fn):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

# New training step
def new_train_step(manager, model, inputs, targets, optimizer, device):
    step_metrics = manager.train_step(model, (inputs, targets), optimizer, device)
    return step_metrics  # Returns dict with loss components and metrics
```

### Step 4: Replace Checkpoint Saving

```python
# Old checkpoint saving
def old_save_checkpoint(model, optimizer, epoch, best_loss):
    if current_loss < best_loss:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': current_loss
        }, 'best_model.pth')

# New checkpoint saving
def new_save_checkpoint(manager, model, optimizer, scheduler, epoch, metrics):
    checkpoint_path = manager.save_checkpoint(
        model, optimizer, scheduler, epoch, metrics
    )
    return checkpoint_path  # Automatic best model detection and cleanup
```

## 📊 Monitoring and Logging

### Training Progress

```python
# Get comprehensive training summary
summary = manager.get_training_summary()

print(f"Total epochs: {summary['training_progress']['total_epochs']}")
print(f"Best metric: {summary['model_info']['best_metric']}")
print(f"Checkpoints saved: {summary['checkpoint_info']['total_checkpoints']}")

# Export complete training data
export_path = manager.export_training_data()
```

### Loss Component Monitoring

```python
# Monitor individual loss components
loss_components = manager.loss_function.get_loss_components(predictions, targets)

for component, value in loss_components.items():
    mlflow.log_metric(component, value, step=epoch)
```

### Checkpoint Analysis

```python
# List all checkpoints
checkpoints = manager.checkpoint_manager.list_checkpoints()

# Get best checkpoint
best_checkpoint = manager.checkpoint_manager.get_best_checkpoint_path()

# Get checkpoint summary
checkpoint_summary = manager.get_checkpoint_summary()
```

## 🎛️ Configuration Options

### Loss Function Configurations

```python
# Dice-focused configuration
dice_config = {
    'type': 'combined',
    'dice_weight': 0.8,
    'bce_weight': 0.2,
    'dice_config': {
        'smooth': 1e-6,
        'squared_pred': True
    }
}

# Focal loss configuration
focal_config = {
    'type': 'combined',
    'dice_weight': 0.6,
    'bce_weight': 0.4,
    'bce_config': {
        'gamma': 2.0,
        'alpha': 0.25
    }
}

# Jaccard-based configuration
jaccard_config = {
    'type': 'combined',
    'dice_weight': 0.7,
    'bce_weight': 0.3,
    'dice_config': {
        'jaccard': True,
        'smooth': 1e-6
    }
}
```

### Checkpoint Configurations

```python
# Best model only
best_only_config = {
    'save_strategy': 'best',
    'monitor_metric': 'val_dice',
    'mode': 'max',
    'max_checkpoints': 3
}

# Every epoch with cleanup
every_epoch_config = {
    'save_strategy': 'epoch',
    'max_checkpoints': 5
}

# Adaptive strategy
adaptive_config = {
    'manager_type': 'auto',
    'auto_strategy': 'adaptive',
    'max_checkpoints': 5
}

# Performance-based strategy
performance_config = {
    'manager_type': 'auto',
    'auto_strategy': 'performance_based',
    'max_checkpoints': 3
}
```

## 🧪 Testing

Run the integration test to verify everything works:

```bash
cd /path/to/project
python ml/utils/test_enhanced_integration.py
```

This will test:
- Training manager setup
- Loss function presets
- Checkpoint strategies
- Training simulation
- Checkpoint saving
- Summary generation

## 📈 Performance Benefits

1. **Intelligent Checkpointing** - Saves storage and time with smart strategies
2. **Enhanced Loss Functions** - Better convergence with focal loss and scheduling
3. **Comprehensive Monitoring** - Detailed metrics and component tracking
4. **Easy Integration** - Minimal code changes required
5. **Automatic Cleanup** - No manual checkpoint management needed
6. **Resume Training** - Seamless training continuation from any checkpoint

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Make sure paths are correct
   sys.path.insert(0, '/path/to/project/ml')
   sys.path.insert(0, '/path/to/project/core')
   ```

2. **CUDA Memory Issues**
   ```python
   # The enhanced system is memory efficient, but for very large models:
   checkpoint_config = {
       'save_optimizer': False,  # Don't save optimizer state
       'save_scheduler': False   # Don't save scheduler state
   }
   ```

3. **Disk Space Issues**
   ```python
   # Reduce maximum checkpoints
   checkpoint_config = {
       'max_checkpoints': 2,
       'save_strategy': 'best'  # Only save best models
   }
   ```

## 📝 Examples

See the example files for complete usage demonstrations:

- `ml/utils/enhanced_training_examples.py` - Comprehensive examples
- `ml/utils/test_enhanced_integration.py` - Integration testing
- Training manager integration patterns

The enhanced training system is designed to be a drop-in replacement that provides significantly more functionality with minimal code changes.
