# Enhanced Training Integration for GUI

## 🎯 Current Status

The enhanced training features (advanced checkpointing and loss management) are **NOT currently integrated into the GUI** but can be easily added. The current GUI uses basic training configuration without the enhanced features.

## 📋 Current GUI Training Features

The existing training form in `core/apps/ml_manager/forms.py` includes:

- Basic model configuration (model type, data path, epochs, batch size)
- Learning rate scheduling
- Early stopping
- Data augmentation options
- Device selection
- Optimizer selection

**Missing Enhanced Features:**
- ❌ Advanced checkpoint strategies
- ❌ Enhanced loss function selection (Dice+BCE combinations)
- ❌ Loss weight scheduling
- ❌ Advanced checkpoint management

## 🚀 How to Integrate Enhanced Features into GUI

### 1. Add Enhanced Fields to TrainingForm

```python
# Add these fields to core/apps/ml_manager/forms.py

class TrainingForm(forms.Form):
    # ... existing fields ...
    
    # Enhanced Loss Function Configuration
    LOSS_FUNCTION_CHOICES = [
        ('bce', 'Binary Cross Entropy'),
        ('dice', 'Dice Loss'),
        ('combined_default', 'Combined (70% Dice + 30% BCE)'),
        ('combined_balanced', 'Combined (50% Dice + 50% BCE)'),
        ('combined_dice_focused', 'Combined (80% Dice + 20% BCE)'),
        ('focal_segmentation', 'Focal Loss + Dice'),
        ('jaccard_based', 'Jaccard + BCE'),
        ('custom', 'Custom Configuration')
    ]
    
    loss_function = forms.ChoiceField(
        choices=LOSS_FUNCTION_CHOICES,
        initial='combined_default',
        required=True,
        help_text="Loss function for training. Combined losses often work better for segmentation."
    )
    
    # Custom loss configuration (shown when 'custom' is selected)
    dice_weight = forms.FloatField(
        min_value=0.0, max_value=1.0, initial=0.7, required=False,
        help_text="Weight for Dice loss component (0.0-1.0)"
    )
    
    bce_weight = forms.FloatField(
        min_value=0.0, max_value=1.0, initial=0.3, required=False,
        help_text="Weight for BCE loss component (0.0-1.0)"
    )
    
    use_focal_loss = forms.BooleanField(
        initial=False, required=False,
        help_text="Use focal loss features for handling class imbalance"
    )
    
    focal_gamma = forms.FloatField(
        min_value=0.0, max_value=5.0, initial=2.0, required=False,
        help_text="Focal loss gamma parameter (higher = more focus on hard examples)"
    )
    
    focal_alpha = forms.FloatField(
        min_value=0.0, max_value=1.0, initial=0.25, required=False,
        help_text="Focal loss alpha parameter for class weighting"
    )
    
    # Loss Weight Scheduling
    use_loss_scheduling = forms.BooleanField(
        initial=False, required=False,
        help_text="Enable dynamic loss weight adjustment during training"
    )
    
    LOSS_SCHEDULER_CHOICES = [
        ('none', 'No Scheduling'),
        ('adaptive', 'Adaptive (based on validation performance)'),
        ('cosine', 'Cosine Annealing'),
        ('step', 'Step-based'),
        ('performance', 'Performance-based')
    ]
    
    loss_scheduler_type = forms.ChoiceField(
        choices=LOSS_SCHEDULER_CHOICES,
        initial='adaptive',
        required=False,
        help_text="Type of loss weight scheduling"
    )
    
    # Enhanced Checkpoint Configuration
    CHECKPOINT_STRATEGY_CHOICES = [
        ('best', 'Best Model Only'),
        ('epoch', 'Every Epoch'),
        ('interval', 'Every N Epochs'),
        ('adaptive', 'Adaptive (intelligent saving)'),
        ('performance_based', 'Performance-based'),
        ('all', 'Best + Regular Checkpoints')
    ]
    
    checkpoint_strategy = forms.ChoiceField(
        choices=CHECKPOINT_STRATEGY_CHOICES,
        initial='best',
        required=True,
        help_text="Strategy for saving model checkpoints"
    )
    
    max_checkpoints = forms.IntegerField(
        min_value=1, max_value=20, initial=5, required=True,
        help_text="Maximum number of checkpoints to keep (automatic cleanup)"
    )
    
    CHECKPOINT_METRIC_CHOICES = [
        ('val_dice', 'Validation Dice Score'),
        ('val_loss', 'Validation Loss'),
        ('val_accuracy', 'Validation Accuracy'),
        ('train_loss', 'Training Loss')
    ]
    
    checkpoint_monitor_metric = forms.ChoiceField(
        choices=CHECKPOINT_METRIC_CHOICES,
        initial='val_dice',
        required=True,
        help_text="Metric to monitor for best model selection"
    )
    
    checkpoint_mode = forms.ChoiceField(
        choices=[('min', 'Minimize'), ('max', 'Maximize')],
        initial='max',
        required=True,
        help_text="Whether to minimize or maximize the monitored metric"
    )
    
    save_optimizer_state = forms.BooleanField(
        initial=True, required=False,
        help_text="Save optimizer state in checkpoints (enables exact resume)"
    )
    
    save_scheduler_state = forms.BooleanField(
        initial=True, required=False,
        help_text="Save scheduler state in checkpoints"
    )
    
    # Enhanced Training Features
    use_enhanced_training = forms.BooleanField(
        initial=True, required=False,
        help_text="Enable enhanced training features (recommended)"
    )
```

### 2. Update Training View to Use Enhanced Features

```python
# Update core/apps/ml_manager/views.py

class StartTrainingView(LoginRequiredMixin, FormView):
    # ... existing code ...
    
    def form_valid(self, form):
        # ... existing setup code ...
        
        form_data = form.cleaned_data
        
        # Add enhanced training configuration
        enhanced_config = {}
        
        if form_data.get('use_enhanced_training', True):
            # Loss function configuration
            loss_config = self._get_loss_config(form_data)
            enhanced_config['loss_config'] = loss_config
            
            # Checkpoint configuration
            checkpoint_config = self._get_checkpoint_config(form_data)
            enhanced_config['checkpoint_config'] = checkpoint_config
            
            # Loss scheduling configuration
            if form_data.get('use_loss_scheduling'):
                scheduler_config = self._get_loss_scheduler_config(form_data)
                enhanced_config['loss_scheduler_config'] = scheduler_config
        
        # Update training_data_info with enhanced config
        training_data_info = {
            # ... existing fields ...
            'enhanced_training': enhanced_config,
            'use_enhanced_features': form_data.get('use_enhanced_training', True)
        }
        
        ml_model = MLModel.objects.create(
            # ... existing fields ...
            training_data_info=training_data_info
        )
        
        # Update command to include enhanced features
        command = self._build_enhanced_command(ml_model, form_data)
        
        # ... rest of subprocess launch ...
    
    def _get_loss_config(self, form_data):
        """Build loss configuration from form data."""
        loss_type = form_data['loss_function']
        
        if loss_type == 'bce':
            return {'type': 'bce'}
        elif loss_type == 'dice':
            return {'type': 'dice', 'smooth': 1e-6}
        elif loss_type == 'combined_default':
            return {'type': 'combined', 'dice_weight': 0.7, 'bce_weight': 0.3}
        elif loss_type == 'combined_balanced':
            return {'type': 'combined', 'dice_weight': 0.5, 'bce_weight': 0.5}
        elif loss_type == 'combined_dice_focused':
            return {'type': 'combined', 'dice_weight': 0.8, 'bce_weight': 0.2}
        elif loss_type == 'focal_segmentation':
            return {
                'type': 'combined',
                'dice_weight': 0.6,
                'bce_weight': 0.4,
                'bce_config': {
                    'gamma': form_data.get('focal_gamma', 2.0),
                    'alpha': form_data.get('focal_alpha', 0.25)
                }
            }
        elif loss_type == 'custom':
            config = {
                'type': 'combined',
                'dice_weight': form_data.get('dice_weight', 0.7),
                'bce_weight': form_data.get('bce_weight', 0.3)
            }
            if form_data.get('use_focal_loss'):
                config['bce_config'] = {
                    'gamma': form_data.get('focal_gamma', 2.0),
                    'alpha': form_data.get('focal_alpha', 0.25)
                }
            return config
        
        return {'type': 'combined', 'dice_weight': 0.7, 'bce_weight': 0.3}
    
    def _get_checkpoint_config(self, form_data):
        """Build checkpoint configuration from form data."""
        return {
            'save_strategy': form_data['checkpoint_strategy'],
            'max_checkpoints': form_data['max_checkpoints'],
            'monitor_metric': form_data['checkpoint_monitor_metric'],
            'mode': form_data['checkpoint_mode'],
            'save_optimizer': form_data.get('save_optimizer_state', True),
            'save_scheduler': form_data.get('save_scheduler_state', True)
        }
    
    def _get_loss_scheduler_config(self, form_data):
        """Build loss scheduler configuration from form data."""
        return {
            'type': form_data.get('loss_scheduler_type', 'adaptive')
        }
    
    def _build_enhanced_command(self, ml_model, form_data):
        """Build training command with enhanced features."""
        command = [
            sys.executable,
            str(Path(__file__).parent.parent.parent.parent / 'ml' / 'training' / 'train.py'),
            '--mode=train',
            f'--model-id={ml_model.id}',
            # ... existing arguments ...
        ]
        
        # Add enhanced training flag
        if form_data.get('use_enhanced_training', True):
            command.append('--use-enhanced-training')
            command.append(f'--loss-function={form_data["loss_function"]}')
            command.append(f'--checkpoint-strategy={form_data["checkpoint_strategy"]}')
            
            if form_data.get('use_loss_scheduling'):
                command.append('--use-loss-scheduling')
                command.append(f'--loss-scheduler-type={form_data.get("loss_scheduler_type", "adaptive")}')
        
        return command
```

### 3. Update Training Template with Enhanced Fields

```html
<!-- Add to core/apps/ml_manager/templates/ml_manager/start_training.html -->

<!-- Enhanced Training Section -->
<div class="row mb-4">
    <div class="col-12">
        <h5 class="text-primary">Enhanced Training Features</h5>
        <hr>
    </div>
</div>

<!-- Enable Enhanced Features -->
<div class="row mb-3">
    <div class="col-md-12">
        <div class="form-check">
            {{ form.use_enhanced_training }}
            <label class="form-check-label" for="{{ form.use_enhanced_training.id_for_label }}">
                {{ form.use_enhanced_training.label }}
            </label>
            <div class="form-text">{{ form.use_enhanced_training.help_text }}</div>
        </div>
    </div>
</div>

<div id="enhanced-features-section" style="display: none;">
    <!-- Loss Function Configuration -->
    <div class="card mb-4">
        <div class="card-header">
            <h6 class="mb-0">Loss Function Configuration</h6>
        </div>
        <div class="card-body">
            <div class="row mb-3">
                <div class="col-md-6">
                    <label for="{{ form.loss_function.id_for_label }}" class="form-label">
                        {{ form.loss_function.label }}
                    </label>
                    {{ form.loss_function }}
                    <div class="form-text">{{ form.loss_function.help_text }}</div>
                </div>
                <div class="col-md-6">
                    <div class="form-check">
                        {{ form.use_loss_scheduling }}
                        <label class="form-check-label" for="{{ form.use_loss_scheduling.id_for_label }}">
                            {{ form.use_loss_scheduling.label }}
                        </label>
                    </div>
                    <div class="mt-2" id="loss-scheduler-options" style="display: none;">
                        {{ form.loss_scheduler_type }}
                    </div>
                </div>
            </div>
            
            <!-- Custom Loss Configuration (shown when custom is selected) -->
            <div id="custom-loss-config" style="display: none;">
                <div class="row">
                    <div class="col-md-3">
                        <label for="{{ form.dice_weight.id_for_label }}" class="form-label">
                            {{ form.dice_weight.label }}
                        </label>
                        {{ form.dice_weight }}
                    </div>
                    <div class="col-md-3">
                        <label for="{{ form.bce_weight.id_for_label }}" class="form-label">
                            {{ form.bce_weight.label }}
                        </label>
                        {{ form.bce_weight }}
                    </div>
                    <div class="col-md-6">
                        <div class="form-check">
                            {{ form.use_focal_loss }}
                            <label class="form-check-label" for="{{ form.use_focal_loss.id_for_label }}">
                                {{ form.use_focal_loss.label }}
                            </label>
                        </div>
                        <div id="focal-loss-params" class="row mt-2" style="display: none;">
                            <div class="col-6">{{ form.focal_gamma }}</div>
                            <div class="col-6">{{ form.focal_alpha }}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Checkpoint Configuration -->
    <div class="card mb-4">
        <div class="card-header">
            <h6 class="mb-0">Checkpoint Management</h6>
        </div>
        <div class="card-body">
            <div class="row mb-3">
                <div class="col-md-4">
                    <label for="{{ form.checkpoint_strategy.id_for_label }}" class="form-label">
                        {{ form.checkpoint_strategy.label }}
                    </label>
                    {{ form.checkpoint_strategy }}
                    <div class="form-text">{{ form.checkpoint_strategy.help_text }}</div>
                </div>
                <div class="col-md-4">
                    <label for="{{ form.checkpoint_monitor_metric.id_for_label }}" class="form-label">
                        {{ form.checkpoint_monitor_metric.label }}
                    </label>
                    {{ form.checkpoint_monitor_metric }}
                </div>
                <div class="col-md-4">
                    <label for="{{ form.max_checkpoints.id_for_label }}" class="form-label">
                        {{ form.max_checkpoints.label }}
                    </label>
                    {{ form.max_checkpoints }}
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-4">
                    <div class="form-check">
                        {{ form.save_optimizer_state }}
                        <label class="form-check-label" for="{{ form.save_optimizer_state.id_for_label }}">
                            {{ form.save_optimizer_state.label }}
                        </label>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="form-check">
                        {{ form.save_scheduler_state }}
                        <label class="form-check-label" for="{{ form.save_scheduler_state.id_for_label }}">
                            {{ form.save_scheduler_state.label }}
                        </label>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
// JavaScript to show/hide enhanced features
document.getElementById('{{ form.use_enhanced_training.id_for_label }}').addEventListener('change', function() {
    const section = document.getElementById('enhanced-features-section');
    section.style.display = this.checked ? 'block' : 'none';
});

// Show/hide custom loss configuration
document.getElementById('{{ form.loss_function.id_for_label }}').addEventListener('change', function() {
    const customConfig = document.getElementById('custom-loss-config');
    customConfig.style.display = this.value === 'custom' ? 'block' : 'none';
});

// Show/hide loss scheduler options
document.getElementById('{{ form.use_loss_scheduling.id_for_label }}').addEventListener('change', function() {
    const options = document.getElementById('loss-scheduler-options');
    options.style.display = this.checked ? 'block' : 'none';
});

// Show/hide focal loss parameters
document.getElementById('{{ form.use_focal_loss.id_for_label }}').addEventListener('change', function() {
    const params = document.getElementById('focal-loss-params');
    params.style.display = this.checked ? 'block' : 'none';
});

// Initialize visibility
document.addEventListener('DOMContentLoaded', function() {
    // Trigger change events to set initial visibility
    document.getElementById('{{ form.use_enhanced_training.id_for_label }}').dispatchEvent(new Event('change'));
    document.getElementById('{{ form.loss_function.id_for_label }}').dispatchEvent(new Event('change'));
});
</script>
```

### 4. Update train.py to Handle Enhanced Arguments

```python
# Update ml/training/train.py argument parser

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train ML models with enhanced features')
    
    # ... existing arguments ...
    
    # Enhanced training arguments
    parser.add_argument('--use-enhanced-training', action='store_true',
                       help='Use enhanced training features')
    parser.add_argument('--loss-function', type=str, default='combined_default',
                       help='Loss function type')
    parser.add_argument('--checkpoint-strategy', type=str, default='best',
                       help='Checkpoint saving strategy')
    parser.add_argument('--use-loss-scheduling', action='store_true',
                       help='Enable loss weight scheduling')
    parser.add_argument('--loss-scheduler-type', type=str, default='adaptive',
                       help='Type of loss scheduler')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    if args.use_enhanced_training:
        # Use enhanced training manager
        from ml.utils.training_manager import create_training_manager
        
        training_manager = create_training_manager(
            model_dir=model_dir,
            model_name=model_name,
            loss_preset=args.loss_function,
            checkpoint_preset=args.checkpoint_strategy
        )
        
        # Enhanced training loop
        # ... implementation ...
    else:
        # Use existing training code
        # ... existing implementation ...
```

## 🚀 Quick Integration Steps

To add enhanced features to the GUI:

1. **Add form fields** to `TrainingForm` in `forms.py`
2. **Update the view** in `views.py` to handle enhanced configuration
3. **Update the template** to show enhanced options
4. **Modify train.py** to use enhanced features when enabled

## 📊 Benefits of GUI Integration

Once integrated, users will have access to:

✅ **Advanced Loss Functions**: Dice+BCE combinations, focal loss, custom weights
✅ **Smart Checkpointing**: Multiple strategies, automatic cleanup, best model detection
✅ **Loss Scheduling**: Dynamic weight adjustment during training
✅ **Enhanced Monitoring**: Detailed metrics and progress tracking
✅ **Easy Configuration**: User-friendly interface for complex features

## 🔧 Current Workaround

Until the GUI integration is complete, you can:

1. **Use the command line** with enhanced features:
   ```bash
   python -m ml.utils.integration_example
   ```

2. **Modify training manually** by editing the train.py script to use enhanced features

3. **Create custom training scripts** using the enhanced training manager

The enhanced training system is fully functional and ready for GUI integration!
