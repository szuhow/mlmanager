from django import forms
from pathlib import Path
import importlib.util
import inspect
import os
from .utils.device_utils import get_device_choices, get_default_device, get_device_info_for_display

# Get default values from environment variables
def get_env_default(key, default_value, value_type=int):
    """Get default value from environment variable with type conversion"""
    try:
        env_value = os.environ.get(key)
        if env_value is None:
            return default_value
        if value_type == int:
            return int(env_value)
        elif value_type == float:
            return float(env_value)
        else:
            return env_value
    except (ValueError, TypeError):
        return default_value

# Import the new architecture registry system
try:
    from ml.utils.architecture_registry import registry as architecture_registry, get_available_models
except ImportError:
    # Fallback to legacy system if registry not available
    def get_available_models():
        base_dir = Path(__file__).parent.parent.parent.parent / 'ml'
        models = []
        
        # Scan unet directory
        unet_dir = base_dir / 'unet'
        if unet_dir.exists():
            spec = importlib.util.spec_from_file_location("unet_model", str(unet_dir / "unet_model.py"))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                models.append(('unet', 'U-Net (PyTorch)'))
        
        # Scan unet-old directory
        unet_old_dir = base_dir / 'unet-old'
        if unet_old_dir.exists():
            spec = importlib.util.spec_from_file_location("unet_old", str(unet_old_dir / "unet.py"))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                models.append(('unet-old', 'U-Net (Legacy)'))
                
        return models

class TrainingTemplateForm(forms.ModelForm):
    class Meta:
        from .models import TrainingTemplate
        model = TrainingTemplate
        fields = [
            'name', 'description', 'model_type', 'batch_size', 'epochs', 
            'learning_rate', 'validation_split', 'resolution', 'device',
            'optimizer', 'lr_scheduler', 'lr_patience', 'lr_factor', 'lr_step_size', 'lr_gamma', 'min_lr',
            'use_early_stopping', 'early_stopping_patience', 'early_stopping_min_epochs', 
            'early_stopping_min_delta', 'early_stopping_metric',
            'use_random_flip', 'flip_probability', 'use_random_rotate', 'rotation_range',
            'use_random_scale', 'scale_range_min', 'scale_range_max', 
            'use_random_intensity', 'intensity_range', 'use_random_crop', 'crop_size',
            'use_pos_neg_cropping',
            'use_elastic_transform', 'elastic_alpha', 'elastic_sigma',
            'use_gaussian_noise', 'noise_std', 'num_workers', 'threshold', 'is_default'
        ]
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
            'learning_rate': forms.NumberInput(attrs={'step': 'any'}),
            'validation_split': forms.NumberInput(attrs={'step': 'any', 'min': 0, 'max': 1}),
        }
        help_texts = {
            'name': 'Unique name for this training template',
            'description': 'Optional description of this configuration',
            'model_type': 'Model architecture to use',
            'batch_size': 'Number of samples per batch',
            'epochs': 'Number of training epochs',
            'learning_rate': 'Learning rate for optimization',
            'validation_split': 'Validation set size (0-1)',
            'resolution': 'Training image crop size. Higher crop sizes require more memory.',
            'device': 'Device to use for training. Auto will detect the best available device.',
            'optimizer': 'Optimizer algorithm to use for training',
            'use_random_flip': 'Apply random horizontal flip augmentation',
            'flip_probability': 'Probability of applying flip (0.0-1.0)',
            'use_random_rotate': 'Apply random rotation augmentation',
            'rotation_range': 'Maximum rotation angle in degrees (±range)',
            'use_random_scale': 'Apply random scaling augmentation',
            'scale_range_min': 'Minimum scale factor',
            'scale_range_max': 'Maximum scale factor',
            'use_random_intensity': 'Apply random intensity scaling augmentation',
            'intensity_range': 'Intensity variation range (±range)',
            'use_random_crop': 'Apply random cropping for data augmentation',
            'crop_size': 'Size of random crop (pixels)',
            'use_pos_neg_cropping': 'Use advanced positive/negative region cropping for segmentation tasks',
            'use_elastic_transform': 'Apply elastic deformation for medical image augmentation',
            'elastic_alpha': 'Elastic transformation strength',
            'elastic_sigma': 'Elastic transformation smoothness',
            'use_gaussian_noise': 'Add Gaussian noise to simulate real-world conditions',
            'noise_std': 'Standard deviation of Gaussian noise',
            'num_workers': 'Number of data loading workers',
            'is_default': 'Make this the default template for new trainings',
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set model_type choices dynamically
        model_choices = get_available_models()
        self.fields['model_type'].choices = model_choices
        self.fields['model_type'].widget = forms.Select(choices=model_choices)
        
        # Set device choices dynamically based on system capabilities
        device_choices = [
            ('auto', 'Auto (CUDA if available, else CPU)'),
            ('cpu', 'CPU'),
        ]
        
        # Add CUDA options if available
        available_devices = get_device_choices()
        for device_value, device_label in available_devices:
            if device_value.startswith('cuda'):
                device_choices.append((device_value, device_label))
        
        self.fields['device'].choices = device_choices
        self.fields['device'].widget = forms.Select(choices=device_choices)
        
        # Add Bootstrap classes
        for field_name, field in self.fields.items():
            if isinstance(field.widget, forms.CheckboxInput):
                field.widget.attrs.update({'class': 'form-check-input'})
            elif isinstance(field.widget, forms.Textarea):
                field.widget.attrs.update({'class': 'form-control', 'rows': 3})
            else:
                field.widget.attrs.update({'class': 'form-control'})

class TrainingForm(forms.Form):
    RESOLUTION_CHOICES = [
        ('original', 'Original Size'),
        ('128', '128 x 128 pixels'),
        ('256', '256 x 256 pixels'),
        ('384', '384 x 384 pixels'),
        ('512', '512 x 512 pixels'),
    ]
    
    # Add template selection field at the top
    template = forms.ModelChoiceField(
        queryset=None,  # Will be set in __init__
        required=False,
        empty_label="Select a template (optional)",
        help_text="Choose a pre-configured template or fill manually",
        widget=forms.Select(attrs={'class': 'form-control', 'id': 'template-select'})
    )
    
    name = forms.CharField(max_length=200, help_text="Name of the training run")
    description = forms.CharField(widget=forms.Textarea, required=False, help_text="Description of the training run")
    model_type = forms.ChoiceField(choices=[], help_text="Model architecture to use")  # Will be set dynamically in __init__
    data_path = forms.CharField(
        initial="/app/data/datasets/", 
        help_text="Path to dataset directory. Use '/app/data/datasets/' for ARCADE or '/app/data/datasets/basic' for basic"
    )
    
    # Dataset type selection
    dataset_type = forms.ChoiceField(
        choices=[
            ('auto', 'Auto-detect dataset type'),
            ('coronary', 'Standard Coronary Dataset'),
            ('arcade_binary', 'ARCADE Binary Segmentation'),
            ('arcade_semantic', 'ARCADE Semantic Segmentation'),
            ('arcade_stenosis', 'ARCADE Stenosis Detection'),
            ('arcade_classification', 'ARCADE Artery Classification')
        ],
        initial='auto',
        required=True,
        help_text="Type of dataset to use for training"
    )
    
    batch_size = forms.IntegerField(
        min_value=1, 
        initial=get_env_default('DEFAULT_BATCH_SIZE', 8), 
        help_text="Number of samples per batch. Larger batches require more memory."
    )
    epochs = forms.IntegerField(min_value=1, initial=10, help_text="Number of training epochs")
    learning_rate = forms.FloatField(min_value=0.0, initial=0.001, help_text="Learning rate")
    validation_split = forms.FloatField(min_value=0.0, max_value=1.0, initial=0.2, help_text="Validation set size (0-1)")
    
    # Image resolution for training
    resolution = forms.ChoiceField(
        choices=RESOLUTION_CHOICES,
        initial='256',
        required=True,
        help_text="Training image crop size. Higher crop sizes require more memory."
    )
    
    # Device selection for training
    device = forms.ChoiceField(
        choices=[],  # Will be set dynamically in __init__
        initial='auto',
        required=True,
        help_text="Device to use for training. Auto will detect the best available device."
    )
    
    # Optimizer selection
    OPTIMIZER_CHOICES = [
        ('adam', 'Adam'),
        ('sgd', 'SGD'),
        ('rmsprop', 'RMSprop'),
        ('adamw', 'AdamW'),
    ]
    optimizer = forms.ChoiceField(
        choices=OPTIMIZER_CHOICES,
        initial='adam',
        required=True,
        help_text="Optimizer algorithm to use for training"
    )
    
    # Learning Rate Scheduler options
    LR_SCHEDULER_CHOICES = [
        ('none', 'No Scheduler'),
        ('plateau', 'ReduceLROnPlateau'),
        ('step', 'StepLR'),
        ('exponential', 'ExponentialLR'),
        ('cosine', 'CosineAnnealingLR'),
        ('adaptive', 'Adaptive (Custom)'),
    ]
    
    lr_scheduler = forms.ChoiceField(
        choices=LR_SCHEDULER_CHOICES,
        initial='plateau',
        required=True,
        help_text="Learning rate scheduling strategy"
    )
    
    # Scheduler-specific parameters
    lr_patience = forms.IntegerField(
        min_value=1, 
        initial=5, 
        required=False,
        help_text="Epochs to wait before reducing LR (for plateau scheduler)"
    )
    
    lr_factor = forms.FloatField(
        min_value=0.01, 
        max_value=0.99, 
        initial=0.5, 
        required=False,
        help_text="Factor to reduce LR by (for plateau scheduler)"
    )
    
    lr_step_size = forms.IntegerField(
        min_value=1, 
        initial=10, 
        required=False,
        help_text="Epochs between LR reductions (for step scheduler)"
    )
    
    lr_gamma = forms.FloatField(
        min_value=0.01, 
        max_value=0.99, 
        initial=0.1, 
        required=False,
        help_text="Multiplicative factor for LR decay"
    )
    
    min_lr = forms.FloatField(
        min_value=1e-8, 
        initial=1e-7, 
        required=False,
        help_text="Minimum learning rate threshold"
    )
    
    # Early Stopping Configuration
    use_early_stopping = forms.BooleanField(
        initial=False, 
        required=False, 
        help_text="Enable early stopping to prevent overfitting"
    )
    
    early_stopping_patience = forms.IntegerField(
        min_value=1, 
        initial=10, 
        required=False,
        help_text="Number of epochs to wait for improvement before stopping"
    )
    
    early_stopping_min_epochs = forms.IntegerField(
        min_value=1, 
        initial=20, 
        required=False,
        help_text="Minimum number of epochs before early stopping can occur"
    )
    
    early_stopping_min_delta = forms.FloatField(
        min_value=0.0, 
        initial=1e-4, 
        required=False,
        help_text="Minimum improvement required to reset patience counter"
    )
    
    EARLY_STOPPING_METRIC_CHOICES = [
        ('val_dice', 'Validation Dice Score'),
        ('val_loss', 'Validation Loss'),
        ('val_accuracy', 'Validation Accuracy'),
    ]
    
    early_stopping_metric = forms.ChoiceField(
        choices=EARLY_STOPPING_METRIC_CHOICES,
        initial='val_dice',
        required=False,
        help_text="Metric to monitor for early stopping decisions"
    )
    
    # Enhanced Augmentation options with richer controls
    use_random_flip = forms.BooleanField(initial=True, required=False, help_text="Apply random horizontal flip to increase data diversity")
    flip_probability = forms.FloatField(min_value=0.0, max_value=1.0, initial=0.5, required=False, help_text="Probability of applying flip (0.0-1.0)")
    
    use_random_rotate = forms.BooleanField(initial=True, required=False, help_text="Apply random rotation to images")
    rotation_range = forms.IntegerField(min_value=0, max_value=180, initial=30, required=False, help_text="Maximum rotation angle in degrees (±range)")
    
    use_random_scale = forms.BooleanField(initial=True, required=False, help_text="Apply random scaling/zoom to images")
    scale_range_min = forms.FloatField(min_value=0.1, max_value=2.0, initial=0.8, required=False, help_text="Minimum scale factor")
    scale_range_max = forms.FloatField(min_value=0.1, max_value=2.0, initial=1.2, required=False, help_text="Maximum scale factor")
    
    use_random_intensity = forms.BooleanField(initial=True, required=False, help_text="Apply random intensity/brightness adjustments")
    intensity_range = forms.FloatField(min_value=0.0, max_value=1.0, initial=0.2, required=False, help_text="Intensity variation range (±range)")
    
    use_random_crop = forms.BooleanField(initial=False, required=False, help_text="Apply random cropping for data augmentation")
    # Note: crop_size is automatically determined from resolution field
    crop_size = forms.IntegerField(
        min_value=64,
        max_value=1024,
        initial=256,
        required=False,
        widget=forms.HiddenInput(),
        help_text="Size of crops for training (pixels). Automatically set from resolution."
    )
    
    # Advanced cropping for segmentation
    use_pos_neg_cropping = forms.BooleanField(
        initial=False, 
        required=False, 
        help_text="Use advanced positive/negative region cropping for segmentation tasks (RandCropByPosNegLabeld)"
    )
    
    use_elastic_transform = forms.BooleanField(initial=False, required=False, help_text="Apply elastic deformation for medical image augmentation")
    elastic_alpha = forms.FloatField(min_value=0.0, max_value=100.0, initial=34.0, required=False, help_text="Elastic transformation strength")
    elastic_sigma = forms.FloatField(min_value=0.0, max_value=10.0, initial=4.0, required=False, help_text="Elastic transformation smoothness")
    
    use_gaussian_noise = forms.BooleanField(initial=False, required=False, help_text="Add Gaussian noise to simulate real-world conditions")
    noise_std = forms.FloatField(min_value=0.0, max_value=0.1, initial=0.01, required=False, help_text="Standard deviation of Gaussian noise")
    
    num_workers = forms.IntegerField(
        min_value=0, 
        initial=get_env_default('DEFAULT_NUM_WORKERS', 1), 
        help_text="Number of data loading workers. Reduce if you get memory errors."
    )
    
    # Enhanced Training Features
    LOSS_FUNCTION_CHOICES = [
        ('bce', 'Binary Cross Entropy'),
        ('dice', 'Dice Loss'),
        ('combined', 'Combined Dice + BCE (Recommended)'),
        ('focal', 'Focal Loss'),
        ('focal_segmentation', 'Focal Segmentation (Dice + Focal BCE)'),
        ('balanced_segmentation', 'Balanced Segmentation'),
        ('dice_focused', 'Dice Focused'),
        ('jaccard_based', 'Jaccard Based'),
    ]
    
    loss_function = forms.ChoiceField(
        choices=LOSS_FUNCTION_CHOICES,
        initial='dice',
        required=True,
        help_text="Loss function for training. Dice Loss is recommended for segmentation."
    )
    
    # Loss function weights (for combined losses)
    dice_weight = forms.FloatField(
        min_value=0.0, 
        max_value=1.0, 
        initial=0.7, 
        required=False,
        widget=forms.NumberInput(attrs={'step': '0.1', 'class': 'form-control'}),
        help_text="Weight for Dice loss component (0.0-1.0). Higher values focus more on shape accuracy."
    )
    
    bce_weight = forms.FloatField(
        min_value=0.0, 
        max_value=1.0, 
        initial=0.3, 
        required=False,
        widget=forms.NumberInput(attrs={'step': '0.1', 'class': 'form-control'}),
        help_text="Weight for BCE loss component (0.0-1.0). Higher values focus more on pixel accuracy."
    )
    
    # Loss scheduling
    use_loss_scheduling = forms.BooleanField(
        initial=False, 
        required=False,
        help_text="Enable dynamic loss weight adjustment during training"
    )
    
    LOSS_SCHEDULER_CHOICES = [
        ('adaptive', 'Adaptive (Adjust based on performance)'),
        ('cosine', 'Cosine Annealing'),
        ('step', 'Step-based'),
        ('performance', 'Performance-based'),
    ]
    
    loss_scheduler_type = forms.ChoiceField(
        choices=LOSS_SCHEDULER_CHOICES,
        initial='adaptive',
        required=False,
        help_text="Type of loss weight scheduling to use"
    )
    
    # Enhanced Checkpointing
    CHECKPOINT_STRATEGY_CHOICES = [
        ('best', 'Best Model Only (Recommended)'),
        ('epoch', 'Every Epoch'),
        ('interval', 'Every N Epochs'),
        ('all', 'Best + Regular Checkpoints'),
        ('adaptive', 'Adaptive Strategy'),
        ('performance_based', 'Performance Based'),
    ]
    
    checkpoint_strategy = forms.ChoiceField(
        choices=CHECKPOINT_STRATEGY_CHOICES,
        initial='best',
        required=True,
        help_text="Checkpoint saving strategy. 'Best' saves only when performance improves."
    )
    
    max_checkpoints = forms.IntegerField(
        min_value=1, 
        max_value=20, 
        initial=5, 
        required=True,
        help_text="Maximum number of checkpoints to keep. Older checkpoints are automatically removed."
    )
    
    MONITOR_METRIC_CHOICES = [
        ('val_dice', 'Validation Dice Score (Recommended)'),
        ('val_loss', 'Validation Loss'),
        ('val_accuracy', 'Validation Accuracy'),
        ('val_iou', 'Validation IoU'),
    ]
    
    monitor_metric = forms.ChoiceField(
        choices=MONITOR_METRIC_CHOICES,
        initial='val_dice',
        required=True,
        help_text="Metric to monitor for best model selection"
    )
    
    # Advanced Training Options
    use_enhanced_training = forms.BooleanField(
        initial=True,
        required=False,
        help_text="Enable enhanced training features (checkpointing, loss scheduling, detailed monitoring)"
    )
    
    use_mixed_precision = forms.BooleanField(
        initial=False,
        required=False,
        help_text="Enable mixed precision training for faster training and lower memory usage (requires CUDA GPU - automatically disabled on CPU)"
    )
    
    # Post-processing configuration
    threshold = forms.FloatField(
        initial=0.5,
        min_value=0.1,
        max_value=0.9,
        widget=forms.NumberInput(attrs={'step': '0.1', 'class': 'form-control'}),
        required=False,
        label="Binary Segmentation Threshold",
        help_text="Threshold for converting soft predictions to hard binary masks (0.5 is standard)"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Import here to avoid circular imports
        from .models import TrainingTemplate
        
        # Set template queryset
        self.fields['template'].queryset = TrainingTemplate.objects.all()
        
        # Set model_type choices dynamically
        model_choices = get_available_models()
        self.fields['model_type'].choices = model_choices
        
        # Set device choices dynamically based on system capabilities
        device_choices = [
            ('auto', 'Auto (CUDA if available, else CPU)'),
            ('cpu', 'CPU'),
        ]
        
        # Add CUDA options if available
        available_devices = get_device_choices()
        for device_value, device_label in available_devices:
            if device_value.startswith('cuda'):
                device_choices.append((device_value, device_label))
        
        self.fields['device'].choices = device_choices
        self.fields['device'].widget = forms.Select(choices=device_choices)
        self.fields['device'].initial = get_default_device() if get_default_device() != 'cpu' else 'auto'
        
        # Try to set default template values
        try:
            default_template = TrainingTemplate.objects.filter(is_default=True).first()
            if default_template and not kwargs.get('data'):  # Only set defaults if form is not bound
                template_data = default_template.get_form_data()
                for field_name, value in template_data.items():
                    if field_name in self.fields:
                        self.fields[field_name].initial = value
        except:
            pass  # Handle case where table doesn't exist yet
    
    def clean(self):
        cleaned_data = super().clean()
        resolution = cleaned_data.get('resolution')
        
        # Auto-set crop_size based on resolution
        if resolution and resolution.isdigit():
            cleaned_data['crop_size'] = int(resolution)
        elif resolution == 'original':
            # For original size, use a default crop size
            cleaned_data['crop_size'] = 512
        else:
            # Fallback default
            cleaned_data['crop_size'] = 256
                
        return cleaned_data

class InferenceForm(forms.Form):
    CROP_SIZE_CHOICES = [
        ('original', 'Original Size'),
        ('128', '128 x 128 pixels'),
        ('256', '256 x 256 pixels'),
        ('384', '384 x 384 pixels'),
        ('512', '512 x 512 pixels'),
    ]
    
    model_id = forms.ModelChoiceField(
        queryset=None,  # Will be set in __init__
        required=True,
        help_text="Select a trained model for inference"
    )
    image = forms.ImageField(
        help_text="Upload an image for segmentation",
        required=True
    )
    crop_size = forms.ChoiceField(
        choices=CROP_SIZE_CHOICES,
        initial='original',
        required=True,
        help_text="Choose the input image crop size for processing"
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Import here to avoid circular imports
        from .models import MLModel
        # Set queryset to only completed models
        self.fields['model_id'].queryset = MLModel.objects.filter(status='completed').order_by('-created_at')
        
        # Add Bootstrap classes
        for field_name, field in self.fields.items():
            if isinstance(field.widget, forms.Select):
                field.widget.attrs.update({'class': 'form-select'})
            elif isinstance(field.widget, forms.FileInput):
                field.widget.attrs.update({'class': 'form-control'})
            else:
                field.widget.attrs.update({'class': 'form-control'})

class EnhancedInferenceForm(forms.Form):
    """Enhanced inference form with post-processing options."""
    
    # Model selection
    model_id = forms.ChoiceField(
        choices=[],  # Will be populated dynamically
        required=True,
        label="Model",
        help_text="Select the trained model to use for inference"
    )
    
    checkpoint_path = forms.ChoiceField(
        choices=[],  # Will be populated dynamically
        required=False,
        label="Checkpoint (Optional)",
        help_text="Select specific checkpoint, or leave empty to use the best model"
    )
    
    # Image upload
    image = forms.FileField(
        widget=forms.FileInput(attrs={'accept': 'image/*'}),
        required=True,
        label="Image",
        help_text="Upload image for inference"
    )
    
    # Image resolution
    RESOLUTION_CHOICES = [
        (256, '256x256'),
        (512, '512x512'),
        (1024, '1024x1024'),
    ]
    
    resolution = forms.ChoiceField(
        choices=RESOLUTION_CHOICES,
        initial=512,
        label="Resolution",
        help_text="Choose the input image resolution for processing"
    )
    
    # Post-processing configuration
    threshold = forms.FloatField(
        initial=0.5,
        min_value=0.1,
        max_value=0.9,
        widget=forms.NumberInput(attrs={'step': '0.1', 'class': 'form-control'}),
        required=False,
        label="Binary Segmentation Threshold",
        help_text="Threshold for converting soft predictions to hard binary masks (0.5 is standard)"
    )

    def __init__(self, *args, **kwargs):
        # Extract model_id for model-specific forms
        model_id = kwargs.pop('model_id', None)
        all_models = kwargs.pop('all_models', False)
        super().__init__(*args, **kwargs)
        
        # Populate model choices
        from .models import MLModel
        
        if model_id:
            # Single model - get only its checkpoints
            try:
                model = MLModel.objects.get(id=model_id)
                self.fields['model_id'].choices = [(model.id, model.name)]
                self.fields['model_id'].initial = model.id
                self.fields['model_id'].widget.attrs['readonly'] = True
            except MLModel.DoesNotExist:
                self.fields['model_id'].choices = []
        elif all_models:
            # All models - populate with completed models
            models = MLModel.objects.filter(status='completed')
            self.fields['model_id'].choices = [(m.id, f"{m.name} (ID: {m.id})") for m in models]
        else:
            self.fields['model_id'].choices = []


class TrainingConfigForm(forms.Form):
    """Enhanced training configuration form with loss function options."""
    
    # Basic training parameters
    epochs = forms.IntegerField(
        initial=100,
        min_value=1,
        max_value=1000,
        label="Epochs",
        help_text="Number of training epochs"
    )
    
    batch_size = forms.IntegerField(
        initial=8,
        min_value=1,
        max_value=64,
        label="Batch Size",
        help_text="Number of samples per batch"
    )
    
    learning_rate = forms.FloatField(
        initial=0.001,
        min_value=1e-6,
        max_value=1.0,
        widget=forms.NumberInput(attrs={'step': 'any'}),
        label="Learning Rate",
        help_text="Initial learning rate"
    )
    
    # Enhanced loss function options
    LOSS_CHOICES = [
        ('bce', 'Binary Cross Entropy'),
        ('dice', 'Dice Loss'),
        ('combined_dice_focal', 'Combined Dice + Focal Loss'),
        ('tversky', 'Tversky Loss'),
        ('focal', 'Focal Loss')
    ]
    
    loss_function = forms.ChoiceField(
        choices=LOSS_CHOICES,
        initial='combined_dice_focal',
        label="Loss Function",
        help_text="Loss function optimized for segmentation and class imbalance"
    )
    
    # Loss function parameters
    dice_weight = forms.FloatField(
        initial=0.7,
        min_value=0.0,
        max_value=1.0,
        widget=forms.NumberInput(attrs={'step': '0.1'}),
        label="Dice Loss Weight",
        help_text="Weight for Dice loss in combined loss (higher = more segmentation focus)"
    )
    
    focal_alpha = forms.FloatField(
        initial=0.25,
        min_value=0.01,
        max_value=1.0,
        widget=forms.NumberInput(attrs={'step': '0.01'}),
        label="Focal Alpha",
        help_text="Alpha parameter for Focal loss (class weighting)"
    )
    
    focal_gamma = forms.FloatField(
        initial=2.0,
        min_value=0.5,
        max_value=5.0,
        widget=forms.NumberInput(attrs={'step': '0.5'}),
        label="Focal Gamma",
        help_text="Gamma parameter for Focal loss (focus on hard examples)"
    )
    
    # Regularization
    weight_decay = forms.FloatField(
        initial=1e-4,
        min_value=0.0,
        max_value=1e-2,
        widget=forms.NumberInput(attrs={'step': 'any'}),
        label="Weight Decay",
        help_text="L2 regularization strength"
    )
    
    dropout_rate = forms.FloatField(
        initial=0.1,
        min_value=0.0,
        max_value=0.5,
        widget=forms.NumberInput(attrs={'step': '0.05'}),
        label="Dropout Rate",
        help_text="Dropout probability for regularization"
    )
    
    # Early stopping
    early_stopping_patience = forms.IntegerField(
        initial=15,
        min_value=5,
        max_value=50,
        label="Early Stopping Patience",
        help_text="Number of epochs to wait before stopping if no improvement"
    )
    
    METRIC_CHOICES = [
        ('val_dice_score', 'Validation Dice Score'),
        ('val_loss', 'Validation Loss'),
        ('val_accuracy', 'Validation Accuracy'),
        ('val_iou', 'Validation IoU')
    ]
    
    early_stopping_metric = forms.ChoiceField(
        choices=METRIC_CHOICES,
        initial='val_dice_score',
        label="Early Stopping Metric",
        help_text="Metric to monitor for early stopping"
    )
    
    # Data augmentation for noise robustness
    augmentation_probability = forms.FloatField(
        initial=0.5,
        min_value=0.0,
        max_value=1.0,
        widget=forms.NumberInput(attrs={'step': '0.1'}),
        label="Augmentation Probability",
        help_text="Probability of applying data augmentation (helps with noise robustness)"
    )

# Update the original InferenceForm to inherit from EnhancedInferenceForm
class InferenceForm(EnhancedInferenceForm):
    """Backward compatibility alias for InferenceForm."""
    pass
