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
            'use_gaussian_noise', 'noise_std', 'num_workers', 'threshold', 'is_default',
            # Medical preprocessing fields - only include fields that exist in model
            'use_medical_preprocessing', 'preprocessing_type', 'clahe_clip_limit', 'clahe_tile_size',
            'use_unsharp_masking', 'unsharp_amount', 'unsharp_radius',
            'use_frangi_filter', 'frangi_sigma_min', 'frangi_sigma_max', 'frangi_sigma_step',
            'use_denoising', 'noise_reduction_sigma', 'use_histogram_equalization',
            'normalize_intensity', 'gamma_correction', 'custom_preprocessing_pipeline'
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
            # Medical preprocessing help texts
            'use_medical_preprocessing': 'Enable advanced medical image preprocessing',
            'preprocessing_type': 'Type of medical imaging modality for optimal preprocessing',
            'clahe_clip_limit': 'CLAHE clip limit for contrast enhancement (1.0-8.0)',
            'clahe_tile_size': 'CLAHE tile grid size (4-16)',
            'use_unsharp_masking': 'Enable unsharp masking for edge enhancement',
            'unsharp_amount': 'Unsharp masking strength (0.5-2.0)',
            'unsharp_radius': 'Unsharp masking radius (0.5-3.0)',
            'use_frangi_filter': 'Enable Frangi vesselness filter for vessel enhancement',
            'frangi_sigma_min': 'Minimum sigma for Frangi filter (detects thin vessels)',
            'frangi_sigma_max': 'Maximum sigma for Frangi filter (detects thick vessels)',
            'frangi_sigma_step': 'Step size for sigma range (1.0-3.0)',
            'use_denoising': 'Enable denoising filters to reduce image noise',
            'noise_reduction_sigma': 'Noise reduction strength (0.5-3.0)',
            'use_histogram_equalization': 'Enable histogram equalization for global contrast',
            'normalize_intensity': 'Normalize image intensity to standard range',
            'gamma_correction': 'Gamma correction (0.5-2.0, 1.0=no correction)',
            'custom_preprocessing_pipeline': 'Custom preprocessing pipeline (comma-separated)',
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
    
    # Dataset type selection - ARCADE support with all 6 task types
    dataset_type = forms.ChoiceField(
        choices=[
            ('auto', 'Auto-detect dataset type'),
            ('coronary', 'Standard Coronary Dataset'),
            ('arcade_binary', 'ARCADE: Binary Segmentation (image → binary mask)'),
            ('arcade_semantic_segmentation', 'ARCADE: Semantic Segmentation (image → multi-class mask)'),
            ('arcade_stenosis_detection', 'ARCADE: Stenosis Detection (image → bounding boxes)'),
            ('arcade_artery_classification', 'ARCADE: Artery Classification (binary mask → left/right)'),
            ('arcade_semantic_seg_binary', 'ARCADE: Semantic from Binary (binary mask → multi-class)'),
            ('arcade_stenosis_segmentation', 'ARCADE: Stenosis Segmentation (image → stenosis mask)')
        ],
        initial='auto',
        required=True,
        help_text="Type of dataset to use for training. ARCADE tasks support different input/output combinations."
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
        # Advanced loss functions from pywick
        ('tversky_recall', 'Tversky Loss (Recall-focused) - Fewer missed arteries'),
        ('tversky_precision', 'Tversky Loss (Precision-focused) - Cleaner segmentations'),
        ('focal_advanced', 'Advanced Focal Loss - Better class imbalance handling'),
        ('combo_dice_bce_focal', 'Combined Dice + Focal BCE - Advanced combination'),
        ('boundary_aware', 'Boundary-aware Loss - Improved edge detection'),
        ('weighted_bce_adaptive', 'Adaptive Weighted BCE - Dynamic class balancing'),
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
    
    # Medical Preprocessing Configuration
    use_medical_preprocessing = forms.BooleanField(
        initial=False,
        required=False,
        help_text="Enable advanced medical image preprocessing for better results with angiography and coronary images"
    )
    
    PREPROCESSING_TYPE_CHOICES = [
        ('angiography', 'Angiography (X-ray coronary images)'),
        ('ct_coronary', 'CT Coronary Angiography'),
        ('oct_coronary', 'OCT Coronary Images'),
        ('general', 'General Medical Images'),
    ]
    
    preprocessing_type = forms.ChoiceField(
        choices=PREPROCESSING_TYPE_CHOICES,
        initial='angiography',
        required=False,
        help_text="Type of medical imaging modality for optimal preprocessing"
    )
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe_clip_limit = forms.FloatField(
        min_value=1.0,
        max_value=8.0,
        initial=3.0,
        required=False,
        widget=forms.NumberInput(attrs={'step': '0.5', 'class': 'form-control'}),
        help_text="CLAHE clip limit for contrast enhancement (1.0-8.0). Higher values = stronger contrast."
    )
    
    clahe_tile_size = forms.IntegerField(
        min_value=4,
        max_value=16,
        initial=8,
        required=False,
        help_text="CLAHE tile grid size (4-16). Smaller tiles = more local contrast enhancement."
    )
    
    # Unsharp Masking for edge enhancement
    use_unsharp_masking = forms.BooleanField(
        initial=False,
        required=False,
        help_text="Enable unsharp masking for edge enhancement and vessel definition"
    )
    
    unsharp_amount = forms.FloatField(
        min_value=0.5,
        max_value=2.0,
        initial=1.0,
        required=False,
        widget=forms.NumberInput(attrs={'step': '0.1', 'class': 'form-control'}),
        help_text="Unsharp masking strength (0.5-2.0). Higher values = stronger edge enhancement."
    )
    
    unsharp_radius = forms.FloatField(
        min_value=0.5,
        max_value=3.0,
        initial=1.0,
        required=False,
        widget=forms.NumberInput(attrs={'step': '0.1', 'class': 'form-control'}),
        help_text="Unsharp masking radius (0.5-3.0). Controls the size of details enhanced."
    )
    
    # Frangi vesselness filter
    use_frangi_filter = forms.BooleanField(
        initial=False,
        required=False,
        help_text="Enable Frangi vesselness filter for enhanced vessel detection in angiography"
    )
    
    frangi_sigma_min = forms.FloatField(
        min_value=0.5,
        max_value=5.0,
        initial=1.0,
        required=False,
        widget=forms.NumberInput(attrs={'step': '0.5', 'class': 'form-control'}),
        help_text="Minimum sigma for Frangi filter (0.5-5.0). Detects thin vessels."
    )
    
    frangi_sigma_max = forms.FloatField(
        min_value=5.0,
        max_value=20.0,
        initial=10.0,
        required=False,
        widget=forms.NumberInput(attrs={'step': '1.0', 'class': 'form-control'}),
        help_text="Maximum sigma for Frangi filter (5.0-20.0). Detects thick vessels."
    )
    
    frangi_sigma_step = forms.FloatField(
        min_value=1.0,
        max_value=3.0,
        initial=2.0,
        required=False,
        widget=forms.NumberInput(attrs={'step': '0.5', 'class': 'form-control'}),
        help_text="Step size for sigma range (1.0-3.0). Smaller steps = more precise detection."
    )
    
    # Denoising
    use_denoising = forms.BooleanField(
        initial=False,
        required=False,
        help_text="Enable denoising filters to reduce image noise"
    )
    
    noise_reduction_sigma = forms.FloatField(
        min_value=0.5,
        max_value=3.0,
        initial=1.0,
        required=False,
        widget=forms.NumberInput(attrs={'step': '0.1', 'class': 'form-control'}),
        help_text="Noise reduction strength (0.5-3.0). Higher values = stronger denoising."
    )
    
    # Histogram equalization
    use_histogram_equalization = forms.BooleanField(
        initial=False,
        required=False,
        help_text="Enable histogram equalization for global contrast improvement"
    )
    
    # Intensity normalization
    normalize_intensity = forms.BooleanField(
        initial=True,
        required=False,
        help_text="Normalize image intensity to standard range for consistent processing"
    )
    
    # Gamma correction
    gamma_correction = forms.FloatField(
        min_value=0.5,
        max_value=2.0,
        initial=1.0,
        required=False,
        widget=forms.NumberInput(attrs={'step': '0.1', 'class': 'form-control'}),
        help_text="Gamma correction (0.5-2.0). 1.0 = no correction, <1.0 = brighter, >1.0 = darker."
    )
    
    # Custom preprocessing pipeline
    custom_preprocessing_pipeline = forms.CharField(
        max_length=500,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., clahe,unsharp,frangi,denoise'}),
        help_text="Custom preprocessing pipeline (comma-separated): clahe, unsharp, frangi, denoise, histogram_eq"
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
    
    confidence_threshold = forms.FloatField(
        initial=0.5,
        min_value=0.1,
        max_value=0.9,
        widget=forms.NumberInput(attrs={'step': '0.1', 'class': 'form-control'}),
        required=False,
        label="Confidence Threshold",
        help_text="Minimum confidence score for accepting predictions"
    )
    
    min_component_size = forms.IntegerField(
        initial=100,
        min_value=1,
        max_value=10000,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        required=False,
        label="Minimum Component Size",
        help_text="Remove connected components smaller than this size (in pixels)"
    )
    
    morphology_kernel_size = forms.IntegerField(
        initial=3,
        min_value=1,
        max_value=15,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        required=False,
        label="Morphology Kernel Size",
        help_text="Size of morphological operations kernel (odd numbers only)"
    )
    
    apply_opening = forms.BooleanField(
        initial=True,
        required=False,
        label="Apply Opening",
        help_text="Remove small noise objects using morphological opening"
    )
    
    apply_closing = forms.BooleanField(
        initial=True,
        required=False,
        label="Apply Closing",
        help_text="Fill small holes using morphological closing"
    )
    
    use_adaptive_threshold = forms.BooleanField(
        initial=False,
        required=False,
        label="Use Adaptive Threshold",
        help_text="Use adaptive thresholding instead of fixed threshold"
    )
    
    # Test Time Augmentation (TTA) options
    use_tta = forms.BooleanField(
        initial=False,
        required=False,
        label="Enable Test Time Augmentation",
        help_text="Apply augmentations and average results for better accuracy (slower but more accurate)"
    )
    
    tta_flip_horizontal = forms.BooleanField(
        initial=True,
        required=False,
        label="TTA: Horizontal Flip",
        help_text="Include horizontal flip in test time augmentation"
    )
    
    tta_flip_vertical = forms.BooleanField(
        initial=True,
        required=False,
        label="TTA: Vertical Flip", 
        help_text="Include vertical flip in test time augmentation"
    )
    
    tta_rotate_90 = forms.BooleanField(
        initial=True,
        required=False,
        label="TTA: 90° Rotations",
        help_text="Include 90°, 180°, 270° rotations in test time augmentation"
    )
    
    tta_scale = forms.BooleanField(
        initial=False,
        required=False,
        label="TTA: Multi-scale",
        help_text="Include different scales in test time augmentation (experimental)"
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
