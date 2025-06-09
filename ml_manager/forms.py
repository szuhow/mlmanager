from django import forms
from pathlib import Path
import importlib.util
import inspect
from .device_utils import get_device_choices, get_default_device, get_device_info_for_display

# Import the new architecture registry system
try:
    from shared.architecture_registry import registry as architecture_registry, get_available_models
except ImportError:
    # Fallback to legacy system if registry not available
    def get_available_models():
        base_dir = Path(__file__).parent.parent / 'shared'
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
            'use_random_flip', 'use_random_rotate', 'use_random_scale', 
            'use_random_intensity', 'crop_size', 'num_workers', 'is_default'
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
            'resolution': 'Training image resolution. Higher resolutions require more memory.',
            'device': 'Device to use for training. Auto will detect the best available device.',
            'use_random_flip': 'Apply random horizontal flip augmentation',
            'use_random_rotate': 'Apply random rotation augmentation',
            'use_random_scale': 'Apply random scaling augmentation',
            'use_random_intensity': 'Apply random intensity scaling augmentation',
            'crop_size': 'Size of random crop (pixels)',
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
    data_path = forms.CharField(help_text="Path to dataset directory")
    batch_size = forms.IntegerField(min_value=1, initial=32, help_text="Number of samples per batch")
    epochs = forms.IntegerField(min_value=1, initial=100, help_text="Number of training epochs")
    learning_rate = forms.FloatField(min_value=0.0, initial=0.001, help_text="Learning rate")
    validation_split = forms.FloatField(min_value=0.0, max_value=1.0, initial=0.2, help_text="Validation set size (0-1)")
    
    # Image resolution for training
    resolution = forms.ChoiceField(
        choices=RESOLUTION_CHOICES,
        initial='256',
        required=True,
        help_text="Training image resolution. Higher resolutions require more memory."
    )
    
    # Device selection for training
    device = forms.ChoiceField(
        choices=[],  # Will be set dynamically in __init__
        initial='auto',
        required=True,
        help_text="Device to use for training. Auto will detect the best available device."
    )
    
    # Augmentation options
    use_random_flip = forms.BooleanField(initial=True, required=False, help_text="Apply random horizontal flip")
    use_random_rotate = forms.BooleanField(initial=True, required=False, help_text="Apply random rotation")
    use_random_scale = forms.BooleanField(initial=True, required=False, help_text="Apply random scaling")
    use_random_intensity = forms.BooleanField(initial=True, required=False, help_text="Apply random intensity scaling")
    crop_size = forms.IntegerField(min_value=16, initial=128, help_text="Size of random crop (pixels)")
    num_workers = forms.IntegerField(min_value=0, initial=4, help_text="Number of data loading workers")
    
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

class InferenceForm(forms.Form):
    RESOLUTION_CHOICES = [
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
    resolution = forms.ChoiceField(
        choices=RESOLUTION_CHOICES,
        initial='original',
        required=True,
        help_text="Choose the input image resolution for processing"
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
