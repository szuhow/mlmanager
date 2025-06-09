from django.db import models
from django.utils import timezone
from django.core.serializers.json import DjangoJSONEncoder


def default_dict():
    """Helper function for JSONField default values"""
    return {}

class MLModel(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    # Enhanced identification fields
    version = models.CharField(max_length=50, default="1.0.0", help_text="Semantic version (e.g., 1.0.0)")
    unique_identifier = models.CharField(max_length=100, unique=True, blank=True, help_text="Unique model identifier")
    model_family = models.CharField(max_length=100, blank=True, help_text="Model family/category (e.g., UNet-Coronary)")
    model_type = models.CharField(max_length=50, blank=True, help_text="Model architecture type (e.g., unet, unet-old)")
    
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    mlflow_run_id = models.CharField(max_length=50, unique=True, null=True, blank=True)
    
    # MLflow Model Registry fields
    registry_model_name = models.CharField(max_length=200, blank=True, default='', help_text="Model name in MLflow Registry")
    registry_model_version = models.CharField(max_length=50, blank=True, default='', help_text="Model version in Registry")
    registry_stage = models.CharField(
        max_length=20,
        choices=[
            ('None', 'None'),
            ('Staging', 'Staging'),
            ('Production', 'Production'),
            ('Archived', 'Archived')
        ],
        default='None',
        help_text="Model stage in Registry"
    )
    is_registered = models.BooleanField(default=False, help_text="Whether model is registered in MLflow Registry")
    
    # Enhanced storage and organization
    model_directory = models.CharField(max_length=500, blank=True, help_text="Path to organized model directory")
    model_weights_path = models.CharField(max_length=500, blank=True, help_text="Path to model weights file")
    model_config_path = models.CharField(max_length=500, blank=True, help_text="Path to model config file")
    
    # Enhanced metadata fields
    training_data_info = models.JSONField(default=default_dict, encoder=DjangoJSONEncoder, help_text="Training dataset metadata")
    model_architecture_info = models.JSONField(default=default_dict, encoder=DjangoJSONEncoder, help_text="Model architecture details")
    performance_metrics = models.JSONField(default=default_dict, encoder=DjangoJSONEncoder, help_text="Detailed performance metrics")
    
    status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Pending'),
            ('loading', 'Loading Dataset'),
            ('training', 'Training'),
            ('completed', 'Completed'),
            ('failed', 'Failed'),
            ('archived', 'Archived'),
            ('deployed', 'Deployed')
        ],
        default='pending'
    )
    current_epoch = models.IntegerField(default=0)
    total_epochs = models.IntegerField(default=0)
    train_loss = models.FloatField(default=0.0)
    val_loss = models.FloatField(default=0.0)
    train_dice = models.FloatField(default=0.0)
    val_dice = models.FloatField(default=0.0)
    best_val_dice = models.FloatField(default=0.0)
    stop_requested = models.BooleanField(default=False)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['model_family', 'version']),
            models.Index(fields=['status', 'created_at']),
        ]

    def save(self, *args, **kwargs):
        # Generate unique identifier if not provided
        if not self.unique_identifier:
            import uuid
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.unique_identifier = f"{self.model_family or 'model'}_{timestamp}_{str(uuid.uuid4())[:8]}"
        super().save(*args, **kwargs)

    def get_model_display_name(self):
        """Get a user-friendly display name for the model"""
        if self.model_family:
            return f"{self.model_family} v{self.version}"
        return f"{self.name} v{self.version}"

    def get_organized_model_path(self):
        """Get the organized path where model should be stored"""
        import os
        from datetime import datetime
        
        date_str = self.created_at.strftime("%Y/%m")
        family_str = self.model_family.replace(" ", "_").lower() if self.model_family else "general"
        
        return os.path.join(
            "models",
            "organized",
            date_str,
            family_str,
            f"{self.unique_identifier}_v{self.version}"
        )

    def __str__(self):
        return f"{self.get_model_display_name()} ({self.mlflow_run_id[:8]}...)"

class Prediction(models.Model):
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE)
    input_data = models.JSONField()
    output_data = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    processing_time = models.FloatField(null=True, blank=True)  # in seconds
    input_image = models.ImageField(upload_to='predictions/inputs/', null=True, blank=True)
    output_image = models.ImageField(upload_to='predictions/outputs/', null=True, blank=True)

    def __str__(self):
        return f"Prediction {self.id} using {self.model.name}"

class TrainingTemplate(models.Model):
    name = models.CharField(max_length=200, help_text="Template name")
    description = models.TextField(blank=True, help_text="Template description")
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Training configuration fields - match TrainingForm fields
    model_type = models.CharField(max_length=50, default='unet')
    batch_size = models.IntegerField(default=32)
    epochs = models.IntegerField(default=100)
    learning_rate = models.FloatField(default=0.001)
    validation_split = models.FloatField(default=0.2)
    
    # Augmentation options
    use_random_flip = models.BooleanField(default=True)
    use_random_rotate = models.BooleanField(default=True)
    use_random_scale = models.BooleanField(default=True)
    use_random_intensity = models.BooleanField(default=True)
    crop_size = models.IntegerField(default=128)
    num_workers = models.IntegerField(default=4)
    
    # Additional metadata
    is_default = models.BooleanField(default=False, help_text="Default template for new trainings")
    created_by = models.CharField(max_length=100, blank=True, help_text="Template creator")
    
    class Meta:
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"{self.name}{' (Default)' if self.is_default else ''}"
    
    def get_form_data(self):
        """Return a dictionary suitable for populating TrainingForm"""
        return {
            'model_type': self.model_type,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'validation_split': self.validation_split,
            'use_random_flip': self.use_random_flip,
            'use_random_rotate': self.use_random_rotate,
            'use_random_scale': self.use_random_scale,
            'use_random_intensity': self.use_random_intensity,
            'crop_size': self.crop_size,
            'num_workers': self.num_workers,
        }
    
    def save(self, *args, **kwargs):
        # Ensure only one default template exists
        if self.is_default:
            TrainingTemplate.objects.filter(is_default=True).update(is_default=False)
        super().save(*args, **kwargs)
