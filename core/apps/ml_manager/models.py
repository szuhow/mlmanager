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
            ('stopping', 'Stopping'),
            ('completed', 'Completed'),
            ('stopped', 'Stopped'),
            ('failed', 'Failed'),
            ('archived', 'Archived'),
            ('deployed', 'Deployed')
        ],
        default='pending'
    )
    current_epoch = models.IntegerField(default=0)
    total_epochs = models.IntegerField(default=0)
    current_batch = models.IntegerField(default=0)
    total_batches_per_epoch = models.IntegerField(default=0)
    train_loss = models.FloatField(default=0.0)
    val_loss = models.FloatField(default=0.0)
    train_dice = models.FloatField(default=0.0)
    val_dice = models.FloatField(default=0.0)
    best_val_dice = models.FloatField(default=0.0)
    train_iou = models.FloatField(default=0.0)
    val_iou = models.FloatField(default=0.0)
    best_val_iou = models.FloatField(default=0.0)
    
    # Primary metric configuration for training display
    primary_metric_type = models.CharField(max_length=20, default='dice', help_text="Primary metric type (dice, iou, accuracy, etc.)")
    primary_metric_name = models.CharField(max_length=50, default='Dice', help_text="Display name for primary metric")
    train_metric_name = models.CharField(max_length=50, default='Training Dice', help_text="Display name for training metric")
    val_metric_name = models.CharField(max_length=50, default='Validation Dice', help_text="Display name for validation metric")
    
    stop_requested = models.BooleanField(default=False)
    process_id = models.IntegerField(null=True, blank=True, help_text="Process ID of the training process")
    
    # Training logs field (for backward compatibility and query support)
    training_logs = models.TextField(blank=True, null=True, help_text="Training logs and error messages")

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
            "data",
            "models",
            "organized",
            date_str,
            family_str,
            f"{self.unique_identifier}_v{self.version}"
        )

    def __str__(self):
        return f"{self.get_model_display_name()} ({self.mlflow_run_id[:8]}...)"
    
    @property
    def progress_percentage(self):
        """Calculate training progress percentage including batch progress within epoch"""
        if self.total_epochs > 0:
            # For completed training, return 100%
            if self.status == 'completed':
                return 100.0
                
            # For training models, calculate actual progress
            # current_epoch is already 1-based (stored as epoch+1 in callback)
            # but we need 0-based for percentage calculation
            completed_epochs = max(0, self.current_epoch - 1)
            
            # Base epoch progress (completed epochs)
            epoch_progress = completed_epochs / self.total_epochs
            
            # Add batch progress within current epoch if available
            if self.total_batches_per_epoch > 0 and self.current_batch > 0:
                batch_progress_in_epoch = self.current_batch / self.total_batches_per_epoch
                epoch_progress += batch_progress_in_epoch / self.total_epochs
            
            return min(100, epoch_progress * 100)
        return 0
    
    @property
    def batch_progress_percentage(self):
        """Calculate progress within current epoch based on batches"""
        if self.total_batches_per_epoch > 0:
            return min(100, (self.current_batch / self.total_batches_per_epoch) * 100)
        return 0
    
    @property
    def is_training_active(self):
        """Check if model is currently training"""
        return self.status in ['loading', 'training']
    
    @property
    def training_progress_info(self):
        """Get detailed training progress information"""
        return {
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'current_batch': self.current_batch,
            'total_batches_per_epoch': self.total_batches_per_epoch,
            'progress_percentage': self.progress_percentage,
            'batch_progress_percentage': self.batch_progress_percentage,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_dice': self.train_dice,
            'val_dice': self.val_dice,
            'best_val_dice': self.best_val_dice,
            'train_iou': self.train_iou,
            'val_iou': self.val_iou,
            'best_val_iou': self.best_val_iou,
            'status': self.status
        }

    def delete(self, *args, **kwargs):
        """Override delete to clean up associated files and MLflow data"""
        import os
        import shutil
        from django.conf import settings
        from django.db import transaction
        import logging
        
        logger = logging.getLogger(__name__)
        cleanup_paths = []
        model_name = self.name
        model_id = self.id
        
        try:
            with transaction.atomic():
                # 1. Clean up MLflow run and artifacts
                if self.mlflow_run_id:
                    try:
                        import mlflow
                        # Try to delete MLflow run
                        mlflow.delete_run(self.mlflow_run_id)
                        logger.info(f"Deleted MLflow run: {self.mlflow_run_id}")
                    except Exception as e:
                        logger.warning(f"Could not delete MLflow run {self.mlflow_run_id}: {e}")
                    
                    # Add MLflow artifacts directory to cleanup
                    mlflow_artifacts_path = os.path.join(settings.CORE_DATA_DIR, 'mlflow', self.mlflow_run_id)
                    if os.path.exists(mlflow_artifacts_path):
                        cleanup_paths.append(mlflow_artifacts_path)
                
                # 2. Clean up model directory
                if self.model_directory and os.path.exists(self.model_directory):
                    cleanup_paths.append(self.model_directory)
                
                # 3. Clean up individual files
                file_fields = ['model_weights_path', 'model_config_path']
                for field_name in file_fields:
                    file_path = getattr(self, field_name, None)
                    if file_path and os.path.exists(file_path):
                        cleanup_paths.append(file_path)
                
                # 4. Clean up organized models directory
                if hasattr(settings, 'ORGANIZED_MODELS_DIR'):
                    organized_path = os.path.join(settings.ORGANIZED_MODELS_DIR, f"model_{self.id}")
                    if os.path.exists(organized_path):
                        cleanup_paths.append(organized_path)
                
                # Perform the actual deletion from database
                super().delete(*args, **kwargs)
                logger.info(f"Successfully deleted model {model_name} (ID: {model_id}) from database")
            
            # Clean up files after successful database deletion
            for path in cleanup_paths:
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                        logger.info(f"Deleted file: {path}")
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                        logger.info(f"Deleted directory: {path}")
                except Exception as e:
                    logger.warning(f"Could not delete {path}: {e}")
            
            logger.info(f"Model {model_name} (ID: {model_id}) and associated files deleted successfully")
            
        except Exception as e:
            logger.error(f"Error deleting model {model_name} (ID: {model_id}): {e}")
            raise  # Re-raise to ensure the error is propagated

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
    model_type = models.CharField(max_length=50, default='unet', help_text="Model architecture type (e.g., unet, unet-old)")
    batch_size = models.IntegerField(default=32)
    epochs = models.IntegerField(default=1)
    learning_rate = models.FloatField(default=0.001)
    
    # Validation configuration
    use_validation_split = models.BooleanField(default=True, help_text="Use automatic validation split from training data")
    validation_split = models.FloatField(default=0.2, help_text="Validation set size (0-1), used only when use_validation_split is True")
    custom_validation_dataset = models.CharField(max_length=500, blank=True, help_text="Path to custom validation dataset")
    
    # Image resolution for training
    RESOLUTION_CHOICES = [
        ('original', 'Original Size'),
        ('128', '128 x 128 pixels'),
        ('256', '256 x 256 pixels'),
        ('384', '384 x 384 pixels'),
        ('512', '512 x 512 pixels'),
    ]
    resolution = models.CharField(
        max_length=20,
        choices=RESOLUTION_CHOICES,
        default='256',
        help_text="Training image resolution. Higher resolutions require more memory."
    )
    
    # Device selection for training - choices are set dynamically in forms
    device = models.CharField(
        max_length=20,
        default='auto',
        help_text="Device to use for training. Auto will use CUDA if available."
    )
    
    # Enhanced Augmentation options with richer controls
    use_random_flip = models.BooleanField(default=True)
    flip_probability = models.FloatField(default=0.5, help_text="Probability of applying flip (0.0-1.0)")
    
    use_random_rotate = models.BooleanField(default=True)
    rotation_range = models.IntegerField(default=30, help_text="Maximum rotation angle in degrees (±range)")
    
    use_random_scale = models.BooleanField(default=True)
    scale_range_min = models.FloatField(default=0.8, help_text="Minimum scale factor")
    scale_range_max = models.FloatField(default=1.2, help_text="Maximum scale factor")
    
    use_random_intensity = models.BooleanField(default=True)
    intensity_range = models.FloatField(default=0.2, help_text="Intensity variation range (±range)")
    
    use_random_crop = models.BooleanField(default=False)
    crop_size = models.IntegerField(default=128)
    
    # Advanced cropping for segmentation
    use_pos_neg_cropping = models.BooleanField(
        default=False, 
        help_text="Use RandCropByPosNegLabeld for better segmentation cropping (samples positive and negative regions)"
    )
    
    use_elastic_transform = models.BooleanField(default=False)
    elastic_alpha = models.FloatField(default=34.0, help_text="Elastic transformation strength")
    elastic_sigma = models.FloatField(default=4.0, help_text="Elastic transformation smoothness")
    
    use_gaussian_noise = models.BooleanField(default=False)
    noise_std = models.FloatField(default=0.01, help_text="Standard deviation of Gaussian noise")
    
    num_workers = models.IntegerField(default=1)  # Conservative default for Docker
    
    # Optimizer selection
    OPTIMIZER_CHOICES = [
        ('adam', 'Adam'),
        ('sgd', 'SGD'),
        ('rmsprop', 'RMSprop'),
        ('adamw', 'AdamW'),
    ]
    optimizer = models.CharField(
        max_length=20,
        choices=OPTIMIZER_CHOICES,
        default='adam',
        help_text="Optimizer to use for training"
    )
    
    # Learning Rate Scheduler Configuration
    LR_SCHEDULER_CHOICES = [
        ('none', 'None'),
        ('plateau', 'ReduceLROnPlateau'),
        ('step', 'StepLR'),
        ('exponential', 'ExponentialLR'),
        ('cosine', 'CosineAnnealingLR'),
        ('adaptive', 'Custom Adaptive'),
    ]
    lr_scheduler = models.CharField(
        max_length=20,
        choices=LR_SCHEDULER_CHOICES,
        default='none',
        help_text="Learning rate scheduler type"
    )
    lr_patience = models.IntegerField(default=5, help_text="Patience for plateau scheduler")
    lr_factor = models.FloatField(default=0.5, help_text="Factor to reduce learning rate")
    lr_step_size = models.IntegerField(default=10, help_text="Step size for step scheduler")
    lr_gamma = models.FloatField(default=0.1, help_text="Gamma for step/exponential scheduler")
    min_lr = models.FloatField(default=1e-7, help_text="Minimum learning rate")
    
    # Early Stopping Configuration
    use_early_stopping = models.BooleanField(default=False, help_text="Enable early stopping during training")
    early_stopping_patience = models.IntegerField(default=10, help_text="Epochs to wait for improvement before stopping")
    early_stopping_min_epochs = models.IntegerField(default=20, help_text="Minimum epochs before early stopping can occur")
    early_stopping_min_delta = models.FloatField(default=1e-4, help_text="Minimum improvement required to reset patience")
    
    EARLY_STOPPING_METRIC_CHOICES = [
        ('val_dice', 'Validation Dice Score'),
        ('val_iou', 'Validation IoU Score'),
        ('val_loss', 'Validation Loss'),
        ('val_accuracy', 'Validation Accuracy'),
    ]
    early_stopping_metric = models.CharField(
        max_length=20,
        choices=EARLY_STOPPING_METRIC_CHOICES,
        default='val_dice',
        help_text="Metric to monitor for early stopping"
    )
    
    # Post-processing configuration
    threshold = models.FloatField(
        default=0.5,
        help_text="Binary segmentation threshold for hard predictions (0.1-0.9)"
    )
    
    # Medical Preprocessing Configuration
    use_medical_preprocessing = models.BooleanField(
        default=False,
        help_text="Enable advanced medical image preprocessing"
    )
    
    PREPROCESSING_TYPE_CHOICES = [
        ('angiography', 'Angiography (X-ray coronary images)'),
        ('ct_coronary', 'CT Coronary Angiography'),
        ('oct_coronary', 'OCT Coronary Images'),
        ('general', 'General Medical Images'),
    ]
    preprocessing_type = models.CharField(
        max_length=20,
        choices=PREPROCESSING_TYPE_CHOICES,
        default='angiography',
        help_text="Type of medical imaging modality"
    )
    
    # CLAHE parameters
    clahe_clip_limit = models.FloatField(
        default=3.0,
        help_text="CLAHE clip limit for contrast enhancement (1.0-8.0)"
    )
    clahe_tile_size = models.IntegerField(
        default=8,
        help_text="CLAHE tile grid size (4-16)"
    )
    
    # Unsharp masking
    use_unsharp_masking = models.BooleanField(
        default=False,
        help_text="Enable unsharp masking for edge enhancement"
    )
    unsharp_amount = models.FloatField(
        default=1.0,
        help_text="Unsharp masking strength (0.5-2.0)"
    )
    unsharp_radius = models.FloatField(
        default=1.0,
        help_text="Unsharp masking radius (0.5-3.0)"
    )
    
    # Frangi vesselness filter
    use_frangi_filter = models.BooleanField(
        default=False,
        help_text="Enable Frangi vesselness filter for vessel enhancement"
    )
    frangi_sigma_min = models.FloatField(
        default=1.0,
        help_text="Minimum sigma for Frangi filter (detects thin vessels)"
    )
    frangi_sigma_max = models.FloatField(
        default=10.0,
        help_text="Maximum sigma for Frangi filter (detects thick vessels)"
    )
    frangi_sigma_step = models.FloatField(
        default=2.0,
        help_text="Step size for sigma range (1.0-3.0)"
    )
    
    # Denoising
    use_denoising = models.BooleanField(
        default=False,
        help_text="Enable denoising filters to reduce image noise"
    )
    noise_reduction_sigma = models.FloatField(
        default=1.0,
        help_text="Noise reduction strength (0.5-3.0)"
    )
    
    # Additional preprocessing options
    use_histogram_equalization = models.BooleanField(
        default=False,
        help_text="Enable histogram equalization for global contrast"
    )
    normalize_intensity = models.BooleanField(
        default=True,
        help_text="Normalize image intensity to standard range"
    )
    gamma_correction = models.FloatField(
        default=1.0,
        help_text="Gamma correction (0.5-2.0, 1.0=no correction)"
    )
    custom_preprocessing_pipeline = models.CharField(
        max_length=500,
        blank=True,
        help_text="Custom preprocessing pipeline (comma-separated)"
    )

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
            'use_validation_split': self.use_validation_split,
            'validation_split': self.validation_split,
            'custom_validation_dataset': self.custom_validation_dataset,
            'resolution': self.resolution,
            'device': self.device,
            'use_random_flip': self.use_random_flip,
            'flip_probability': self.flip_probability,
            'use_random_rotate': self.use_random_rotate,
            'rotation_range': self.rotation_range,
            'use_random_scale': self.use_random_scale,
            'scale_range_min': self.scale_range_min,
            'scale_range_max': self.scale_range_max,
            'use_random_intensity': self.use_random_intensity,
            'intensity_range': self.intensity_range,
            'use_random_crop': self.use_random_crop,
            'crop_size': self.crop_size,
            'use_pos_neg_cropping': self.use_pos_neg_cropping,
            'use_elastic_transform': self.use_elastic_transform,
            'elastic_alpha': self.elastic_alpha,
            'elastic_sigma': self.elastic_sigma,
            'use_gaussian_noise': self.use_gaussian_noise,
            'noise_std': self.noise_std,
            'num_workers': self.num_workers,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
            'lr_patience': self.lr_patience,
            'lr_factor': self.lr_factor,
            'lr_step_size': self.lr_step_size,
            'lr_gamma': self.lr_gamma,
            'min_lr': self.min_lr,
            # Early stopping fields
            'use_early_stopping': self.use_early_stopping,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_epochs': self.early_stopping_min_epochs,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'early_stopping_metric': self.early_stopping_metric,
            'threshold': self.threshold,
            # Medical preprocessing fields
            'use_medical_preprocessing': self.use_medical_preprocessing,
            'preprocessing_type': self.preprocessing_type,
            'clahe_clip_limit': self.clahe_clip_limit,
            'clahe_tile_size': self.clahe_tile_size,
            'use_unsharp_masking': self.use_unsharp_masking,
            'unsharp_amount': self.unsharp_amount,
            'unsharp_radius': self.unsharp_radius,
            'use_frangi_filter': self.use_frangi_filter,
            'frangi_sigma_min': self.frangi_sigma_min,
            'frangi_sigma_max': self.frangi_sigma_max,
            'frangi_sigma_step': self.frangi_sigma_step,
            'use_denoising': self.use_denoising,
            'noise_reduction_sigma': self.noise_reduction_sigma,
            'use_histogram_equalization': self.use_histogram_equalization,
            'normalize_intensity': self.normalize_intensity,
            'gamma_correction': self.gamma_correction,
            'custom_preprocessing_pipeline': self.custom_preprocessing_pipeline,
        }
    
    def save(self, *args, **kwargs):
        # Ensure only one default template exists
        if self.is_default:
            TrainingTemplate.objects.filter(is_default=True).update(is_default=False)
        super().save(*args, **kwargs)


class InferenceResult(models.Model):
    """Model to store inference results"""
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='inference_results')
    
    # Input information
    input_image = models.ImageField(upload_to='inference/inputs/%Y/%m/%d/')
    input_filename = models.CharField(max_length=255, help_text="Original filename of uploaded image")
    
    # Output results
    output_mask = models.ImageField(upload_to='inference/outputs/%Y/%m/%d/', null=True, blank=True)
    output_overlay = models.ImageField(upload_to='inference/overlays/%Y/%m/%d/', null=True, blank=True)
    
    # Inference metadata
    inference_config = models.JSONField(default=default_dict, encoder=DjangoJSONEncoder, help_text="Inference configuration used")
    processing_time = models.FloatField(null=True, blank=True, help_text="Processing time in seconds")
    
    # Results metrics
    detected_objects_count = models.IntegerField(default=0, help_text="Number of detected objects")
    total_area_pixels = models.IntegerField(default=0, help_text="Total segmented area in pixels")
    confidence_scores = models.JSONField(default=list, encoder=DjangoJSONEncoder, help_text="Confidence scores for detected objects")
    
    # Timestamps
    created_at = models.DateTimeField(default=timezone.now)
    
    # Status
    STATUS_CHOICES = [
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='processing')
    task_id = models.CharField(max_length=100, blank=True, null=True, help_text="Celery task ID for asynchronous processing")
    error_message = models.TextField(blank=True, help_text="Error message if inference failed")
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Inference Result"
        verbose_name_plural = "Inference Results"
    
    def __str__(self):
        return f"Inference for {self.input_filename} using {self.model.name}"
    
    @property
    def get_display_name(self):
        """Get display name for the inference result"""
        return f"{self.input_filename} → {self.model.name}"
    
    @property
    def processing_time_display(self):
        """Format processing time for display"""
        if self.processing_time:
            return f"{self.processing_time:.2f}s"
        return "N/A"
