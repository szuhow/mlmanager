# Dataset Manager Django App Implementation

from django.db import models
from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator
import json
import os

class AnnotationSchema(models.Model):
    """Schema definition for dataset annotations"""
    ANNOTATION_TYPES = [
        ('classification', 'Classification'),
        ('segmentation', 'Semantic Segmentation'), 
        ('detection', 'Object Detection'),
        ('instance_segmentation', 'Instance Segmentation'),
        ('keypoint', 'Keypoint Detection'),
        ('custom', 'Custom Schema'),
    ]
    
    name = models.CharField(max_length=200, unique=True)
    type = models.CharField(max_length=25, choices=ANNOTATION_TYPES)
    description = models.TextField(blank=True)
    
    # Schema definition - flexible JSON structure
    schema_definition = models.JSONField(default=dict, help_text="JSON schema for annotations")
    
    # Validation rules
    validation_rules = models.JSONField(default=dict, help_text="Validation rules for the schema")
    
    # Example data for UI preview
    example_annotation = models.JSONField(default=dict, help_text="Example annotation for preview")
    
    # Metadata
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    is_public = models.BooleanField(default=False, help_text="Available to all users")
    
    # Usage statistics
    usage_count = models.IntegerField(default=0)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Annotation Schema"
        verbose_name_plural = "Annotation Schemas"
    
    def __str__(self):
        return f"{self.name} ({self.get_type_display()})"
    
    def increment_usage(self):
        self.usage_count += 1
        self.save(update_fields=['usage_count'])

class Dataset(models.Model):
    """Custom dataset registration and management"""
    STATUS_CHOICES = [
        ('uploading', 'Uploading'),
        ('extracting', 'Extracting Files'),
        ('analyzing', 'Analyzing Structure'),
        ('validating', 'Validating Data'),
        ('processing', 'Processing'),
        ('ready', 'Ready for Training'),
        ('error', 'Error'),
        ('archived', 'Archived'),
    ]
    
    FORMAT_CHOICES = [
        ('zip', 'ZIP Archive'),
        ('tar', 'TAR Archive'),
        ('folder', 'Folder Structure'),
        ('csv', 'CSV with Annotations'),
        ('coco', 'COCO Format'),
        ('yolo', 'YOLO Format'),
        ('pascal_voc', 'Pascal VOC'),
        ('custom', 'Custom Format'),
    ]
    
    # Basic information
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    version = models.CharField(max_length=50, default='1.0')
    
    # File information
    original_filename = models.CharField(max_length=500)
    file_path = models.CharField(max_length=1000)
    extracted_path = models.CharField(max_length=1000, blank=True)
    format_type = models.CharField(max_length=20, choices=FORMAT_CHOICES, default='zip')
    
    # Schema and structure
    annotation_schema = models.ForeignKey(AnnotationSchema, on_delete=models.CASCADE, null=True, blank=True)
    detected_structure = models.JSONField(default=dict, help_text="Auto-detected file structure")
    
    # Statistics and metadata
    total_samples = models.IntegerField(default=0)
    file_size_bytes = models.BigIntegerField(default=0)
    class_distribution = models.JSONField(default=dict, help_text="Distribution of classes/labels")
    statistics = models.JSONField(default=dict, help_text="Dataset statistics")
    
    # Status and processing
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='uploading')
    processing_progress = models.IntegerField(default=0, help_text="Processing progress 0-100")
    error_message = models.TextField(blank=True)
    processing_log = models.JSONField(default=list, help_text="Processing steps log")
    
    # Quality checks
    validation_results = models.JSONField(default=dict, help_text="Data validation results")
    quality_score = models.FloatField(null=True, blank=True, help_text="Overall quality score 0-1")
    
    # Pipeline configuration
    pipeline_config = models.JSONField(default=dict, help_text="Data processing pipeline config")
    applied_transformations = models.JSONField(default=list, help_text="List of applied transformations")
    
    # Ownership and permissions
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    is_public = models.BooleanField(default=False)
    allowed_users = models.ManyToManyField(User, related_name='accessible_datasets', blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_used_at = models.DateTimeField(null=True, blank=True)
    
    # Training integration
    compatible_model_types = models.JSONField(default=list, help_text="Compatible ML model types")
    training_ready = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-created_at']
        unique_together = ['name', 'created_by', 'version']
        verbose_name = "Dataset"
        verbose_name_plural = "Datasets"
    
    def __str__(self):
        return f"{self.name} v{self.version} ({self.total_samples} samples)"
    
    @property
    def size_human_readable(self):
        """Return human readable file size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if self.file_size_bytes < 1024.0:
                return f"{self.file_size_bytes:.1f} {unit}"
            self.file_size_bytes /= 1024.0
        return f"{self.file_size_bytes:.1f} TB"
    
    def mark_as_used(self):
        """Update last used timestamp"""
        from django.utils import timezone
        self.last_used_at = timezone.now()
        self.save(update_fields=['last_used_at'])

class DataPipeline(models.Model):
    """Data processing pipeline definition using graph structure"""
    
    # Basic information
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    version = models.CharField(max_length=50, default='1.0')
    
    # Pipeline definition as a directed graph
    pipeline_graph = models.JSONField(help_text="Pipeline as nodes and edges JSON")
    
    # Supported input/output types
    input_formats = models.JSONField(default=list, help_text="Supported input formats")
    output_format = models.CharField(max_length=100, help_text="Output format")
    
    # Configuration
    default_parameters = models.JSONField(default=dict, help_text="Default pipeline parameters")
    required_parameters = models.JSONField(default=list, help_text="Required parameter names")
    
    # Template and sharing
    is_template = models.BooleanField(default=False, help_text="Available as template")
    is_public = models.BooleanField(default=False, help_text="Available to all users")
    
    # Metadata
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Usage tracking
    usage_count = models.IntegerField(default=0)
    average_execution_time = models.DurationField(null=True, blank=True)
    
    # Validation
    is_validated = models.BooleanField(default=False, help_text="Pipeline tested and validated")
    validation_results = models.JSONField(default=dict, help_text="Validation test results")
    
    class Meta:
        ordering = ['-created_at']
        unique_together = ['name', 'created_by', 'version']
        verbose_name = "Data Pipeline"
        verbose_name_plural = "Data Pipelines"
    
    def __str__(self):
        return f"{self.name} v{self.version}"
    
    def increment_usage(self):
        self.usage_count += 1
        self.save(update_fields=['usage_count'])

class PipelineExecution(models.Model):
    """Track individual pipeline execution instances"""
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('initializing', 'Initializing'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    
    # Relationships
    pipeline = models.ForeignKey(DataPipeline, on_delete=models.CASCADE, related_name='executions')
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='pipeline_executions')
    
    # Execution configuration
    parameters = models.JSONField(default=dict, help_text="Execution parameters")
    
    # Status tracking
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    current_step = models.CharField(max_length=200, blank=True, help_text="Current processing step")
    progress_percentage = models.IntegerField(default=0)
    
    # Timing
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Results and logging
    execution_log = models.JSONField(default=list, help_text="Step-by-step execution log")
    output_metadata = models.JSONField(default=dict, help_text="Output statistics and metadata")
    error_details = models.JSONField(default=dict, help_text="Error information if failed")
    
    # Resource usage
    memory_peak_mb = models.IntegerField(null=True, blank=True, help_text="Peak memory usage in MB")
    cpu_time_seconds = models.FloatField(null=True, blank=True, help_text="Total CPU time")
    
    # Output tracking
    output_path = models.CharField(max_length=1000, blank=True, help_text="Path to processed output")
    output_size_bytes = models.BigIntegerField(default=0)
    
    # User tracking
    executed_by = models.ForeignKey(User, on_delete=models.CASCADE)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Pipeline Execution"
        verbose_name_plural = "Pipeline Executions"
    
    def __str__(self):
        return f"{self.pipeline.name} → {self.dataset.name} ({self.status})"
    
    @property
    def duration(self):
        """Calculate execution duration"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    def add_log_entry(self, level, message, details=None):
        """Add entry to execution log"""
        from django.utils import timezone
        entry = {
            'timestamp': timezone.now().isoformat(),
            'level': level,  # info, warning, error
            'message': message,
            'details': details or {}
        }
        self.execution_log.append(entry)
        self.save(update_fields=['execution_log'])

class DatasetSample(models.Model):
    """Individual samples within a dataset for preview and validation"""
    
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='samples')
    
    # File information
    file_path = models.CharField(max_length=1000, help_text="Relative path within dataset")
    file_name = models.CharField(max_length=500)
    file_size_bytes = models.IntegerField(default=0)
    file_type = models.CharField(max_length=100, help_text="MIME type or file extension")
    
    # Sample metadata
    sample_index = models.IntegerField(help_text="Index within dataset")
    
    # Annotations
    annotations = models.JSONField(default=dict, help_text="Sample annotations")
    annotation_confidence = models.FloatField(null=True, blank=True, help_text="Annotation quality/confidence")
    
    # Validation
    is_valid = models.BooleanField(default=True)
    validation_errors = models.JSONField(default=list, help_text="List of validation errors")
    
    # Preview data
    thumbnail_path = models.CharField(max_length=1000, blank=True, help_text="Path to thumbnail image")
    preview_data = models.JSONField(default=dict, help_text="Preview metadata (dimensions, etc.)")
    
    # Classification for quick filtering
    sample_class = models.CharField(max_length=200, blank=True, help_text="Primary class/label")
    secondary_classes = models.JSONField(default=list, help_text="Additional classes")
    
    # Quality metrics
    quality_score = models.FloatField(null=True, blank=True, help_text="Sample quality score")
    complexity_score = models.FloatField(null=True, blank=True, help_text="Annotation complexity")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['sample_index']
        unique_together = ['dataset', 'sample_index']
        verbose_name = "Dataset Sample"
        verbose_name_plural = "Dataset Samples"
    
    def __str__(self):
        return f"{self.dataset.name} - {self.file_name}"
