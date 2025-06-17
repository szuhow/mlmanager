# Dataset Manager - Django App Structure

## Models (apps/dataset_manager/models.py)

from django.db import models
from django.contrib.auth.models import User
import json

class AnnotationSchema(models.Model):
    """Schema definition for dataset annotations"""
    ANNOTATION_TYPES = [
        ('classification', 'Classification'),
        ('segmentation', 'Segmentation'), 
        ('detection', 'Object Detection'),
        ('custom', 'Custom Schema'),
    ]
    
    name = models.CharField(max_length=200)
    type = models.CharField(max_length=20, choices=ANNOTATION_TYPES)
    schema_definition = models.JSONField()  # Flexible schema storage
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.name} ({self.type})"

class Dataset(models.Model):
    """Custom dataset registration"""
    STATUS_CHOICES = [
        ('uploading', 'Uploading'),
        ('processing', 'Processing'),
        ('validating', 'Validating'),
        ('ready', 'Ready'),
        ('error', 'Error'),
    ]
    
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    original_filename = models.CharField(max_length=500)
    file_path = models.CharField(max_length=1000)
    annotation_schema = models.ForeignKey(AnnotationSchema, on_delete=models.CASCADE)
    
    # Metadata
    total_samples = models.IntegerField(default=0)
    file_size_bytes = models.BigIntegerField(default=0)
    format_info = models.JSONField(default=dict)
    statistics = models.JSONField(default=dict)
    
    # Status tracking
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='uploading')
    error_message = models.TextField(blank=True)
    
    # Ownership
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Pipeline configuration
    pipeline_config = models.JSONField(default=dict)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.total_samples} samples)"

class DataPipeline(models.Model):
    """Data processing pipeline definition"""
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    
    # Pipeline definition as a graph
    pipeline_graph = models.JSONField()  # Nodes and edges
    
    # Execution tracking
    is_template = models.BooleanField(default=False)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class PipelineExecution(models.Model):
    """Track pipeline execution instances"""
    pipeline = models.ForeignKey(DataPipeline, on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    
    # Execution status
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ], default='pending')
    
    # Tracking
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    progress_percentage = models.IntegerField(default=0)
    
    # Results
    execution_log = models.JSONField(default=list)
    output_metadata = models.JSONField(default=dict)
    
    def __str__(self):
        return f"{self.pipeline.name} -> {self.dataset.name}"

## Views Structure

### 1. Dataset Upload Wizard
- Multi-step form with AJAX validation
- Real-time file processing status
- Preview of detected structure

### 2. Schema Designer
- Visual form builder
- Drag & drop field types
- Real-time validation

### 3. Pipeline Editor
- Graph-based editor using vis.js or cytoscape.js
- Node library for transformations
- Connection validation

### 4. Dataset Browser
- Filterable table with previews
- Batch operations
- Export capabilities

## API Endpoints

### Dataset Management
POST /api/datasets/upload/
GET /api/datasets/
GET /api/datasets/{id}/
PUT /api/datasets/{id}/
DELETE /api/datasets/{id}/
GET /api/datasets/{id}/preview/
GET /api/datasets/{id}/statistics/

### Schema Management  
POST /api/schemas/
GET /api/schemas/
GET /api/schemas/{id}/
PUT /api/schemas/{id}/
DELETE /api/schemas/{id}/
POST /api/schemas/{id}/validate/

### Pipeline Management
POST /api/pipelines/
GET /api/pipelines/
GET /api/pipelines/{id}/
PUT /api/pipelines/{id}/
POST /api/pipelines/{id}/execute/
GET /api/pipelines/executions/{id}/status/

## Integration with Existing ML Pipeline

### 1. Model Training Integration
- Dataset selection in training forms
- Automatic schema validation
- Pipeline integration with training

### 2. MLflow Integration
- Dataset versioning through MLflow
- Artifact tracking
- Experiment lineage

### 3. Existing Dataset Preview
- Extended preview system
- Custom annotation rendering
- Interactive exploration
