# Dataset Manager API Serializers

from rest_framework import serializers
from django.contrib.auth.models import User
from .models import AnnotationSchema, Dataset, DataPipeline, PipelineExecution, DatasetSample

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'first_name', 'last_name']

class AnnotationSchemaSerializer(serializers.ModelSerializer):
    created_by = UserSerializer(read_only=True)
    
    class Meta:
        model = AnnotationSchema
        fields = [
            'id', 'name', 'type', 'description', 'schema_definition',
            'validation_rules', 'example_annotation', 'created_by',
            'created_at', 'updated_at', 'is_active', 'is_public', 'usage_count'
        ]
        read_only_fields = ['id', 'created_by', 'created_at', 'updated_at', 'usage_count']
    
    def validate_schema_definition(self, value):
        """Validate schema definition structure"""
        if not isinstance(value, dict):
            raise serializers.ValidationError("Schema definition must be a valid JSON object")
        
        # Basic schema validation
        required_fields = ['fields']
        for field in required_fields:
            if field not in value:
                raise serializers.ValidationError(f"Schema definition must contain '{field}' field")
        
        return value

class AnnotationSchemaListSerializer(serializers.ModelSerializer):
    """Simplified serializer for list views"""
    created_by_name = serializers.CharField(source='created_by.username', read_only=True)
    
    class Meta:
        model = AnnotationSchema
        fields = [
            'id', 'name', 'type', 'description', 'created_by_name',
            'created_at', 'is_active', 'is_public', 'usage_count'
        ]

class DatasetSerializer(serializers.ModelSerializer):
    created_by = UserSerializer(read_only=True)
    annotation_schema = AnnotationSchemaSerializer(read_only=True)
    annotation_schema_id = serializers.IntegerField(write_only=True, required=False)
    size_human_readable = serializers.CharField(read_only=True)
    
    class Meta:
        model = Dataset
        fields = [
            'id', 'name', 'description', 'version', 'original_filename',
            'file_path', 'format_type', 'annotation_schema', 'annotation_schema_id',
            'total_samples', 'file_size_bytes', 'size_human_readable',
            'class_distribution', 'statistics', 'status', 'processing_progress',
            'error_message', 'validation_results', 'quality_score',
            'pipeline_config', 'applied_transformations', 'compatible_model_types',
            'created_by', 'is_public', 'created_at', 'updated_at',
            'last_used_at', 'training_ready'
        ]
        read_only_fields = [
            'id', 'created_by', 'created_at', 'updated_at', 'file_size_bytes',
            'total_samples', 'processing_progress', 'size_human_readable'
        ]
    
    def validate(self, data):
        """Validate dataset data"""
        if 'annotation_schema_id' in data:
            try:
                schema = AnnotationSchema.objects.get(id=data['annotation_schema_id'])
                # Check if user has access to this schema
                request = self.context.get('request')
                if request and not schema.is_public and schema.created_by != request.user:
                    raise serializers.ValidationError("You don't have access to this annotation schema")
            except AnnotationSchema.DoesNotExist:
                raise serializers.ValidationError("Invalid annotation schema ID")
        
        return data

class DatasetListSerializer(serializers.ModelSerializer):
    """Simplified serializer for list views"""
    created_by_name = serializers.CharField(source='created_by.username', read_only=True)
    annotation_schema_name = serializers.CharField(source='annotation_schema.name', read_only=True)
    size_human_readable = serializers.CharField(read_only=True)
    
    class Meta:
        model = Dataset
        fields = [
            'id', 'name', 'version', 'status', 'total_samples',
            'size_human_readable', 'created_by_name', 'annotation_schema_name',
            'format_type', 'created_at', 'training_ready', 'quality_score'
        ]

class DataPipelineSerializer(serializers.ModelSerializer):
    created_by = UserSerializer(read_only=True)
    
    class Meta:
        model = DataPipeline
        fields = [
            'id', 'name', 'description', 'version', 'pipeline_graph',
            'input_formats', 'output_format', 'default_parameters',
            'required_parameters', 'is_template', 'is_public',
            'created_by', 'created_at', 'updated_at', 'usage_count',
            'average_execution_time', 'is_validated', 'validation_results'
        ]
        read_only_fields = [
            'id', 'created_by', 'created_at', 'updated_at', 'usage_count',
            'average_execution_time'
        ]
    
    def validate_pipeline_graph(self, value):
        """Validate pipeline graph structure"""
        if not isinstance(value, dict):
            raise serializers.ValidationError("Pipeline graph must be a valid JSON object")
        
        required_fields = ['nodes', 'edges']
        for field in required_fields:
            if field not in value:
                raise serializers.ValidationError(f"Pipeline graph must contain '{field}' field")
        
        # Validate nodes structure
        if not isinstance(value['nodes'], list):
            raise serializers.ValidationError("Nodes must be a list")
        
        # Validate edges structure
        if not isinstance(value['edges'], list):
            raise serializers.ValidationError("Edges must be a list")
        
        return value

class DataPipelineListSerializer(serializers.ModelSerializer):
    """Simplified serializer for list views"""
    created_by_name = serializers.CharField(source='created_by.username', read_only=True)
    
    class Meta:
        model = DataPipeline
        fields = [
            'id', 'name', 'description', 'version', 'is_template',
            'is_public', 'created_by_name', 'created_at', 'usage_count',
            'is_validated'
        ]

class PipelineExecutionSerializer(serializers.ModelSerializer):
    pipeline = DataPipelineListSerializer(read_only=True)
    dataset = DatasetListSerializer(read_only=True)
    executed_by = UserSerializer(read_only=True)
    duration = serializers.DurationField(read_only=True)
    
    class Meta:
        model = PipelineExecution
        fields = [
            'id', 'pipeline', 'dataset', 'parameters', 'status',
            'current_step', 'progress_percentage', 'created_at',
            'started_at', 'completed_at', 'duration', 'execution_log',
            'output_metadata', 'error_details', 'memory_peak_mb',
            'cpu_time_seconds', 'output_path', 'output_size_bytes',
            'executed_by'
        ]
        read_only_fields = [
            'id', 'pipeline', 'dataset', 'executed_by', 'created_at',
            'started_at', 'completed_at', 'duration', 'execution_log',
            'output_metadata', 'error_details', 'memory_peak_mb',
            'cpu_time_seconds', 'output_path', 'output_size_bytes'
        ]

class PipelineExecutionCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating new pipeline executions"""
    pipeline_id = serializers.IntegerField()
    dataset_id = serializers.IntegerField()
    
    class Meta:
        model = PipelineExecution
        fields = ['pipeline_id', 'dataset_id', 'parameters']
    
    def validate(self, data):
        """Validate pipeline execution data"""
        request = self.context.get('request')
        
        # Validate pipeline access
        try:
            pipeline = DataPipeline.objects.get(id=data['pipeline_id'])
            if not pipeline.is_public and pipeline.created_by != request.user:
                raise serializers.ValidationError("You don't have access to this pipeline")
        except DataPipeline.DoesNotExist:
            raise serializers.ValidationError("Invalid pipeline ID")
        
        # Validate dataset access
        try:
            dataset = Dataset.objects.get(id=data['dataset_id'])
            if not dataset.is_public and dataset.created_by != request.user:
                if request.user not in dataset.allowed_users.all():
                    raise serializers.ValidationError("You don't have access to this dataset")
        except Dataset.DoesNotExist:
            raise serializers.ValidationError("Invalid dataset ID")
        
        # Validate parameters
        parameters = data.get('parameters', {})
        required_params = pipeline.required_parameters
        for param in required_params:
            if param not in parameters:
                raise serializers.ValidationError(f"Required parameter '{param}' is missing")
        
        return data

class DatasetSampleSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    
    class Meta:
        model = DatasetSample
        fields = [
            'id', 'dataset', 'dataset_name', 'file_path', 'file_name',
            'file_size_bytes', 'file_type', 'sample_index', 'annotations',
            'annotation_confidence', 'is_valid', 'validation_errors',
            'thumbnail_path', 'preview_data', 'sample_class',
            'secondary_classes', 'quality_score', 'complexity_score',
            'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'dataset_name', 'created_at', 'updated_at'
        ]

class DatasetSampleListSerializer(serializers.ModelSerializer):
    """Simplified serializer for sample list views"""
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    
    class Meta:
        model = DatasetSample
        fields = [
            'id', 'dataset_name', 'file_name', 'sample_index',
            'sample_class', 'is_valid', 'quality_score', 'thumbnail_path'
        ]

# Upload serializers
class DatasetUploadSerializer(serializers.Serializer):
    """Serializer for dataset upload"""
    file = serializers.FileField(
        help_text="Dataset file (ZIP, TAR, or folder archive)"
    )
    name = serializers.CharField(
        max_length=200,
        help_text="Dataset name"
    )
    description = serializers.CharField(
        required=False,
        allow_blank=True,
        help_text="Dataset description"
    )
    annotation_schema_id = serializers.IntegerField(
        required=False,
        help_text="ID of annotation schema to use"
    )
    format_type = serializers.ChoiceField(
        choices=Dataset.FORMAT_CHOICES,
        default='zip',
        help_text="Dataset format type"
    )
    is_public = serializers.BooleanField(
        default=False,
        help_text="Make dataset publicly accessible"
    )
    
    def validate_file(self, value):
        """Validate uploaded file"""
        # Check file size (max 5GB)
        max_size = 5 * 1024 * 1024 * 1024  # 5GB
        if value.size > max_size:
            raise serializers.ValidationError(
                f"File size cannot exceed {max_size // (1024**3)}GB"
            )
        
        # Check file extension
        allowed_extensions = ['.zip', '.tar', '.tar.gz', '.tgz']
        file_extension = '.' + value.name.split('.')[-1].lower()
        if file_extension not in allowed_extensions:
            raise serializers.ValidationError(
                f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        return value

class SchemaValidationSerializer(serializers.Serializer):
    """Serializer for schema validation requests"""
    data = serializers.JSONField(
        help_text="Data to validate against schema"
    )
    schema_id = serializers.IntegerField(
        help_text="ID of schema to validate against"
    )
