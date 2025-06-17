# Dataset Manager Admin Interface

from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils import timezone
from .models import AnnotationSchema, Dataset, DataPipeline, PipelineExecution, DatasetSample

@admin.register(AnnotationSchema)
class AnnotationSchemaAdmin(admin.ModelAdmin):
    list_display = ['name', 'type', 'created_by', 'is_public', 'usage_count', 'created_at']
    list_filter = ['type', 'is_public', 'is_active', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'updated_at', 'usage_count']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'type', 'description')
        }),
        ('Schema Definition', {
            'fields': ('schema_definition', 'validation_rules', 'example_annotation'),
            'classes': ('collapse',)
        }),
        ('Permissions', {
            'fields': ('is_active', 'is_public')
        }),
        ('Metadata', {
            'fields': ('created_by', 'created_at', 'updated_at', 'usage_count'),
            'classes': ('collapse',)
        })
    )
    
    def get_readonly_fields(self, request, obj=None):
        if obj:  # editing an existing object
            return self.readonly_fields + ['created_by']
        return self.readonly_fields
    
    def save_model(self, request, obj, form, change):
        if not change:  # new object
            obj.created_by = request.user
        super().save_model(request, obj, form, change)

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ['name', 'version', 'status', 'total_samples', 'size_display', 'created_by', 'created_at']
    list_filter = ['status', 'format_type', 'is_public', 'training_ready', 'created_at']
    search_fields = ['name', 'description', 'original_filename']
    readonly_fields = ['created_at', 'updated_at', 'last_used_at', 'file_size_bytes', 'processing_progress']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'version', 'description')
        }),
        ('File Information', {
            'fields': ('original_filename', 'file_path', 'extracted_path', 'format_type', 'file_size_bytes')
        }),
        ('Schema and Structure', {
            'fields': ('annotation_schema', 'detected_structure'),
            'classes': ('collapse',)
        }),
        ('Statistics', {
            'fields': ('total_samples', 'class_distribution', 'statistics', 'quality_score'),
            'classes': ('collapse',)
        }),
        ('Processing Status', {
            'fields': ('status', 'processing_progress', 'error_message', 'validation_results'),
        }),
        ('Pipeline Configuration', {
            'fields': ('pipeline_config', 'applied_transformations', 'compatible_model_types'),
            'classes': ('collapse',)
        }),
        ('Permissions', {
            'fields': ('is_public', 'allowed_users', 'training_ready')
        }),
        ('Timestamps', {
            'fields': ('created_by', 'created_at', 'updated_at', 'last_used_at'),
            'classes': ('collapse',)
        })
    )
    
    filter_horizontal = ['allowed_users']
    
    def size_display(self, obj):
        """Display human readable file size"""
        return obj.size_human_readable
    size_display.short_description = 'File Size'
    
    def get_readonly_fields(self, request, obj=None):
        if obj:  # editing an existing object
            return self.readonly_fields + ['created_by']
        return self.readonly_fields
    
    def save_model(self, request, obj, form, change):
        if not change:  # new object
            obj.created_by = request.user
        super().save_model(request, obj, form, change)

@admin.register(DataPipeline)
class DataPipelineAdmin(admin.ModelAdmin):
    list_display = ['name', 'version', 'is_template', 'is_public', 'usage_count', 'created_by', 'created_at']
    list_filter = ['is_template', 'is_public', 'is_validated', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'updated_at', 'usage_count', 'average_execution_time']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'version', 'description')
        }),
        ('Pipeline Definition', {
            'fields': ('pipeline_graph', 'input_formats', 'output_format'),
            'classes': ('collapse',)
        }),
        ('Configuration', {
            'fields': ('default_parameters', 'required_parameters'),
            'classes': ('collapse',)
        }),
        ('Template and Sharing', {
            'fields': ('is_template', 'is_public')
        }),
        ('Validation', {
            'fields': ('is_validated', 'validation_results'),
            'classes': ('collapse',)
        }),
        ('Statistics', {
            'fields': ('usage_count', 'average_execution_time'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_by', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def get_readonly_fields(self, request, obj=None):
        if obj:  # editing an existing object
            return self.readonly_fields + ['created_by']
        return self.readonly_fields
    
    def save_model(self, request, obj, form, change):
        if not change:  # new object
            obj.created_by = request.user
        super().save_model(request, obj, form, change)

@admin.register(PipelineExecution)
class PipelineExecutionAdmin(admin.ModelAdmin):
    list_display = ['pipeline_name', 'dataset_name', 'status', 'progress_percentage', 'duration_display', 'executed_by', 'created_at']
    list_filter = ['status', 'created_at', 'pipeline__name']
    search_fields = ['pipeline__name', 'dataset__name']
    readonly_fields = ['created_at', 'started_at', 'completed_at', 'duration_display', 'output_size_bytes']
    
    fieldsets = (
        ('Execution Information', {
            'fields': ('pipeline', 'dataset', 'executed_by', 'parameters')
        }),
        ('Status', {
            'fields': ('status', 'current_step', 'progress_percentage')
        }),
        ('Timing', {
            'fields': ('created_at', 'started_at', 'completed_at', 'duration_display')
        }),
        ('Results', {
            'fields': ('output_path', 'output_size_bytes', 'output_metadata'),
            'classes': ('collapse',)
        }),
        ('Logging', {
            'fields': ('execution_log', 'error_details'),
            'classes': ('collapse',)
        }),
        ('Resource Usage', {
            'fields': ('memory_peak_mb', 'cpu_time_seconds'),
            'classes': ('collapse',)
        })
    )
    
    def pipeline_name(self, obj):
        return obj.pipeline.name
    pipeline_name.short_description = 'Pipeline'
    
    def dataset_name(self, obj):
        return obj.dataset.name
    dataset_name.short_description = 'Dataset'
    
    def duration_display(self, obj):
        if obj.duration:
            return str(obj.duration)
        return '-'
    duration_display.short_description = 'Duration'

@admin.register(DatasetSample)
class DatasetSampleAdmin(admin.ModelAdmin):
    list_display = ['dataset_name', 'file_name', 'sample_index', 'sample_class', 'is_valid', 'quality_score']
    list_filter = ['is_valid', 'sample_class', 'dataset__name']
    search_fields = ['file_name', 'sample_class', 'dataset__name']
    readonly_fields = ['created_at', 'updated_at', 'file_size_bytes']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('dataset', 'file_name', 'file_path', 'sample_index')
        }),
        ('File Details', {
            'fields': ('file_size_bytes', 'file_type', 'thumbnail_path')
        }),
        ('Classification', {
            'fields': ('sample_class', 'secondary_classes')
        }),
        ('Annotations', {
            'fields': ('annotations', 'annotation_confidence'),
            'classes': ('collapse',)
        }),
        ('Validation', {
            'fields': ('is_valid', 'validation_errors')
        }),
        ('Quality Metrics', {
            'fields': ('quality_score', 'complexity_score'),
            'classes': ('collapse',)
        }),
        ('Preview', {
            'fields': ('preview_data',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def dataset_name(self, obj):
        return obj.dataset.name
    dataset_name.short_description = 'Dataset'
    
    # Add custom actions
    actions = ['mark_as_valid', 'mark_as_invalid']
    
    def mark_as_valid(self, request, queryset):
        updated = queryset.update(is_valid=True, validation_errors=[])
        self.message_user(request, f'{updated} samples marked as valid.')
    mark_as_valid.short_description = "Mark selected samples as valid"
    
    def mark_as_invalid(self, request, queryset):
        updated = queryset.update(is_valid=False)
        self.message_user(request, f'{updated} samples marked as invalid.')
    mark_as_invalid.short_description = "Mark selected samples as invalid"
