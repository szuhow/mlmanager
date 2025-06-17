# Dataset Manager Views

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, Http404
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.generic import ListView, DetailView
from django.db.models import Q, Count
from django.core.paginator import Paginator
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
import json
import os
from .models import AnnotationSchema, Dataset, DataPipeline, PipelineExecution, DatasetSample
from .serializers import (
    AnnotationSchemaSerializer, AnnotationSchemaListSerializer,
    DatasetSerializer, DatasetListSerializer, DatasetUploadSerializer,
    DataPipelineSerializer, DataPipelineListSerializer,
    PipelineExecutionSerializer, PipelineExecutionCreateSerializer,
    DatasetSampleSerializer, DatasetSampleListSerializer,
    SchemaValidationSerializer
)
from .utils import DatasetProcessor, PipelineExecutor, SchemaValidator

# ================== Django Views ==================

@login_required
def dataset_manager_home(request):
    """Main dataset manager dashboard"""
    # Get user's datasets
    user_datasets = Dataset.objects.filter(
        Q(created_by=request.user) | Q(is_public=True) | Q(allowed_users=request.user)
    ).distinct()[:5]
    
    # Get user's schemas
    user_schemas = AnnotationSchema.objects.filter(
        Q(created_by=request.user) | Q(is_public=True)
    ).distinct()[:5]
    
    # Get recent pipeline executions
    recent_executions = PipelineExecution.objects.filter(
        executed_by=request.user
    )[:5]
    
    # Statistics
    stats = {
        'total_datasets': user_datasets.count(),
        'ready_datasets': user_datasets.filter(status='ready').count(),
        'total_schemas': user_schemas.count(),
        'running_executions': recent_executions.filter(status='running').count()
    }
    
    context = {
        'datasets': user_datasets,
        'schemas': user_schemas,
        'recent_executions': recent_executions,
        'stats': stats
    }
    
    return render(request, 'dataset_manager/dashboard.html', context)

class DatasetListView(ListView):
    """List view for datasets"""
    model = Dataset
    template_name = 'dataset_manager/dataset_list.html'
    context_object_name = 'datasets'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = Dataset.objects.filter(
            Q(created_by=self.request.user) | 
            Q(is_public=True) | 
            Q(allowed_users=self.request.user)
        ).distinct()
        
        # Apply filters
        status_filter = self.request.GET.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
            
        format_filter = self.request.GET.get('format')
        if format_filter:
            queryset = queryset.filter(format_type=format_filter)
            
        search = self.request.GET.get('search')
        if search:
            queryset = queryset.filter(
                Q(name__icontains=search) | 
                Q(description__icontains=search)
            )
        
        return queryset.order_by('-created_at')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['status_choices'] = Dataset.STATUS_CHOICES
        context['format_choices'] = Dataset.FORMAT_CHOICES
        context['current_filters'] = {
            'status': self.request.GET.get('status', ''),
            'format': self.request.GET.get('format', ''),
            'search': self.request.GET.get('search', '')
        }
        return context

class DatasetDetailView(DetailView):
    """Detail view for datasets"""
    model = Dataset
    template_name = 'dataset_manager/dataset_detail.html'
    context_object_name = 'dataset'
    
    def get_queryset(self):
        return Dataset.objects.filter(
            Q(created_by=self.request.user) | 
            Q(is_public=True) | 
            Q(allowed_users=self.request.user)
        ).distinct()
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        dataset = self.object
        
        # Get sample data for preview
        samples = dataset.samples.all()[:20]
        context['samples'] = samples
        
        # Get pipeline executions
        executions = dataset.pipeline_executions.all()[:10]
        context['executions'] = executions
        
        # Get available pipelines
        available_pipelines = DataPipeline.objects.filter(
            Q(created_by=self.request.user) | Q(is_public=True)
        ).distinct()
        context['available_pipelines'] = available_pipelines
        
        return context

@login_required
def dataset_upload(request):
    """Dataset upload wizard"""
    if request.method == 'POST':
        # Handle AJAX upload
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            try:
                # Process upload data
                upload_data = json.loads(request.body)
                # TODO: Implement async upload processing
                return JsonResponse({'status': 'success', 'message': 'Upload started'})
            except Exception as e:
                return JsonResponse({'status': 'error', 'message': str(e)})
        else:
            # Handle form upload
            # TODO: Implement form-based upload
            messages.success(request, 'Dataset upload started')
            return redirect('dataset_manager:dataset_list')
    
    # Get available schemas for the form
    schemas = AnnotationSchema.objects.filter(
        Q(created_by=request.user) | Q(is_public=True)
    ).distinct()
    
    context = {
        'schemas': schemas,
        'format_choices': Dataset.FORMAT_CHOICES
    }
    
    return render(request, 'dataset_manager/dataset_upload.html', context)

class SchemaListView(ListView):
    """List view for annotation schemas"""
    model = AnnotationSchema
    template_name = 'dataset_manager/schema_list.html'
    context_object_name = 'schemas'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = AnnotationSchema.objects.filter(
            Q(created_by=self.request.user) | Q(is_public=True)
        ).distinct()
        
        # Apply filters
        type_filter = self.request.GET.get('type')
        if type_filter:
            queryset = queryset.filter(type=type_filter)
            
        search = self.request.GET.get('search')
        if search:
            queryset = queryset.filter(
                Q(name__icontains=search) | 
                Q(description__icontains=search)
            )
        
        return queryset.order_by('-created_at')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['type_choices'] = AnnotationSchema.ANNOTATION_TYPES
        context['current_filters'] = {
            'type': self.request.GET.get('type', ''),
            'search': self.request.GET.get('search', '')
        }
        return context

@login_required
def schema_designer(request, schema_id=None):
    """Visual schema designer"""
    schema = None
    if schema_id:
        schema = get_object_or_404(
            AnnotationSchema,
            id=schema_id,
            created_by=request.user
        )
    
    if request.method == 'POST':
        # Handle schema save
        try:
            data = json.loads(request.body)
            # TODO: Implement schema saving logic
            return JsonResponse({'status': 'success', 'message': 'Schema saved'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    context = {
        'schema': schema,
        'annotation_types': AnnotationSchema.ANNOTATION_TYPES
    }
    
    return render(request, 'dataset_manager/schema_designer.html', context)

class PipelineListView(ListView):
    """List view for data pipelines"""
    model = DataPipeline
    template_name = 'dataset_manager/pipeline_list.html'
    context_object_name = 'pipelines'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = DataPipeline.objects.filter(
            Q(created_by=self.request.user) | Q(is_public=True)
        ).distinct()
        
        # Apply filters
        template_filter = self.request.GET.get('template')
        if template_filter == 'true':
            queryset = queryset.filter(is_template=True)
        elif template_filter == 'false':
            queryset = queryset.filter(is_template=False)
            
        search = self.request.GET.get('search')
        if search:
            queryset = queryset.filter(
                Q(name__icontains=search) | 
                Q(description__icontains=search)
            )
        
        return queryset.order_by('-created_at')

@login_required
def pipeline_editor(request, pipeline_id=None):
    """Visual pipeline editor"""
    pipeline = None
    if pipeline_id:
        pipeline = get_object_or_404(
            DataPipeline,
            id=pipeline_id,
            created_by=request.user
        )
    
    if request.method == 'POST':
        # Handle pipeline save
        try:
            data = json.loads(request.body)
            # TODO: Implement pipeline saving logic
            return JsonResponse({'status': 'success', 'message': 'Pipeline saved'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    context = {
        'pipeline': pipeline
    }
    
    return render(request, 'dataset_manager/pipeline_editor.html', context)

# ================== API ViewSets ==================

class AnnotationSchemaViewSet(viewsets.ModelViewSet):
    """API viewset for annotation schemas"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get_serializer_class(self):
        if self.action == 'list':
            return AnnotationSchemaListSerializer
        return AnnotationSchemaSerializer
    
    def get_queryset(self):
        return AnnotationSchema.objects.filter(
            Q(created_by=self.request.user) | Q(is_public=True)
        ).distinct()
    
    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)
    
    @action(detail=True, methods=['post'])
    def validate_data(self, request, pk=None):
        """Validate data against this schema"""
        schema = self.get_object()
        serializer = SchemaValidationSerializer(data=request.data)
        
        if serializer.is_valid():
            validator = SchemaValidator(schema)
            validation_result = validator.validate(serializer.validated_data['data'])
            return Response(validation_result)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class DatasetViewSet(viewsets.ModelViewSet):
    """API viewset for datasets"""
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    
    def get_serializer_class(self):
        if self.action == 'create':
            return DatasetUploadSerializer
        elif self.action == 'list':
            return DatasetListSerializer
        return DatasetSerializer
    
    def get_queryset(self):
        return Dataset.objects.filter(
            Q(created_by=self.request.user) | 
            Q(is_public=True) | 
            Q(allowed_users=self.request.user)
        ).distinct()
    
    def perform_create(self, serializer):
        # Handle file upload and processing
        dataset = serializer.save(created_by=self.request.user)
        
        # Start async processing
        processor = DatasetProcessor(dataset)
        processor.start_processing()
    
    @action(detail=True, methods=['get'])
    def samples(self, request, pk=None):
        """Get dataset samples"""
        dataset = self.get_object()
        samples = dataset.samples.all()
        
        # Apply pagination
        paginator = Paginator(samples, 50)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        
        serializer = DatasetSampleListSerializer(page_obj, many=True)
        return Response({
            'results': serializer.data,
            'count': paginator.count,
            'num_pages': paginator.num_pages,
            'current_page': page_obj.number
        })
    
    @action(detail=True, methods=['get'])
    def statistics(self, request, pk=None):
        """Get detailed dataset statistics"""
        dataset = self.get_object()
        # TODO: Generate comprehensive statistics
        stats = {
            'total_samples': dataset.total_samples,
            'class_distribution': dataset.class_distribution,
            'file_size': dataset.file_size_bytes,
            'quality_score': dataset.quality_score
        }
        return Response(stats)
    
    @action(detail=True, methods=['post'])
    def mark_ready(self, request, pk=None):
        """Mark dataset as ready for training"""
        dataset = self.get_object()
        if dataset.created_by != request.user:
            return Response(
                {'error': 'Permission denied'}, 
                status=status.HTTP_403_FORBIDDEN
            )
        
        dataset.training_ready = True
        dataset.save(update_fields=['training_ready'])
        
        return Response({'status': 'Dataset marked as ready'})

class DataPipelineViewSet(viewsets.ModelViewSet):
    """API viewset for data pipelines"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get_serializer_class(self):
        if self.action == 'list':
            return DataPipelineListSerializer
        return DataPipelineSerializer
    
    def get_queryset(self):
        return DataPipeline.objects.filter(
            Q(created_by=self.request.user) | Q(is_public=True)
        ).distinct()
    
    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)
    
    @action(detail=True, methods=['post'])
    def execute(self, request, pk=None):
        """Execute pipeline on a dataset"""
        pipeline = self.get_object()
        serializer = PipelineExecutionCreateSerializer(
            data=request.data,
            context={'request': request}
        )
        
        if serializer.is_valid():
            dataset = Dataset.objects.get(id=serializer.validated_data['dataset_id'])
            
            # Create execution record
            execution = PipelineExecution.objects.create(
                pipeline=pipeline,
                dataset=dataset,
                parameters=serializer.validated_data.get('parameters', {}),
                executed_by=request.user
            )
            
            # Start async execution
            executor = PipelineExecutor(execution)
            executor.start_execution()
            
            response_serializer = PipelineExecutionSerializer(execution)
            return Response(response_serializer.data, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['post'])
    def validate(self, request, pk=None):
        """Validate pipeline configuration"""
        pipeline = self.get_object()
        # TODO: Implement pipeline validation logic
        return Response({'status': 'Pipeline validation completed'})

class PipelineExecutionViewSet(viewsets.ReadOnlyModelViewSet):
    """API viewset for pipeline executions (read-only)"""
    serializer_class = PipelineExecutionSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return PipelineExecution.objects.filter(
            executed_by=self.request.user
        )
    
    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        """Cancel running execution"""
        execution = self.get_object()
        if execution.status in ['pending', 'running']:
            execution.status = 'cancelled'
            execution.save(update_fields=['status'])
            # TODO: Stop actual processing
            return Response({'status': 'Execution cancelled'})
        
        return Response(
            {'error': 'Cannot cancel execution in current state'},
            status=status.HTTP_400_BAD_REQUEST
        )
