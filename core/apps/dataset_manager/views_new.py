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
from django.utils import timezone
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
    # Get user's datasets queryset (without slicing first)
    user_datasets_qs = Dataset.objects.filter(
        Q(created_by=request.user) | Q(is_public=True) | Q(allowed_users=request.user)
    ).distinct()
    
    # Get user's schemas queryset (without slicing first)
    user_schemas_qs = AnnotationSchema.objects.filter(
        Q(created_by=request.user) | Q(is_public=True)
    ).distinct()
    
    # Get recent pipeline executions queryset (without slicing first)
    recent_executions_qs = PipelineExecution.objects.filter(
        executed_by=request.user
    )
    
    # Statistics (calculate before slicing)
    stats = {
        'total_datasets': user_datasets_qs.count(),
        'ready_datasets': user_datasets_qs.filter(status='ready').count(),
        'total_schemas': user_schemas_qs.count(),
        'running_executions': recent_executions_qs.filter(status='running').count(),
        'datasets_this_month': user_datasets_qs.filter(created_at__gte=timezone.now().replace(day=1)).count(),
        'schemas_this_month': user_schemas_qs.filter(created_at__gte=timezone.now().replace(day=1)).count(),
        'completed_executions': recent_executions_qs.filter(status='completed').count(),
        'failed_executions': recent_executions_qs.filter(status='failed').count(),
        'total_samples': DatasetSample.objects.filter(dataset__in=user_datasets_qs).count(),
        'avg_samples_per_dataset': user_datasets_qs.aggregate(avg=Count('datasetsample'))['avg'] or 0,
    }
    
    # Now slice for display (after statistics are calculated)
    user_datasets = user_datasets_qs[:5]
    user_schemas = user_schemas_qs[:5]
    recent_executions = recent_executions_qs[:5]
    
    context = {
        'datasets': user_datasets,
        'schemas': user_schemas,
        'recent_executions': recent_executions,
        'stats': stats
    }
    
    return render(request, 'dataset_manager/home.html', context)

# ================== Dataset Views ==================

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
        return context

@login_required
def dataset_list(request):
    """List all datasets accessible to the user"""
    datasets = Dataset.objects.filter(
        Q(created_by=request.user) | Q(is_public=True) | Q(allowed_users=request.user)
    ).distinct()
    
    return render(request, 'dataset_manager/dataset_list.html', {
        'datasets': datasets
    })

@login_required
def dataset_upload(request):
    """Dataset upload wizard"""
    from .forms import DatasetUploadForm
    from django.core.files.storage import default_storage
    from django.core.files.base import ContentFile
    
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES, user=request.user)
        if form.is_valid():
            # Save the dataset object
            dataset = form.save()
            
            # Handle the uploaded file
            uploaded_file = request.FILES['dataset_file']
            
            # Create directory for this dataset
            dataset_dir = f'datasets/{dataset.id}_{dataset.name}'
            os.makedirs(os.path.join('data', dataset_dir), exist_ok=True)
            
            # Save the original file
            file_path = os.path.join(dataset_dir, uploaded_file.name)
            saved_path = default_storage.save(file_path, ContentFile(uploaded_file.read()))
            
            # Update dataset with file information
            dataset.original_filename = uploaded_file.name
            dataset.file_path = saved_path
            dataset.file_size_bytes = uploaded_file.size
            dataset.status = 'extracting'
            dataset.save()
            
            # Process the dataset asynchronously (for now, just count files)
            try:
                import zipfile
                import tarfile
                
                full_path = os.path.join('data', saved_path)
                sample_count = 0
                
                if uploaded_file.name.endswith('.zip'):
                    with zipfile.ZipFile(full_path, 'r') as zip_ref:
                        # Count image files
                        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm'}
                        sample_count = sum(1 for name in zip_ref.namelist() 
                                         if any(name.lower().endswith(ext) for ext in image_extensions))
                elif uploaded_file.name.endswith(('.tar', '.tar.gz')):
                    with tarfile.open(full_path, 'r:*') as tar_ref:
                        # Count image files
                        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm'}
                        sample_count = sum(1 for member in tar_ref.getmembers() 
                                         if member.isfile() and any(member.name.lower().endswith(ext) for ext in image_extensions))
                elif uploaded_file.name.endswith('.csv'):
                    # For CSV, count rows (excluding header)
                    import pandas as pd
                    df = pd.read_csv(full_path)
                    sample_count = len(df)
                
                # Update sample count
                dataset.total_samples = sample_count
                dataset.status = 'ready'
                dataset.save()
                
                messages.success(request, f'Dataset "{dataset.name}" uploaded successfully! Found {sample_count} samples.')
                
            except Exception as e:
                dataset.status = 'error'
                dataset.save()
                messages.error(request, f'Error processing dataset: {str(e)}')
            
            return redirect('dataset_manager:dataset_detail', dataset_id=dataset.id)
    else:
        form = DatasetUploadForm(user=request.user)
    
    # Get available schemas for the form
    schemas = AnnotationSchema.objects.filter(
        Q(created_by=request.user) | Q(is_public=True)
    ).distinct()
    
    context = {
        'form': form,
        'schemas': schemas,
        'format_choices': Dataset.FORMAT_CHOICES
    }
    
    return render(request, 'dataset_manager/dataset_upload.html', context)

@login_required 
def dataset_detail(request, dataset_id):
    """Dataset detail view"""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    # TODO: Check permissions
    
    return render(request, 'dataset_manager/dataset_detail.html', {
        'dataset': dataset
    })

@login_required
def dataset_preview(request, dataset_id):
    """Preview dataset samples"""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Get first few samples for preview
    samples = DatasetSample.objects.filter(dataset=dataset)[:10]
    
    return render(request, 'dataset_manager/dataset_preview.html', {
        'dataset': dataset,
        'samples': samples
    })

@login_required
def dataset_samples(request, dataset_id):
    """List dataset samples"""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    samples = DatasetSample.objects.filter(dataset=dataset)
    
    # Pagination
    paginator = Paginator(samples, 50)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'dataset_manager/dataset_samples.html', {
        'dataset': dataset,
        'samples': page_obj
    })

@login_required
def dataset_edit(request, dataset_id):
    """Edit dataset"""
    dataset = get_object_or_404(Dataset, id=dataset_id, created_by=request.user)
    
    if request.method == 'POST':
        from .forms import DatasetUploadForm
        form = DatasetUploadForm(request.POST, instance=dataset, user=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, f'Dataset "{dataset.name}" updated successfully!')
            return redirect('dataset_manager:dataset_detail', dataset_id=dataset.id)
    else:
        from .forms import DatasetUploadForm
        form = DatasetUploadForm(instance=dataset, user=request.user)
    
    return render(request, 'dataset_manager/dataset_edit.html', {
        'dataset': dataset,
        'form': form
    })

@login_required
@require_POST
def dataset_delete(request, dataset_id):
    """Delete dataset"""
    dataset = get_object_or_404(Dataset, id=dataset_id, created_by=request.user)
    
    # Store dataset name for message
    dataset_name = dataset.name
    
    # Delete associated files
    try:
        if dataset.file_path and os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)
        if dataset.extracted_path and os.path.exists(dataset.extracted_path):
            import shutil
            shutil.rmtree(dataset.extracted_path)
    except Exception as e:
        messages.warning(request, f"Dataset deleted but couldn't remove files: {str(e)}")
    
    # Delete dataset from database
    dataset.delete()
    
    messages.success(request, f'Dataset "{dataset_name}" deleted successfully!')
    return redirect('dataset_manager:dataset_list')

# ================== Schema Views ==================

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

@login_required
def schema_list(request):
    """List all schemas"""
    schemas = AnnotationSchema.objects.filter(
        Q(created_by=request.user) | Q(is_public=True)
    ).distinct()
    
    return render(request, 'dataset_manager/schema_list.html', {
        'schemas': schemas
    })

@login_required
def schema_create(request):
    """Create new schema"""
    from .forms import AnnotationSchemaForm
    
    if request.method == 'POST':
        form = AnnotationSchemaForm(request.POST, user=request.user)
        if form.is_valid():
            schema = form.save()
            messages.success(request, f'Schema "{schema.name}" created successfully!')
            return redirect('dataset_manager:schema_detail', pk=schema.pk)
    else:
        form = AnnotationSchemaForm(user=request.user)
    
    return render(request, 'dataset_manager/schema_create.html', {'form': form})

@login_required
def schema_detail(request, pk):
    """Schema detail view"""
    schema = get_object_or_404(AnnotationSchema, id=pk)
    return render(request, 'dataset_manager/schema_detail.html', {'schema': schema})

@login_required
def schema_edit(request, pk):
    """Edit schema"""
    schema = get_object_or_404(AnnotationSchema, id=pk, created_by=request.user)
    
    if request.method == 'POST':
        from .forms import AnnotationSchemaForm
        form = AnnotationSchemaForm(request.POST, instance=schema, user=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, f'Schema "{schema.name}" updated successfully!')
            return redirect('dataset_manager:schema_detail', pk=schema.pk)
    else:
        from .forms import AnnotationSchemaForm
        form = AnnotationSchemaForm(instance=schema, user=request.user)
    
    return render(request, 'dataset_manager/schema_edit.html', {
        'schema': schema,
        'form': form
    })

@login_required
@require_POST
def schema_delete(request, pk):
    """Delete schema"""
    schema = get_object_or_404(AnnotationSchema, id=pk, created_by=request.user)
    schema_name = schema.name
    schema.delete()
    messages.success(request, f'Schema "{schema_name}" deleted successfully!')
    return redirect('dataset_manager:schema_list')

@login_required
def schema_preview(request, pk):
    """Preview schema"""
    schema = get_object_or_404(AnnotationSchema, id=pk)
    return render(request, 'dataset_manager/schema_preview.html', {'schema': schema})

# ================== Pipeline Views ==================

@login_required
def pipeline_list(request):
    """List all pipelines"""
    pipelines = DataPipeline.objects.filter(
        Q(created_by=request.user) | Q(is_public=True)
    ).distinct()
    
    return render(request, 'dataset_manager/pipeline_list.html', {
        'pipelines': pipelines
    })

@login_required
def pipeline_create(request):
    """Create new pipeline"""
    from .forms import DataPipelineForm
    
    if request.method == 'POST':
        form = DataPipelineForm(request.POST, user=request.user)
        if form.is_valid():
            pipeline = form.save()
            messages.success(request, f'Pipeline "{pipeline.name}" created successfully!')
            return redirect('dataset_manager:pipeline_detail', pk=pipeline.pk)
    else:
        form = DataPipelineForm(user=request.user)
    
    return render(request, 'dataset_manager/pipeline_create.html', {'form': form})

@login_required
def pipeline_detail(request, pk):
    """Pipeline detail view"""
    pipeline = get_object_or_404(DataPipeline, id=pk)
    return render(request, 'dataset_manager/pipeline_detail.html', {'pipeline': pipeline})

@login_required
def pipeline_execute(request, pk):
    """Execute pipeline"""
    pipeline = get_object_or_404(DataPipeline, id=pk)
    
    if request.method == 'POST':
        # Create execution record
        execution = PipelineExecution.objects.create(
            pipeline=pipeline,
            executed_by=request.user,
            status='pending',
            parameters=request.POST.dict()
        )
        
        # TODO: Execute pipeline asynchronously
        execution.status = 'running'
        execution.save()
        
        messages.success(request, f'Pipeline execution started! ID: {execution.id}')
        return redirect('dataset_manager:execution_detail', pk=execution.id)
    
    return JsonResponse({'status': 'success', 'execution_id': 1})

@login_required
def pipeline_clone(request, pk):
    """Clone pipeline"""
    original = get_object_or_404(DataPipeline, id=pk)
    
    # Create a copy
    clone = DataPipeline.objects.create(
        name=f"{original.name} (Copy)",
        description=original.description,
        version="1.0",
        pipeline_graph=original.pipeline_graph,
        input_formats=original.input_formats,
        output_format=original.output_format,
        default_parameters=original.default_parameters,
        required_parameters=original.required_parameters,
        created_by=request.user
    )
    
    messages.success(request, f'Pipeline cloned as "{clone.name}"!')
    return redirect('dataset_manager:pipeline_detail', pk=clone.id)

# ================== Execution Views ==================

@login_required
def execution_list(request):
    """List all executions"""
    executions = PipelineExecution.objects.filter(executed_by=request.user)
    return render(request, 'dataset_manager/execution_list.html', {'executions': executions})

@login_required
def execution_detail(request, pk):
    """Execution detail view"""
    execution = get_object_or_404(PipelineExecution, id=pk)
    return render(request, 'dataset_manager/execution_detail.html', {'execution': execution})

@login_required
def execution_logs(request, pk):
    """View execution logs"""
    execution = get_object_or_404(PipelineExecution, id=pk)
    logs = ["Sample log entry 1", "Sample log entry 2", "Sample log entry 3"]
    return render(request, 'dataset_manager/execution_logs.html', {
        'execution': execution,
        'logs': logs
    })

@login_required
@require_POST
def execution_stop(request, pk):
    """Stop execution"""
    execution = get_object_or_404(PipelineExecution, id=pk, executed_by=request.user)
    execution.status = 'cancelled'
    execution.save()
    messages.success(request, 'Execution stopped successfully!')
    return JsonResponse({'status': 'success'})

# ================== AJAX Views ==================

@csrf_exempt
def validate_schema_ajax(request):
    """AJAX schema validation"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            # TODO: Implement schema validation
            return JsonResponse({'valid': True, 'message': 'Schema is valid'})
        except Exception as e:
            return JsonResponse({'valid': False, 'message': str(e)})
    
    return JsonResponse({'valid': True})

@csrf_exempt
def process_upload_ajax(request):
    """AJAX file upload processing"""
    if request.method == 'POST':
        try:
            # TODO: Implement file processing
            return JsonResponse({
                'status': 'success',
                'samples': 247,  # Will be replaced with real data
                'classes': ['class1', 'class2'],
                'format': 'detected_format'
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'success'})

@csrf_exempt
def pipeline_nodes_ajax(request):
    """AJAX pipeline nodes"""
    nodes = [
        {'id': 'input', 'name': 'Input', 'type': 'input'},
        {'id': 'resize', 'name': 'Resize', 'type': 'transform'},
        {'id': 'output', 'name': 'Output', 'type': 'output'}
    ]
    return JsonResponse({'nodes': nodes})

@csrf_exempt
def execution_status_ajax(request, pk):
    """AJAX execution status"""
    try:
        execution = PipelineExecution.objects.get(id=pk)
        return JsonResponse({
            'status': execution.status,
            'progress': 50,  # TODO: Calculate real progress
            'logs': ['Sample log entry']
        })
    except PipelineExecution.DoesNotExist:
        return JsonResponse({'status': 'not_found'})

# ================== ViewSets for API ==================

class AnnotationSchemaViewSet(viewsets.ModelViewSet):
    """ViewSet for managing annotation schemas"""
    queryset = AnnotationSchema.objects.all()
    serializer_class = AnnotationSchemaSerializer
    
    def list(self, request):
        """List all annotation schemas"""
        schemas = self.get_queryset()
        serializer = self.get_serializer(schemas, many=True)
        return Response(serializer.data)
    
    def retrieve(self, request, pk=None):
        """Get specific annotation schema"""
        try:
            schema = self.get_object()
            serializer = self.get_serializer(schema)
            return Response(serializer.data)
        except AnnotationSchema.DoesNotExist:
            return Response({'error': 'Schema not found'}, status=404)

class DatasetViewSet(viewsets.ModelViewSet):
    """ViewSet for managing datasets"""
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    
    def list(self, request):
        """List all datasets"""
        datasets = self.get_queryset()
        serializer = self.get_serializer(datasets, many=True)
        return Response(serializer.data)
    
    def retrieve(self, request, pk=None):
        """Get specific dataset"""
        try:
            dataset = self.get_object()
            serializer = self.get_serializer(dataset)
            return Response(serializer.data)
        except Dataset.DoesNotExist:
            return Response({'error': 'Dataset not found'}, status=404)
    
    @action(detail=True, methods=['get'])
    def samples(self, request, pk=None):
        """Get samples for a dataset"""
        dataset = self.get_object()
        samples = DatasetSample.objects.filter(dataset=dataset)
        serializer = DatasetSampleSerializer(samples, many=True)
        return Response(serializer.data)

class DataPipelineViewSet(viewsets.ModelViewSet):
    """ViewSet for managing data pipelines"""
    queryset = DataPipeline.objects.all()
    serializer_class = DataPipelineSerializer
    
    def list(self, request):
        """List all pipelines"""
        pipelines = self.get_queryset()
        serializer = self.get_serializer(pipelines, many=True)
        return Response(serializer.data)
    
    def retrieve(self, request, pk=None):
        """Get specific pipeline"""
        try:
            pipeline = self.get_object()
            serializer = self.get_serializer(pipeline)
            return Response(serializer.data)
        except DataPipeline.DoesNotExist:
            return Response({'error': 'Pipeline not found'}, status=404)

class PipelineExecutionViewSet(viewsets.ModelViewSet):
    """ViewSet for managing pipeline executions"""
    queryset = PipelineExecution.objects.all()
    serializer_class = PipelineExecutionSerializer
    
    def list(self, request):
        """List all executions"""
        executions = self.get_queryset()
        serializer = self.get_serializer(executions, many=True)
        return Response(serializer.data)
    
    def retrieve(self, request, pk=None):
        """Get specific execution"""
        try:
            execution = self.get_object()
            serializer = self.get_serializer(execution)
            return Response(serializer.data)
        except PipelineExecution.DoesNotExist:
            return Response({'error': 'Execution not found'}, status=404)

class DatasetSampleViewSet(viewsets.ModelViewSet):
    """ViewSet for managing dataset samples"""
    queryset = DatasetSample.objects.all()
    serializer_class = DatasetSampleSerializer
    
    def list(self, request):
        """List all dataset samples"""
        samples = self.get_queryset()
        serializer = self.get_serializer(samples, many=True)
        return Response(serializer.data)
    
    def retrieve(self, request, pk=None):
        """Get specific dataset sample"""
        try:
            sample = self.get_object()
            serializer = self.get_serializer(sample)
            return Response(serializer.data)
        except DatasetSample.DoesNotExist:
            return Response({'error': 'Sample not found'}, status=404)
