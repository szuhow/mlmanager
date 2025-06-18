# Dataset Manager URL Configuration

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

app_name = 'dataset_manager'

# Django REST Framework API Router
router = DefaultRouter()
router.register(r'schemas', views.AnnotationSchemaViewSet, basename='schema')
router.register(r'datasets', views.DatasetViewSet, basename='dataset')
router.register(r'pipelines', views.DataPipelineViewSet, basename='pipeline')
router.register(r'executions', views.PipelineExecutionViewSet, basename='execution')
router.register(r'samples', views.DatasetSampleViewSet, basename='sample')

urlpatterns = [
    # Main dashboard
    path('', views.dataset_manager_home, name='home'),
    
    # Dataset management
    path('datasets/', views.dataset_list, name='dataset_list'),
    path('datasets/upload/', views.dataset_upload, name='dataset_upload'),
    path('datasets/<int:dataset_id>/', views.dataset_detail, name='dataset_detail'),
    path('datasets/<int:dataset_id>/preview/', views.dataset_preview, name='dataset_preview'),
    path('datasets/<int:dataset_id>/samples/', views.dataset_samples, name='dataset_samples'),
    path('datasets/<int:dataset_id>/edit/', views.dataset_edit, name='dataset_edit'),
    path('datasets/<int:dataset_id>/delete/', views.dataset_delete, name='dataset_delete'),
    
    # Schema management
    path('schemas/', views.schema_list, name='schema_list'),
    path('schemas/create/', views.schema_create, name='schema_create'),
    path('schemas/<int:schema_id>/', views.schema_detail, name='schema_detail'),
    path('schemas/<int:schema_id>/edit/', views.schema_edit, name='schema_edit'),
    path('schemas/<int:schema_id>/delete/', views.schema_delete, name='schema_delete'),
    path('schemas/<int:schema_id>/preview/', views.schema_preview, name='schema_preview'),
    
    # Pipeline management
    path('pipelines/', views.pipeline_list, name='pipeline_list'),
    path('pipelines/create/', views.pipeline_create, name='pipeline_create'),
    path('pipelines/<int:pipeline_id>/', views.pipeline_detail, name='pipeline_detail'),
    path('pipelines/<int:pipeline_id>/edit/', views.pipeline_editor, name='pipeline_editor'),
    path('pipelines/<int:pipeline_id>/execute/', views.pipeline_execute, name='pipeline_execute'),
    path('pipelines/<int:pipeline_id>/clone/', views.pipeline_clone, name='pipeline_clone'),
    
    # Execution monitoring
    path('executions/', views.execution_list, name='execution_list'),
    path('executions/<int:execution_id>/', views.execution_detail, name='execution_detail'),
    path('executions/<int:execution_id>/logs/', views.execution_logs, name='execution_logs'),
    path('executions/<int:execution_id>/stop/', views.execution_stop, name='execution_stop'),
    
    # AJAX/API endpoints
    path('ajax/validate-schema/', views.validate_schema_ajax, name='validate_schema_ajax'),
    path('ajax/process-upload/', views.process_upload_ajax, name='process_upload_ajax'),
    path('ajax/analyze-dataset/', views.analyze_dataset_ajax, name='analyze_dataset_ajax'),
    path('ajax/pipeline-nodes/', views.pipeline_nodes_ajax, name='pipeline_nodes_ajax'),
    path('ajax/execution-status/', views.execution_status_ajax, name='execution_status_ajax'),
    
    # API endpoints
    path('api/', include(router.urls)),
]
