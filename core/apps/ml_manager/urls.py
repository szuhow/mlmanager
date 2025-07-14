from django.urls import path
from . import views
from . import views_model_visualization

app_name = 'ml_manager'

urlpatterns = [
    # Web interface URLs
    path('', views.ModelListView.as_view(), name='model-list'),
    path('model/<int:pk>/', views.ModelDetailView.as_view(), name='model-detail'),
    path('model/<int:pk>/predictions/', views.ModelPredictionListView.as_view(), name='model-predictions'),
    path('start-training/', views.StartTrainingView.as_view(), name='start-training'),
    path('model/<int:model_id>/stop/', views.stop_training, name='stop-training'),
    path('model/<int:pk>/delete/', views.ModelDeleteView.as_view(), name='model-delete'),
    path('models/batch-delete/', views.batch_delete_models, name='batch-delete-models'),
    path('model/<int:pk>/inference/', views.ModelInferenceView.as_view(), name='model-inference'),
    path('inference/', views.GeneralInferenceView.as_view(), name='general-inference'),
    path('inference/results/', views.InferenceResultListView.as_view(), name='inference-results'),
    path('inference/result/<int:pk>/', views.InferenceResultView.as_view(), name='inference-result'),
    path('model/<int:pk>/save-as-template/', views.SaveAsTemplateView.as_view(), name='save-as-template'),
    
    # Model Architecture Visualization URLs
    path('architecture/', views_model_visualization.model_architecture_dashboard, name='model-architecture-dashboard'),
    path('api/models/visualize/', views_model_visualization.create_model_visualization, name='api-model-visualize'),
    path('api/models/compare/', views_model_visualization.compare_models_view, name='api-model-compare'),
    path('api/models/templates/', views_model_visualization.get_model_templates, name='api-model-templates'),
    path('api/models/download-architecture/<str:model_config_b64>/', views_model_visualization.download_model_architecture, name='api-download-architecture'),
    path('api/docs/', views_model_visualization.model_architecture_api_docs, name='api-docs'),
    
    # Training template URLs
    path('templates/', views.TrainingTemplateListView.as_view(), name='template-list'),
    path('templates/create/', views.TrainingTemplateCreateView.as_view(), name='template-create'),
    path('templates/<int:pk>/', views.TrainingTemplateDetailView.as_view(), name='template-detail'),
    path('templates/<int:pk>/edit/', views.TrainingTemplateUpdateView.as_view(), name='template-edit'),
    path('templates/<int:pk>/delete/', views.TrainingTemplateDeleteView.as_view(), name='template-delete'),
    path('api/template/<int:template_id>/', views.get_template_data, name='get-template-data'),
    
    # MLflow Model Registry URLs
    path('model/<int:pk>/registry/register/', views.register_model_in_registry, name='register-model'),
    path('model/<int:pk>/registry/transition/', views.transition_model_stage, name='transition-stage'),
    path('model/<int:pk>/registry/sync/', views.sync_registry_info, name='sync-registry'),
    path('registry/', views.registry_models_list, name='registry-list'),
    
    # MLflow Dashboard redirect
    path('mlflow/', views.mlflow_redirect_view, name='mlflow-dashboard'),
    
    # Training log URL
    path('model/<int:model_id>/logs/', views.get_training_log, name='model-training-log'),
    path('model/<int:model_id>/logs/realtime/', views.get_realtime_logs, name='model-realtime-logs'),
    
    # Training progress URL
    path('model/<int:model_id>/progress/', views.get_training_progress, name='model-progress'),
    
    # API endpoints for training monitoring
    path('api/training-progress/<int:model_id>/', views.get_training_progress, name='api-training-progress'),
    path('api/latest-training-model/', views.get_latest_training_model, name='api-latest-training-model'),
    path('api/stop-training/<int:model_id>/', views.stop_training_api, name='api-stop-training'),
    
    # MLflow synchronization endpoints
    path('api/sync-mlflow-status/', views.sync_mlflow_status, name='api-sync-mlflow-status'),
    path('api/cleanup-orphaned-runs/', views.cleanup_orphaned_runs, name='api-cleanup-orphaned-runs'),
    path('api/sync-all-mlflow-data/', views.sync_all_mlflow_data, name='api-sync-all-mlflow-data'),
    path('api/force-end-run/<str:run_id>/', views.force_end_mlflow_run, name='api-force-end-run'),
    path('api/force-end-all-runs/', views.force_end_all_mlflow_runs, name='api-force-end-all-runs'),
    path('api/test-mlflow/', views.test_mlflow_connection, name='api-test-mlflow'),
    
    # MLflow experiments management
    path('api/mlflow-experiments/', views.get_mlflow_experiments, name='api-get-mlflow-experiments'),
    path('api/create-mlflow-experiment/', views.create_mlflow_experiment_api, name='api-create-mlflow-experiment'),
    
    # Model preview endpoint
    path('api/model-summary/', views.generate_model_summary_api, name='api-model-summary'),
    
    # Model checkpoints API
    path('api/model-checkpoints/', views.get_model_checkpoints_api, name='api-model-checkpoints'),
    
    # Training preview image URL  
    path('model/<int:model_id>/training-preview/<str:filename>', views.serve_training_preview_image, name='training-preview-image'),
    
    # Dataset preview URL
    path('dataset-preview/', views.dataset_preview_view, name='dataset-preview'),
    
    # Serve preview image URL
    path('serve-preview-image/', views.serve_preview_image, name='serve-preview-image'),
    
    # Preprocessing preview URL
    path('preprocessing-preview/', views.preprocessing_preview, name='preprocessing-preview'),
]
