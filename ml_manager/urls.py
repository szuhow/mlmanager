from django.urls import path
from . import views

app_name = 'ml_manager'

urlpatterns = [
    path('', views.ModelListView.as_view(), name='model-list'),
    path('model/<int:pk>/', views.ModelDetailView.as_view(), name='model-detail'),
    path('model/<int:pk>/predictions/', views.ModelPredictionListView.as_view(), name='model-predictions'),
    path('start-training/', views.StartTrainingView.as_view(), name='start-training'),
    path('model/<int:model_id>/stop/', views.stop_training, name='stop-training'),
    path('model/<int:pk>/delete/', views.ModelDeleteView.as_view(), name='model-delete'),
    path('models/batch-delete/', views.batch_delete_models, name='batch-delete-models'),
    path('model/<int:pk>/inference/', views.ModelInferenceView.as_view(), name='model-inference'),
    path('model/<int:pk>/save-as-template/', views.SaveAsTemplateView.as_view(), name='save-as-template'),
    
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
    
    # Training log URL
    path('model/<int:model_id>/logs/', views.get_training_log, name='model-training-log'),
    
    # Training preview image URL
    path('model/<int:model_id>/training-preview/<str:filename>/', views.serve_training_preview_image, name='training-preview-image'),
]
