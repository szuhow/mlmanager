# API URLs for ML Manager
from django.urls import path
from . import api_views

app_name = 'ml_manager_api'

urlpatterns = [
    # Training API
    path('training/start/', api_views.start_training_api, name='start_training'),
    path('training/status/<int:model_id>/', api_views.training_status_api, name='training_status'),
    path('training/stop/<int:model_id>/', api_views.stop_training_api, name='stop_training'),
    
    # Models and Datasets API
    path('models/', api_views.list_models_api, name='list_models'),
    path('datasets/', api_views.list_datasets_api, name='list_datasets'),
]
