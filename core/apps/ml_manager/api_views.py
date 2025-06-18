# Training API Views
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.contrib.auth.models import User

from core.apps.ml_manager.models import MLModel
from core.apps.dataset_manager.models import Dataset
from core.apps.ml_manager.serializers import MLModelSerializer

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def start_training_api(request):
    """
    Start training via API
    
    POST /api/training/start/
    {
        "model_name": "My Model",
        "model_type": "classification",
        "dataset_id": 1,
        "hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        },
        "architecture": "resnet50",
        "description": "Training description"
    }
    """
    try:
        data = request.data
        
        # Validate required fields
        required_fields = ['model_name', 'model_type', 'dataset_id']
        for field in required_fields:
            if field not in data:
                return Response({
                    'error': f'Missing required field: {field}'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get dataset
        try:
            dataset = Dataset.objects.get(id=data['dataset_id'])
        except Dataset.DoesNotExist:
            return Response({
                'error': f'Dataset with ID {data["dataset_id"]} not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # Check if dataset is ready
        if dataset.status != 'ready':
            return Response({
                'error': f'Dataset is not ready for training. Current status: {dataset.status}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Create model
        model = MLModel.objects.create(
            name=data['model_name'],
            model_type=data['model_type'],
            architecture=data.get('architecture', 'resnet50'),
            description=data.get('description', ''),
            version="1.0.0",
            status='training'
        )
        
        # Since we don't have TrainingSession model, we'll store training info in model
        model.training_data_info = {
            'dataset_id': data['dataset_id'],
            'hyperparameters': data.get('hyperparameters', {}),
            'created_by': request.user.username
        }
        model.save()
        
        # Start training (simplified for now)
        # TODO: Implement actual training pipeline
        
        return Response({
            'success': True,
            'message': 'Training started successfully',
            'model_id': model.id,
            'model': MLModelSerializer(model).data
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        return Response({
            'error': f'Error starting training: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def training_status_api(request, model_id):
    """
    Get model status via API
    
    GET /api/ml/training/status/{model_id}/
    """
    try:
        model = get_object_or_404(MLModel, id=model_id)
        
        return Response({
            'success': True,
            'model': MLModelSerializer(model).data
        })
        
    except Exception as e:
        return Response({
            'error': f'Error getting model status: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def stop_training_api(request, model_id):
    """
    Stop training via API
    
    POST /api/ml/training/stop/{model_id}/
    """
    try:
        model = get_object_or_404(MLModel, id=model_id)
        
        # Stop training by updating model status
        if model.status in ['training', 'pending']:
            model.status = 'cancelled'
            model.save()
            
            return Response({
                'success': True,
                'message': 'Training stopped successfully'
            })
        else:
            return Response({
                'error': f'Cannot stop training in status: {model.status}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
    except Exception as e:
        return Response({
            'error': f'Error stopping training: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_models_api(request):
    """
    List available models via API
    
    GET /api/models/
    """
    try:
        models = MLModel.objects.all().order_by('-created_at')
        
        return Response({
            'success': True,
            'models': MLModelSerializer(models, many=True).data
        })
        
    except Exception as e:
        return Response({
            'error': f'Error listing models: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_datasets_api(request):
    """
    List available datasets via API
    
    GET /api/datasets/
    """
    try:
        datasets = Dataset.objects.filter(
            created_by=request.user,
            status='ready'
        ).order_by('-created_at')
        
        from core.apps.dataset_manager.serializers import DatasetListSerializer
        
        return Response({
            'success': True,
            'datasets': DatasetListSerializer(datasets, many=True).data
        })
        
    except Exception as e:
        return Response({
            'error': f'Error listing datasets: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
