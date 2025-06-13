"""
API views for ML Manager.
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from ..models import MLModel, Prediction
from .serializers import MLModelSerializer, PredictionSerializer


class MLModelViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing ML models.
    """
    queryset = MLModel.objects.all()
    serializer_class = MLModelSerializer

    @action(detail=True, methods=['post'])
    def start_training(self, request, pk=None):
        """Start training for a specific model."""
        model = self.get_object()
        # Training logic here
        return Response({'status': 'training started'})

    @action(detail=True, methods=['get'])
    def training_status(self, request, pk=None):
        """Get training status for a specific model."""
        model = self.get_object()
        return Response({'status': model.status})


class PredictionViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing predictions.
    """
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer
