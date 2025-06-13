"""
API serializers for ML Manager.
"""

from rest_framework import serializers
from ..models import MLModel, Prediction


class MLModelSerializer(serializers.ModelSerializer):
    """Serializer for MLModel."""
    
    class Meta:
        model = MLModel
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at', 'status')


class PredictionSerializer(serializers.ModelSerializer):
    """Serializer for Prediction."""
    
    class Meta:
        model = Prediction
        fields = '__all__'
        read_only_fields = ('created_at',)
