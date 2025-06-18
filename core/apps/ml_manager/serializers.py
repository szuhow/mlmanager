# ML Manager Serializers
from rest_framework import serializers
from .models import MLModel, Prediction, TrainingTemplate

class MLModelSerializer(serializers.ModelSerializer):
    """Serializer for MLModel"""
    
    class Meta:
        model = MLModel
        fields = [
            'id', 'name', 'description', 'version', 'unique_identifier',
            'model_family', 'model_type', 'created_at', 'updated_at',
            'mlflow_run_id', 'registry_model_name', 'registry_model_version',
            'registry_stage', 'is_registered', 'model_directory',
            'model_weights_path', 'model_config_path', 'training_data_info',
            'model_architecture_info', 'performance_metrics', 'status'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'mlflow_run_id']

class PredictionSerializer(serializers.ModelSerializer):
    """Serializer for Prediction"""
    
    class Meta:
        model = Prediction
        fields = '__all__'
        read_only_fields = ['id', 'created_at']

class TrainingTemplateSerializer(serializers.ModelSerializer):
    """Serializer for TrainingTemplate"""
    
    class Meta:
        model = TrainingTemplate
        fields = '__all__'
        read_only_fields = ['id', 'created_at', 'updated_at']
