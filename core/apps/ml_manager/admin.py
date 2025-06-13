from django.contrib import admin
from .models import MLModel, Prediction

@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'status', 'created_at', 'updated_at')
    list_filter = ('status', 'created_at')
    search_fields = ('name', 'description', 'mlflow_run_id')
    readonly_fields = ('created_at', 'updated_at')

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('model', 'created_at', 'processing_time')
    list_filter = ('model', 'created_at')
    readonly_fields = ('created_at', 'processing_time')
