from django.shortcuts import render, get_object_or_404, redirect
from django.views.generic import ListView, DetailView, FormView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.contrib import messages
from django.db import models
from django.core.files import File
from django.http import JsonResponse, HttpResponse
from django.utils.http import http_date
from .forms import TrainingForm, InferenceForm, EnhancedInferenceForm, TrainingTemplateForm
from .models import MLModel, Prediction, TrainingTemplate
import mlflow
import subprocess
import sys
import os
from pathlib import Path
import json
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
import logging
import tempfile
import time
import torch
import shutil
from PIL import Image
import re
import hashlib
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings
import numpy as np

# Create logger
logger = logging.getLogger(__name__)

# Import run_inference from ml.training.train
try:
    from ml.training.train import run_inference
except ImportError:
    # Fallback if import fails
    def run_inference(*args, **kwargs):
        raise ImportError("run_inference function not available")

# Create your views here.

class ModelListView(LoginRequiredMixin, ListView):
    model = MLModel
    template_name = 'ml_manager/model_list.html'
    context_object_name = 'models'
    paginate_by = 20

    def get_queryset(self):
        queryset = MLModel.objects.all()
        
        # Search functionality
        search_query = self.request.GET.get('search', '')
        if search_query:
            queryset = queryset.filter(
                models.Q(name__icontains=search_query) |
                models.Q(description__icontains=search_query) |
                models.Q(mlflow_run_id__icontains=search_query)
            )
        
        # Status filter
        status_filter = self.request.GET.get('status', '')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        # Sorting
        sort_by = self.request.GET.get('sort', '-created_at')
        allowed_sorts = ['created_at', '-created_at', 'name', '-name', '-best_val_dice', 'status']
        if sort_by in allowed_sorts:
            queryset = queryset.order_by(sort_by)
        else:
            queryset = queryset.order_by('-created_at')
        
        return queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Add MLflow experiments info
        try:
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()
            context['experiments'] = experiments
        except Exception as e:
            context['experiments'] = []
            
        # Calculate progress percentage for each model
        # Handle both paginated and non-paginated contexts
        object_list = context.get('object_list')
        if object_list is not None:
            # This is a paginated list view - process the paginated objects
            models_with_progress = []
            for model in object_list:
                # progress_percentage is already calculated by the model property
                models_with_progress.append(model)
            
            # Replace the object_list with our processed models
            context['object_list'] = models_with_progress
            # Also update 'models' context for template compatibility
            context['models'] = context['page_obj'] if 'page_obj' in context else models_with_progress
        
        # Add filter context
        context['search_query'] = self.request.GET.get('search', '')
        context['status_filter'] = self.request.GET.get('status', '')
        context['sort_by'] = self.request.GET.get('sort', '-created_at')
        
        return context

class ModelDetailView(LoginRequiredMixin, DetailView):
    model = MLModel
    template_name = 'ml_manager/model_detail.html'
    context_object_name = 'model'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Initialize mlflow_error to None by default
        context['mlflow_error'] = None
        
        # Get MLflow run info with enhanced error handling
        try:
            if self.object.mlflow_run_id:
                client = mlflow.tracking.MlflowClient()
                run = client.get_run(self.object.mlflow_run_id)
                context['run'] = run
            else:
                context['run'] = None
                context['mlflow_error'] = "No MLflow run ID associated with this model"
        except Exception as e:
            # Handle missing MLflow run gracefully
            logging.warning(f"MLflow run {self.object.mlflow_run_id} not found: {e}")
            context['run'] = None
            context['mlflow_error'] = f"MLflow run not found: {e}. Consider cleaning up orphaned references."
            
            # Optionally clear the orphaned run ID
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                logging.info(f"Clearing orphaned MLflow run ID for model {self.object.id}")
                self.object.mlflow_run_id = None
                self.object.save()

        # Get comprehensive training details
        context['training_details'] = self._get_training_details()
        
        # Get model architecture details
        context['architecture_details'] = self._get_architecture_details()

        # Get MLflow Model Registry information
        context['registry_info'] = None
        context['registry_error'] = None
        if self.object.is_registered and self.object.registry_model_name:
            try:
                from .utils.mlflow_utils import get_registered_model_info, get_model_version_details
                
                # Get general model info
                model_info = get_registered_model_info(self.object.registry_model_name)
                if model_info:
                    context['registry_info'] = model_info
                    
                    # Get specific version details
                    if self.object.registry_model_version:
                        version_details = get_model_version_details(
                            self.object.registry_model_name, 
                            self.object.registry_model_version
                        )
                        if version_details:
                            context['registry_version_info'] = version_details
                            
                            # Check if stage matches what's stored in Django
                            stored_stage = self.object.registry_stage or "None"
                            actual_stage = version_details.get('current_stage', 'None')
                            if stored_stage != actual_stage:
                                context['stage_mismatch'] = {
                                    'stored': stored_stage,
                                    'actual': actual_stage
                                }
                        
            except Exception as e:
                logging.warning(f"Failed to get registry info for model {self.object.id}: {e}")
                context['registry_error'] = f"Failed to fetch registry information: {e}"

        # progress_percentage is already calculated by the model property
        context['progress_percentage'] = round(self.object.progress_percentage, 1)

        # Get training logs with enhanced parsing
        try:
            # Use the existing method to get training logs
            log_lines = self._get_training_logs()
            
            if log_lines:
                # Enhanced log processing: categorize and parse logs
                parsed_logs = self._parse_enhanced_logs(log_lines)
                context['training_logs'] = log_lines
                context['parsed_logs'] = parsed_logs
                
                # Add log statistics
                context['log_stats'] = {
                    'total_count': len(log_lines),
                    'epoch_count': len(parsed_logs['epoch_logs']),
                    'batch_count': len(parsed_logs['batch_logs']),
                    'metrics_count': len(parsed_logs['metrics_logs']),
                    'validation_count': len(parsed_logs['validation_logs']),
                    'last_update': log_lines[-1] if log_lines else 'No logs yet'
                }
            else:
                context['training_logs'] = ['No training logs found for this model']
                context['parsed_logs'] = {
                    'epoch_logs': [],
                    'batch_logs': [],
                    'model_logs': [],
                    'config_logs': [],
                    'metrics_logs': [],
                    'validation_logs': [],
                    'general_logs': []
                }
                context['log_stats'] = {
                    'total_count': 0,
                    'epoch_count': 0,
                    'batch_count': 0,
                    'metrics_count': 0,
                    'validation_count': 0,
                    'last_update': 'No logs yet'
                }
        except Exception as e:
            context['training_logs'] = [f"Could not load logs: {e}"]
            context['parsed_logs'] = {
                'epoch_logs': [],
                'batch_logs': [],
                'model_logs': [],
                'config_logs': [],
                'metrics_logs': [],
                'validation_logs': [],
                'general_logs': [],
            }
            context['log_stats'] = {
                'total_count': 1,
                'epoch_count': 0,
                'batch_count': 0,
                'metrics_count': 0,
                'validation_count': 0,
                'last_update': f"Error: {e}"
            }

        # Get training preview images from MLflow artifacts
        context['training_preview'] = self._get_training_preview()
        
        # Get training details from model directory
        context['training_details'] = self._get_training_details()
        
        # Get model architecture details
        context['architecture_details'] = self._get_architecture_details()
        
        # Add missing template context variables
        if self.object.training_data_info:
            # Add total_samples if missing
            context['total_samples'] = self.object.training_data_info.get('total_samples', 'N/A')
            
            # Add training_config for backward compatibility
            context['training_config'] = {
                'parameters': self.object.training_data_info
            }
        else:
            context['total_samples'] = 'N/A'
            context['training_config'] = {'parameters': {}}
        
        # Generate MLflow UI URL for this model's run
        context['mlflow_ui_url'] = None
        if self.object.mlflow_run_id:
            try:
                from .utils.mlflow_utils import get_mlflow_ui_url
                context['mlflow_ui_url'] = get_mlflow_ui_url(run_id=self.object.mlflow_run_id)
            except Exception as e:
                logging.warning(f"Error generating MLflow URL for model {self.object.id}: {e}")
        
        return context

    def _parse_enhanced_logs(self, log_lines):
        """Parse logs into categories for better GUI display"""
        parsed = {
            'epoch_logs': [],
            'batch_logs': [],
            'model_logs': [],
            'config_logs': [],
            'metrics_logs': [],
            'validation_logs': [],
            'general_logs': []
        }
        
        for line in log_lines:
            if '[EPOCH]' in line:
                parsed['epoch_logs'].append(line)
            elif '[TRAIN]' in line or '[VAL]' in line:
                if '[VAL]' in line:
                    parsed['validation_logs'].append(line)
                else:
                    parsed['batch_logs'].append(line)
            elif '[MODEL]' in line:
                parsed['model_logs'].append(line)
            elif '[CONFIG]' in line:
                parsed['config_logs'].append(line)
            elif '[METRICS]' in line or '[STATS]' in line:
                parsed['metrics_logs'].append(line)
            else:
                parsed['general_logs'].append(line)
        
        return parsed

    def render_to_response(self, context, **response_kwargs):
        if self.request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            from django.http import JsonResponse
            import hashlib
            from django.utils.http import http_date
            
            # Ensure training_logs is properly formatted for JavaScript
            training_logs = context.get('training_logs', [])
            formatted_logs = []
            
            for i, log_line in enumerate(training_logs):
                if isinstance(log_line, str):
                    # Convert string logs to object format expected by JavaScript
                    formatted_logs.append({
                        'line_number': i + 1,
                        'content': log_line,
                        'timestamp': self._extract_timestamp_from_line(log_line),
                        'level': self._extract_log_level_from_line(log_line)
                    })
                elif isinstance(log_line, dict):
                    # Ensure all required properties exist
                    formatted_logs.append({
                        'line_number': log_line.get('line_number', i + 1),
                        'content': log_line.get('content', str(log_line)),
                        'timestamp': log_line.get('timestamp', None),
                        'level': log_line.get('level', 'INFO')
                    })
                else:
                    # Fallback for other types
                    formatted_logs.append({
                        'line_number': i + 1,
                        'content': str(log_line),
                        'timestamp': None,
                        'level': 'INFO'
                    })
            
            # Create response data
            response_data = {
                'status': self.object.status,
                'progress': {
                    'current_epoch': self.object.current_epoch or 0,
                    'total_epochs': self.object.total_epochs or 0,
                    'progress_percentage': context.get('progress_percentage', 0),
                    'train_loss': self.object.train_loss,
                    'val_loss': self.object.val_loss,
                    'train_dice': self.object.train_dice,
                    'val_dice': self.object.val_dice,
                    'best_val_dice': self.object.best_val_dice or 0.0,
                },
                'logs': formatted_logs,
                'parsed_logs': context.get('parsed_logs', {}),
                'log_stats': {
                    'total_lines': len(formatted_logs),
                    'epoch_logs': len(context.get('parsed_logs', {}).get('epoch_logs', [])),
                    'batch_logs': len(context.get('parsed_logs', {}).get('batch_logs', [])),
                    'model_logs': len(context.get('parsed_logs', {}).get('model_logs', [])),
                    'last_updated': formatted_logs[-1]['content'] if formatted_logs else ''
                }
            }
            
            # Generate ETag based on relevant data that changes
            etag_data = f"{self.object.status}:{self.object.current_epoch}:{self.object.updated_at.isoformat()}:{len(formatted_logs)}"
            etag = hashlib.md5(etag_data.encode()).hexdigest()
            
            # Check if client has current version
            client_etag = self.request.headers.get('If-None-Match')
            if client_etag and client_etag.strip('"') == etag:
                from django.http import HttpResponseNotModified
                return HttpResponseNotModified()
            
            # Create response with ETag
            response = JsonResponse(response_data)
            response['ETag'] = f'"{etag}"'
            response['Last-Modified'] = http_date(self.object.updated_at.timestamp())
            response['Cache-Control'] = 'no-cache, must-revalidate'
            
            return response
        return super().render_to_response(context, **response_kwargs)

    def _extract_timestamp_from_line(self, line):
        """Extract timestamp from log line"""
        import re
        timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
        match = re.search(timestamp_pattern, line)
        return match.group(1) if match else None

    def _extract_log_level_from_line(self, line):
        """Extract log level from log line"""
        if 'ERROR' in line or 'Exception' in line:
            return 'ERROR'
        elif 'WARNING' in line or 'WARN' in line:
            return 'WARNING'
        elif 'INFO' in line:
            return 'INFO'
        elif 'DEBUG' in line:
            return 'DEBUG'
        elif '[EPOCH]' in line:
            return 'EPOCH'
        elif '[TRAIN]' in line or '[VAL]' in line:
            return 'TRAINING'
        elif '[METRICS]' in line or '[STATS]' in line:
            return 'METRICS'
        else:
            return 'INFO'

    def _get_training_preview(self):
        """Get training preview images from MLflow artifacts"""
        preview_data = {
            'images': [],
            'error': None,
            'latest_epoch': 0
        }
        
        try:
            if not self.object.mlflow_run_id:
                preview_data['error'] = "No MLflow run ID available"
                return preview_data
            
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(self.object.mlflow_run_id)
            
            # Try multiple artifact and prediction directory paths to handle different MLflow configurations
            base_paths = [
                # PRIORITY: Check model-specific directory first (most accurate for organized structure)
                os.path.join(self.object.model_directory, 'predictions') if self.object.model_directory else None,
                os.path.join(self.object.model_directory, 'artifacts') if self.object.model_directory else None,
                # Fallback: Check organized model directory structure by run_id (legacy)
                os.path.join(settings.BASE_ORGANIZED_MODELS_DIR, run.info.run_id, 'predictions'),
                os.path.join(settings.BASE_ORGANIZED_MODELS_DIR, run.info.run_id, 'artifacts'),
                # Direct run ID path (current MLflow structure)
                os.path.join(settings.BASE_MLRUNS_DIR, run.info.run_id, 'artifacts'),
                # Legacy experiment-based paths  
                os.path.join(settings.BASE_MLRUNS_DIR, run.info.experiment_id, run.info.run_id, 'artifacts'),
                os.path.join(settings.BASE_MLRUNS_DIR, '0', run.info.run_id, 'artifacts'),
                os.path.join(settings.BASE_MLRUNS_DIR, '1', run.info.run_id, 'artifacts'),
                os.path.join(settings.BASE_MLRUNS_DIR, str(run.info.experiment_id), run.info.run_id, 'artifacts'),
                # Fallback to legacy mlruns structure
                os.path.join('mlruns', run.info.run_id, 'artifacts'),
                os.path.join('mlruns', run.info.experiment_id, run.info.run_id, 'artifacts'),
                os.path.join('mlruns', '0', run.info.run_id, 'artifacts'),
                os.path.join('mlruns', '1', run.info.run_id, 'artifacts'),
                os.path.join('mlruns', str(run.info.experiment_id), run.info.run_id, 'artifacts'),
            ]
            
            # Filter out None paths
            base_paths = [path for path in base_paths if path is not None]
            
            base_search_path = None
            for path in base_paths:
                if os.path.exists(path):
                    base_search_path = path
                    break
            
            if not base_search_path:
                preview_data['error'] = f"Search directory not found. Tried paths: {', '.join(base_paths)}"
                return preview_data
            
            # Find all prediction images using multiple search strategies for enhanced MLflow structure
            prediction_files = []
            
            def search_for_predictions(search_path, search_patterns):
                """Helper function to search for prediction files"""
                found_files = []
                seen_paths = set()  # Track already found files to avoid duplicates
                
                try:
                    if os.path.exists(search_path):
                        for item in os.listdir(search_path):
                            item_path = os.path.join(search_path, item)
                            
                            # Check if it's a file matching our patterns
                            if os.path.isfile(item_path) and item_path not in seen_paths:
                                for pattern_info in search_patterns:
                                    pattern = pattern_info['pattern']
                                    epoch_extract = pattern_info['epoch_extract']
                                    
                                    if pattern(item):
                                        try:
                                            epoch_num = epoch_extract(item)
                                            found_files.append({
                                                'filename': item,
                                                'epoch': epoch_num,
                                                'path': item_path,
                                                'relative_path': os.path.relpath(item_path, base_search_path)
                                            })
                                            seen_paths.add(item_path)
                                            break  # Found a match, don't check other patterns for this file
                                        except (ValueError, IndexError):
                                            continue
                            
                            # Check subdirectories for enhanced MLflow structure
                            elif os.path.isdir(item_path):
                                # Look in organized subdirectories like predictions/epoch_001/, artifacts/, or direct epoch directories
                                if item in ['predictions', 'visualizations', 'artifacts'] or item.startswith('epoch_'):
                                    # Handle both predictions/epoch_XXX/ and direct epoch_XXX/ structures
                                    def check_epoch_directory(epoch_dir_path, epoch_dir_name):
                                        """Check a directory that might contain epoch files"""
                                        if not os.path.isdir(epoch_dir_path):
                                            return
                                            
                                        for epoch_item in os.listdir(epoch_dir_path):
                                            epoch_item_path = os.path.join(epoch_dir_path, epoch_item)
                                            if os.path.isfile(epoch_item_path) and epoch_item_path not in seen_paths:
                                                for pattern_info in search_patterns:
                                                    pattern = pattern_info['pattern']
                                                    epoch_extract_alt = pattern_info.get('epoch_extract_alt')
                                                    
                                                    if pattern(epoch_item):
                                                        try:
                                                            # Try to extract epoch from directory name first
                                                            if epoch_extract_alt and any(c.isdigit() for c in epoch_dir_name):
                                                                epoch_num = epoch_extract_alt(epoch_dir_name)
                                                            else:
                                                                # Fallback to extracting from filename
                                                                epoch_extract = pattern_info['epoch_extract']
                                                                epoch_num = epoch_extract(epoch_item)
                                                            
                                                            found_files.append({
                                                                'filename': epoch_item,
                                                                'epoch': epoch_num,
                                                                'path': epoch_item_path,
                                                                'relative_path': os.path.relpath(epoch_item_path, base_search_path)
                                                            })
                                                            seen_paths.add(epoch_item_path)
                                                            break  # Found a match, don't check other patterns
                                                        except (ValueError, IndexError):
                                                            continue
                                    
                                    if item.startswith('epoch_'):
                                        # Direct epoch directory (like epoch_001/)
                                        check_epoch_directory(item_path, item)
                                    else:
                                        # Subdirectory that might contain epoch directories (like predictions/)
                                        for subitem in os.listdir(item_path):
                                            subitem_path = os.path.join(item_path, subitem)
                                            if os.path.isdir(subitem_path):
                                                check_epoch_directory(subitem_path, subitem)
                                            elif os.path.isfile(subitem_path) and subitem_path not in seen_paths:
                                                # Direct files in predictions/ directory
                                                for pattern_info in search_patterns:
                                                    pattern = pattern_info['pattern']
                                                    epoch_extract = pattern_info['epoch_extract']
                                                    
                                                    if pattern(subitem):
                                                        try:
                                                            epoch_num = epoch_extract(subitem)
                                                            found_files.append({
                                                                'filename': subitem,
                                                                'epoch': epoch_num,
                                                                'path': subitem_path,
                                                                'relative_path': os.path.relpath(subitem_path, base_search_path)
                                                            })
                                                            seen_paths.add(subitem_path)
                                                            break  # Found a match, don't check other patterns
                                                        except (ValueError, IndexError):
                                                            continue
                except OSError:
                    pass
                return found_files
            
            # Define search patterns for different MLflow structures (ordered by specificity)
            search_patterns = [
                # Most specific: predictions_epoch_N.png (numbered)
                {
                    'pattern': lambda f: f.startswith('predictions_epoch_') and f.endswith('.png') and any(c.isdigit() for c in f),
                    'epoch_extract': lambda f: int(''.join(filter(str.isdigit, f)))
                },
                # Directory-based pattern: any PNG file in epoch_N directories
                {
                    'pattern': lambda f: f.endswith('.png'),
                    'epoch_extract': lambda f: 0,  # Will use directory-based extraction
                    'epoch_extract_alt': lambda d: int(''.join(filter(str.isdigit, d))) if any(c.isdigit() for c in d) else 0
                }
            ]
            
            # Search in the main artifacts directory
            try:
                prediction_files.extend(search_for_predictions(base_search_path, search_patterns))
            except OSError as e:
                preview_data['error'] = f"Error reading artifacts directory: {str(e)}"
                return preview_data
            
            # Remove duplicates based on filename and epoch
            seen_files = set()
            unique_files = []
            for file_info in prediction_files:
                # Create a unique identifier using filename and epoch
                file_key = (file_info['filename'], file_info['epoch'])
                if file_key not in seen_files:
                    seen_files.add(file_key)
                    unique_files.append(file_info)
            
            # Sort by epoch number
            unique_files.sort(key=lambda x: x['epoch'])
            
            if unique_files:
                preview_data['images'] = unique_files
                preview_data['latest_epoch'] = unique_files[-1]['epoch']
            else:
                preview_data['error'] = f"No prediction images found. Searched in: {base_search_path} using multiple patterns for enhanced MLflow structure."
                
        except Exception as e:
            logging.error(f"Error getting training preview for model {self.object.id}: {e}")
            preview_data['error'] = f"Error loading training preview: {str(e)}"
        
        return preview_data

    def _get_training_details(self):
        """Extract comprehensive training configuration and details"""
        details = {
            'config': {},
            'hardware': {
                'device': 'N/A',
                'config_device': None,
                'pytorch_version': 'N/A',
                'cuda_available': False,
                'cuda_version': 'N/A',
            },
            'dataset': {},
            'augmentation': {},
            'optimizer': {},
            'architecture': {},
            'error': None
        }
        
        try:
            # Initialize hardware with current system info
            details['hardware'] = {
                'device': 'N/A',
                'config_device': None,
                'pytorch_version': torch.__version__ if torch else 'N/A',
                'cuda_available': torch.cuda.is_available() if torch else False,
                'cuda_version': torch.version.cuda if torch and torch.cuda.is_available() else 'N/A',
            }
            
            # Try to load from model directory if available
            if self.object.model_directory and os.path.exists(self.object.model_directory):
                config_path = os.path.join(self.object.model_directory, 'training_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    training_params = config_data.get('training_params', {})
                    
                    # Extract configuration details
                    details['config'] = {
                        'batch_size': training_params.get('batch_size', 'N/A'),
                        'epochs': training_params.get('epochs', 'N/A'),
                        'learning_rate': training_params.get('learning_rate', 'N/A'),
                        'validation_split': training_params.get('validation_split', 'N/A'),
                        'model_family': training_params.get('model_family', 'N/A'),
                        'model_type': training_params.get('model_type', 'N/A'),
                        'data_path': training_params.get('data_path', 'N/A'),
                        'crop_size': training_params.get('crop_size', 'N/A'),
                        'num_workers': training_params.get('num_workers', 'N/A'),
                    }
                    
                    # Extract hardware details - get actual runtime device from logs if available
                    runtime_device = self._extract_runtime_device_from_logs()
                    config_device = training_params.get('device', config_data.get('device', 'N/A'))
                    
                    details['hardware'].update({
                        'device': runtime_device if runtime_device else config_device,
                        'config_device': config_device if runtime_device and runtime_device != config_device else None,
                        'pytorch_version': config_data.get('pytorch_version', details['hardware']['pytorch_version']),
                    })
                    
                    # Extract augmentation details
                    details['augmentation'] = {
                        'random_flip': training_params.get('random_flip', False),
                        'random_rotate': training_params.get('random_rotate', False),
                        'random_scale': training_params.get('random_scale', False),
                        'random_intensity': training_params.get('random_intensity', False),
                    }
                    
                # Try to load model summary for architecture details
                summary_path = os.path.join(self.object.model_directory, 'model_summary.txt')
                if os.path.exists(summary_path):
                    with open(summary_path, 'r') as f:
                        summary_content = f.read()
                    
                    # Parse parameter count from summary
                    total_params = 0
                    param_lines = [line for line in summary_content.split('\n') if 'params:' in line]
                    for line in param_lines:
                        try:
                            param_count = int(line.split('params: ')[-1])
                            total_params += param_count
                        except (ValueError, IndexError):
                            continue
                    
                    details['architecture'] = {
                        'total_parameters': total_params,
                        'summary_available': True,
                        'summary_content': summary_content[:1000] + '...' if len(summary_content) > 1000 else summary_content
                    }
            
            # Fallback to MLflow data if model directory not available
            elif self.object.mlflow_run_id:
                try:
                    client = mlflow.tracking.MlflowClient()
                    run = client.get_run(self.object.mlflow_run_id)
                    
                    # Extract from MLflow parameters
                    params = run.data.params
                    details['config'] = {
                        'batch_size': params.get('batch_size', 'N/A'),
                        'epochs': params.get('epochs', 'N/A'),
                        'learning_rate': params.get('learning_rate', 'N/A'),
                        'validation_split': params.get('validation_split', 'N/A'),
                        'model_type': params.get('model_type', 'N/A'),
                        'data_path': params.get('data_path', 'N/A'),
                    }
                    
                except Exception as e:
                    details['error'] = f"Could not load MLflow data: {e}"
            
            # Add dataset information if available
            if self.object.training_data_info:
                details['dataset'] = self.object.training_data_info
                
                # Also extract augmentation info from training_data_info if not already set
                if not details['augmentation'] or not any(details['augmentation'].values()):
                    training_info = self.object.training_data_info
                    details['augmentation'] = {
                        'random_flip': training_info.get('use_random_flip', False),
                        'random_rotate': training_info.get('use_random_rotate', False),
                        'random_scale': training_info.get('use_random_scale', False),
                        'random_intensity': training_info.get('use_random_intensity', False),
                    }
                
                # Extract additional config info from training_data_info if not already set
                if details['config'].get('crop_size') == 'N/A':
                    details['config']['crop_size'] = training_info.get('crop_size', 'N/A')
                if details['config'].get('num_workers') == 'N/A':
                    details['config']['num_workers'] = training_info.get('num_workers', 'N/A')
            
            # Ensure all sections have default values even if not loaded from config
            if not details['config']:
                details['config'] = {
                    'batch_size': 'N/A',
                    'epochs': 'N/A',
                    'learning_rate': 'N/A',
                    'validation_split': 'N/A',
                    'model_type': 'N/A',
                    'data_path': 'N/A',
                    'crop_size': 'N/A',
                    'num_workers': 'N/A',
                    'model_family': 'N/A',
                }
            else:
                # Ensure all required keys exist even if some are missing
                required_keys = ['batch_size', 'epochs', 'learning_rate', 'validation_split', 
                               'model_type', 'data_path', 'crop_size', 'num_workers', 'model_family']
                for key in required_keys:
                    if key not in details['config']:
                        details['config'][key] = 'N/A'
            
            if not details['augmentation']:
                details['augmentation'] = {
                    'random_flip': False,
                    'random_rotate': False,
                    'random_scale': False,
                    'random_intensity': False,
                }
                
        except Exception as e:
            details['error'] = f"Could not load training details: {e}"
            
        return details
    
    def _get_architecture_details(self):
        """Get detailed model architecture information"""
        architecture = {
            'name': 'Unknown',
            'type': 'Unknown',
            'framework': 'Unknown',
            'details': {},
            'error': None
        }
        
        try:
            # Use model architecture info if available
            if self.object.model_architecture_info:
                architecture.update(self.object.model_architecture_info)
            
            # Try to determine architecture from model type or config data
            model_type = None
            if hasattr(self.object, 'model_type'):
                model_type = self.object.model_type
            elif self.object.training_data_info and 'model_type' in self.object.training_data_info:
                model_type = self.object.training_data_info.get('model_type')
            elif self.object.model_directory and os.path.exists(self.object.model_directory):
                # Try to load from config file
                config_path = os.path.join(self.object.model_directory, 'training_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    model_type = config_data.get('training_params', {}).get('model_type')
                
            if model_type == 'unet':
                architecture.update({
                    'name': 'MonaiUNet',
                    'type': 'U-Net',
                    'framework': 'MONAI/PyTorch',
                    'details': {
                        'architecture_family': 'Encoder-Decoder',
                        'primary_use': 'Medical Image Segmentation',
                        'skip_connections': True,
                        'activation': 'ReLU',
                        'normalization': 'Batch Normalization'
                    }
                })
            elif model_type == 'unet-old':
                architecture.update({
                    'name': 'PyTorch UNet',
                    'type': 'U-Net (Legacy)',
                    'framework': 'PyTorch',
                    'details': {
                        'architecture_family': 'Encoder-Decoder',
                        'primary_use': 'Image Segmentation',
                        'skip_connections': True,
                    }
                })
                    
        except Exception as e:
            architecture['error'] = f"Could not determine architecture: {e}"
            
        return architecture

    def _extract_runtime_device_from_logs(self):
        """Extract the actual runtime device from training logs"""
        try:
            # First try to get logs from the current model training
            if hasattr(self, 'object') and self.object:
                training_logs = self._get_training_logs()
                if training_logs:
                    # Look for device information in logs
                    for log_line in training_logs:
                        line_content = log_line if isinstance(log_line, str) else str(log_line)
                        
                        # Look for the specific log pattern from train.py
                        if '[TRAINING] Using device:' in line_content:
                            # Extract device from log line like: "[TRAINING] Using device: cuda"
                            import re
                            device_match = re.search(r'\[TRAINING\] Using device:\s*(\w+)', line_content)
                            if device_match:
                                return device_match.group(1).lower()
                        
                        # Alternative patterns to catch device information
                        elif 'Device:' in line_content and any(dev in line_content.lower() for dev in ['cuda', 'cpu', 'mps']):
                            # Extract from patterns like "Device: cuda" or "Device: cpu"
                            device_match = re.search(r'Device:\s*(\w+)', line_content, re.IGNORECASE)
                            if device_match:
                                return device_match.group(1).lower()
                        
                        # Also check for CUDA availability logs
                        elif 'cuda.is_available()' in line_content.lower():
                            if 'true' in line_content.lower() or 'available' in line_content.lower():
                                return 'cuda'
                            elif 'false' in line_content.lower() or 'not available' in line_content.lower():
                                return 'cpu'
                            
            return None
        except Exception as e:
            # Log the error but don't fail the whole view
            import logging
            logging.warning(f"Could not extract runtime device from logs: {e}")
            return None

    def _get_training_logs(self):
        """Get training logs for this model"""
        try:
            # Try to find logs in the model-specific location first
            if self.object.model_directory and os.path.exists(self.object.model_directory):
                log_path = os.path.join(self.object.model_directory, 'logs', 'training.log')
                if os.path.exists(log_path):
                    with open(log_path, 'r') as f:
                        return f.read().splitlines()
            
            # Fallback to global log location
            log_path = os.path.join('data', 'logs', 'training.log')
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    return f.read().splitlines()
            
            return []
        except Exception as e:
            import logging
            logging.warning(f"Could not load training logs: {e}")
            return []


class ModelPredictionListView(LoginRequiredMixin, ListView):
    model = Prediction
    template_name = 'ml_manager/model_predictions.html'
    context_object_name = 'predictions'
    paginate_by = 20

    def get_queryset(self):
        # Get the model based on the pk from URL
        self.ml_model = get_object_or_404(MLModel, pk=self.kwargs['pk'])
        # Filter predictions by the model
        return Prediction.objects.filter(model=self.ml_model).order_by('-created_at')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Add the model to context for the template
        context['model'] = self.ml_model
        return context


class StartTrainingView(LoginRequiredMixin, FormView):
    form_class = TrainingForm
    template_name = 'ml_manager/start_training.html'
    success_url = reverse_lazy('ml_manager:model-list')

    def get_initial(self):
        initial = super().get_initial()
        rerun_model_id = self.request.GET.get('rerun')
        if rerun_model_id:
            try:
                model = get_object_or_404(MLModel, pk=rerun_model_id)
                if model.training_data_info:
                    # Clean up name to avoid multiple (Rerun) tags
                    base_name = model.name
                    if base_name.endswith('(Rerun)'):
                        base_name = base_name[:-7].rstrip()
                    initial.update(model.training_data_info)
                    initial['name'] = f"{base_name} (Rerun)"
            except MLModel.DoesNotExist:
                pass
        return initial

    def form_valid(self, form):
        logger = logging.getLogger(__name__)
        logger.info("StartTrainingView.form_valid() called")
        try:
            form_data = form.cleaned_data
            logger.info(f"Form data: {form_data}")
            from .utils.mlflow_utils import setup_mlflow
            setup_mlflow()
            
            # Check for any active MLflow run and end it before starting a new one
            active_run = mlflow.active_run()
            if active_run:
                logger.warning(f"Found active MLflow run {active_run.info.run_id}, ending it to start new training")
                try:
                    mlflow.end_run()
                except Exception as e:
                    logger.warning(f"Failed to end active MLflow run {active_run.info.run_id}: {e}")
                    # Force clear the active run by setting environment variable
                    if 'MLFLOW_TRACKING_RUN_ID' in os.environ:
                        del os.environ['MLFLOW_TRACKING_RUN_ID']
                    # Try force end with KILLED status
                    try:
                        mlflow.end_run(status='KILLED')
                    except:
                        logger.warning("Could not force kill MLflow run, continuing anyway")
            
            # Create MLflow run first, but DO NOT call mlflow.end_run() here!
            mlflow_run = mlflow.start_run()
            mlflow_run_id = mlflow_run.info.run_id
            logger.info(f"Started new MLflow run: {mlflow_run_id}")
            # Do not end the run here; let train.py manage run lifecycle
            # Remove any accidental end_run patching
            ml_model = MLModel.objects.create(
                name=form_data['name'],
                description=form_data.get('description', ''),
                status='pending',  # Will be updated to 'training' immediately by training manager
                current_epoch=0,
                total_epochs=form_data['epochs'],
                train_loss=0.0,
                val_loss=0.0,
                train_dice=0.0,
                val_dice=0.0,
                best_val_dice=0.0,
                mlflow_run_id=mlflow_run_id,
                training_data_info={
                    'model_type': form_data['model_type'],
                    'data_path': form_data['data_path'],
                    'dataset_type': form_data.get('dataset_type', 'auto'),
                    'batch_size': form_data['batch_size'],
                    'learning_rate': form_data['learning_rate'],
                    'optimizer': form_data.get('optimizer', 'adam'),
                    'validation_split': form_data['validation_split'],
                    'resolution': form_data['resolution'],
                    'device': form_data['device'],
                    'use_random_flip': form_data['use_random_flip'],
                    'use_random_rotate': form_data['use_random_rotate'],
                    'use_random_scale': form_data['use_random_scale'],
                    'use_random_intensity': form_data['use_random_intensity'],
                    'crop_size': form_data['crop_size'],
                    'num_workers': form_data['num_workers'],
                    # Dataset info placeholders (will be updated during training)
                    'training_samples': 0,
                    'validation_samples': 0,
                    'total_samples': 0,
                    # Learning rate scheduler parameters
                    'lr_scheduler': form_data.get('lr_scheduler', 'none'),
                    'lr_patience': form_data.get('lr_patience', 5),
                    'lr_factor': form_data.get('lr_factor', 0.5),
                    'lr_step_size': form_data.get('lr_step_size', 10),
                    'lr_gamma': form_data.get('lr_gamma', 0.1),
                    'min_lr': form_data.get('min_lr', 1e-7),
                    # Early stopping parameters
                    'use_early_stopping': form_data.get('use_early_stopping', False),
                    'early_stopping_patience': form_data.get('early_stopping_patience', 10),
                    'early_stopping_min_epochs': form_data.get('early_stopping_min_epochs', 20),
                    'early_stopping_min_delta': form_data.get('early_stopping_min_delta', 1e-4),
                    'early_stopping_metric': form_data.get('early_stopping_metric', 'val_dice'),
                    # Enhanced Training Features
                    'loss_function': form_data.get('loss_function', 'combined'),
                    'dice_weight': form_data.get('dice_weight', 0.7),
                    'bce_weight': form_data.get('bce_weight', 0.3),
                    'use_loss_scheduling': form_data.get('use_loss_scheduling', False),
                    'loss_scheduler_type': form_data.get('loss_scheduler_type', 'adaptive'),
                    'checkpoint_strategy': form_data.get('checkpoint_strategy', 'best'),
                    'max_checkpoints': form_data.get('max_checkpoints', 5),
                    'monitor_metric': form_data.get('monitor_metric', 'val_dice'),
                    'use_enhanced_training': form_data.get('use_enhanced_training', True),
                    'use_mixed_precision': form_data.get('use_mixed_precision', False),
                },
                model_type=form_data['model_type']
            )
            logger.info(f"Created MLModel instance with ID: {ml_model.id}, model_type: {ml_model.model_type}, mlflow_run_id: {mlflow_run_id}")
            
            # Debug form data for crop_size
            logger.info(f"DEBUG: crop_size in form_data = {form_data.get('crop_size')}, type = {type(form_data.get('crop_size'))}")
            logger.info(f"DEBUG: All form keys: {list(form_data.keys())}")
            logger.info(f"DEBUG: Form data crop_size related: {[k for k in form_data.keys() if 'crop' in k.lower()]}")
            
            # Prepare training configuration for direct training
            training_config = {
                'model_type': form_data['model_type'],
                'data_path': form_data['data_path'],
                'dataset_type': form_data['dataset_type'],
                'batch_size': form_data['batch_size'],
                'epochs': form_data['epochs'],
                'learning_rate': form_data['learning_rate'],
                'optimizer': form_data.get('optimizer', 'adam'),
                'validation_split': form_data['validation_split'],
                'resolution': form_data['resolution'],
                'device': form_data['device'],
                'crop_size': form_data.get('crop_size', 512),
                'num_workers': form_data['num_workers'],
                # Learning rate scheduler parameters
                'lr_scheduler': form_data.get('lr_scheduler', 'none'),
                'lr_patience': form_data.get('lr_patience') or 5,
                'lr_factor': form_data.get('lr_factor') or 0.5,
                'lr_step_size': form_data.get('lr_step_size') or 10,
                'lr_gamma': form_data.get('lr_gamma') or 0.1,
                'min_lr': form_data.get('min_lr') or 1e-7,
                # Early stopping parameters
                'early_stopping_patience': form_data.get('early_stopping_patience') or 10,
                'early_stopping_min_epochs': form_data.get('early_stopping_min_epochs') or 20,
                'early_stopping_min_delta': form_data.get('early_stopping_min_delta') or 1e-4,
                'early_stopping_metric': form_data.get('early_stopping_metric') or 'val_dice',
                # Enhanced Training Parameters
                'loss_function': form_data.get('loss_function', 'mixed'),
                'checkpoint_strategy': form_data.get('checkpoint_strategy', 'best'),
                'max_checkpoints': form_data.get('max_checkpoints') or 5,
                'monitor_metric': form_data.get('monitor_metric', 'val_dice'),
                'loss_scheduler_type': form_data.get('loss_scheduler_type', 'adaptive'),
                # Loss function weights
                'dice_weight': form_data.get('dice_weight') or 0.7,
                'bce_weight': form_data.get('bce_weight') or 0.3,
                # Training flags
                'use_early_stopping': form_data.get('use_early_stopping', False),
                'use_enhanced_training': form_data.get('use_enhanced_training', True),
                'use_loss_scheduling': form_data.get('use_loss_scheduling', False),
                'use_mixed_precision': form_data.get('use_mixed_precision', False),
                # Augmentation flags
                'use_random_flip': form_data.get('use_random_flip', False),
                'use_random_rotate': form_data.get('use_random_rotate', False),
                'use_random_scale': form_data.get('use_random_scale', False),
                'use_random_intensity': form_data.get('use_random_intensity', False),
                'mlflow_run_id': mlflow_run_id
            }
            
            # Start training using subprocess instead of direct_training manager
            logger.info(f"Starting training subprocess for model {ml_model.id}")
            
            # Build command arguments
            training_args = [
                sys.executable, 'ml/training/train.py',
                '--mode', 'train',
                '--model-id', str(ml_model.id),
                '--mlflow-run-id', str(mlflow_run_id),
                '--model-family', form_data.get('model_family', 'UNet-Coronary'),
                '--model-type', form_data['model_type'],
                '--data-path', form_data['data_path'],
                '--dataset-type', form_data['dataset_type'],
                '--batch-size', str(form_data['batch_size']),
                '--epochs', str(form_data['epochs']),
                '--learning-rate', str(form_data['learning_rate']),
                '--optimizer', form_data['optimizer'],
                '--validation-split', str(form_data['validation_split']),
                '--crop-size', str(form_data['crop_size']),
                '--threshold', str(form_data.get('threshold', 0.5)),
                '--loss-function', form_data.get('loss_function', 'combined'),
                '--dice-weight', str(form_data.get('dice_weight', 0.7)),
                '--bce-weight', str(form_data.get('bce_weight', 0.3)),
                '--num-workers', str(form_data['num_workers']),
                '--lr-scheduler', form_data['lr_scheduler'],
                '--lr-patience', str(form_data.get('lr_patience', 5)),
            ]
            
            # Only enable mixed precision on GPU devices
            device = form_data.get('device', 'auto')
            if form_data.get('use_mixed_precision', False) and device != 'cpu':
                training_args.extend(['--use-mixed-precision', 'True'])
            elif form_data.get('use_mixed_precision', False) and device == 'cpu':
                logger.info("Mixed precision disabled for CPU device")
            
            # Add enhanced training flags
            if form_data.get('use_enhanced_training', True):
                training_args.append('--use-enhanced-training')
            
            # Add augmentation flags
            if form_data.get('use_random_flip', False):
                training_args.append('--random-flip')
            if form_data.get('use_random_rotate', False):
                training_args.append('--random-rotate')
            if form_data.get('use_random_scale', False):
                training_args.append('--random-scale')
            if form_data.get('use_random_intensity', False):
                training_args.append('--random-intensity')
            
            # Start training process in background
            import subprocess
            try:
                # Create log directory for this training session
                log_dir = Path(settings.MEDIA_ROOT) / 'logs' / f'model_{ml_model.id}'
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / f'training_{ml_model.id}_{int(time.time())}.log'
                
                # Start process and redirect output to log file
                with open(log_file, 'w') as f:
                    process = subprocess.Popen(
                        training_args,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        cwd=Path(__file__).parent.parent.parent.parent,  # Project root
                    )
                
                # Update model with process info
                ml_model.status = 'loading'  # Will be updated to 'training' by callback
                ml_model.process_id = process.pid
                ml_model.mlflow_run_id = mlflow_run_id
                ml_model.training_logs = f"Training started. Logs: {log_file}"
                ml_model.save()
                
                logger.info(f"Training process started with PID {process.pid}, logs: {log_file}")
                
            except Exception as subprocess_error:
                logger.error(f"Failed to start training subprocess: {subprocess_error}")
                raise subprocess_error
            
            logger.info(f"Direct training started for model {ml_model.id}")
            messages.success(self.request, f"Training for '{ml_model.name}' (ID: {ml_model.id}) started successfully with model type '{form_data['model_type']}'.")
            return super().form_valid(form)
        except Exception as e:
            logger.error(f"Error in StartTrainingView.form_valid: {e}", exc_info=True)
            messages.error(self.request, f"Failed to start training: {e}")
            if 'ml_model' in locals() and ml_model:
                ml_model.status = 'failed'
                ml_model.training_logs = f"Failed to start training process: {e}"
                ml_model.save()
            return self.form_invalid(form)

    def form_invalid(self, form):
        """Handle form validation errors"""
        logger = logging.getLogger(__name__)
        logger.error("StartTrainingView.form_invalid() called")
        logger.error(f"Form errors: {form.errors}")
        logger.error(f"Non-field errors: {form.non_field_errors()}")
        
        # Add a message to help with debugging
        messages.error(self.request, "Form validation failed. Please check all required fields.")
        
        return super().form_invalid(form)

    def get_initial(self):
        """Pre-populate form data from rerun parameter or template"""
        initial = super().get_initial()
        
        # Handle rerun parameter - pre-populate from existing model
        rerun_model_id = self.request.GET.get('rerun')
        if rerun_model_id:
            try:
                model = get_object_or_404(MLModel, pk=rerun_model_id)
                if model.training_data_info:
                    # Clean up name to avoid multiple (Rerun) tags
                    base_name = model.name
                    if base_name.endswith('(Rerun)'):
                        base_name = base_name[:-7].rstrip()
                    initial.update({
                        'name': f"{base_name} (Rerun)",
                        'description': f"Rerun of model: {model.name}",
                        'model_type': model.training_data_info.get('model_type', 'unet'),
                        'data_path': model.training_data_info.get('data_path', ''),
                        'batch_size': model.training_data_info.get('batch_size', 32),
                        'epochs': model.total_epochs or model.training_data_info.get('epochs', 100),
                        'learning_rate': model.training_data_info.get('learning_rate', 0.001),
                        'optimizer': model.training_data_info.get('optimizer', 'adam'),
                        'validation_split': model.training_data_info.get('validation_split', 0.2),
                        'resolution': model.training_data_info.get('resolution', '256'),
                        'device': model.training_data_info.get('device', 'auto'),
                        'use_random_flip': model.training_data_info.get('use_random_flip', True),
                        'use_random_rotate': model.training_data_info.get('use_random_rotate', True),
                        'use_random_scale': model.training_data_info.get('use_random_scale', True),
                        'use_random_intensity': model.training_data_info.get('use_random_intensity', True),
                        'crop_size': model.training_data_info.get('crop_size', 128),
                        'num_workers': model.training_data_info.get('num_workers', 4),
                        # Learning Rate Scheduler Configuration
                        'lr_scheduler': model.training_data_info.get('lr_scheduler', 'plateau'),
                        'lr_patience': model.training_data_info.get('lr_patience', 5),
                        'lr_factor': model.training_data_info.get('lr_factor', 0.5),
                        'lr_step_size': model.training_data_info.get('lr_step_size', 10),
                        'lr_gamma': model.training_data_info.get('lr_gamma', 0.1),
                        'min_lr': model.training_data_info.get('min_lr', 1e-7),
                        # Early stopping parameters  
                        'use_early_stopping': model.training_data_info.get('use_early_stopping', False),
                        'early_stopping_patience': model.training_data_info.get('early_stopping_patience', 10),
                        'early_stopping_min_epochs': model.training_data_info.get('early_stopping_min_epochs', 20),
                        'early_stopping_min_delta': model.training_data_info.get('early_stopping_min_delta', 1e-4),
                        'early_stopping_metric': model.training_data_info.get('early_stopping_metric', 'val_dice'),
                    })
            except MLModel.DoesNotExist:
                pass  # Continue with default initial values
        
        return initial

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Add available templates for template selection
        try:
            from .models import TrainingTemplate
            context['templates'] = TrainingTemplate.objects.all()
        except:
            context['templates'] = []
            
        # Always provide rerun_model variable (None if not applicable)
        context['rerun_model'] = None
        rerun_model_id = self.request.GET.get('rerun')
        if rerun_model_id:
            try:
                context['rerun_model'] = get_object_or_404(MLModel, pk=rerun_model_id)
            except MLModel.DoesNotExist:
                pass
            
        return context


# Additional view functions and classes

@login_required
@require_POST
def stop_training(request, model_id):
    """Stop a running training job"""
    import signal
    try:
        model = get_object_or_404(MLModel, id=model_id)
        
        if model.status not in ['training', 'loading']:
            return JsonResponse({
                'status': 'error',
                'message': 'Model is not currently training'
            })
        
        # Method 1: Set stop_requested flag for graceful shutdown
        model.stop_requested = True
        model.save()
        
        # Method 2: Try to terminate the process if process_id is available
        if hasattr(model, 'process_id') and model.process_id:
            try:
                import psutil
                # Check if process exists and terminate it
                if psutil.pid_exists(model.process_id):
                    process = psutil.Process(model.process_id)
                    process.terminate()  # Send SIGTERM for graceful shutdown
                    
                    # Wait a bit for graceful shutdown, then force kill if needed
                    try:
                        process.wait(timeout=10)  # Wait up to 10 seconds
                    except psutil.TimeoutExpired:
                        process.kill()  # Force kill if graceful shutdown failed
                    
                    logging.info(f"Terminated training process {model.process_id} for model {model_id}")
                else:
                    logging.warning(f"Process {model.process_id} not found for model {model_id}")
            except ImportError:
                # Fallback to os.kill if psutil not available
                try:
                    os.kill(model.process_id, signal.SIGTERM)
                    logging.info(f"Sent SIGTERM to process {model.process_id} for model {model_id}")
                except ProcessLookupError:
                    logging.warning(f"Process {model.process_id} not found for model {model_id}")
            except Exception as proc_error:
                logging.warning(f"Error terminating process {model.process_id}: {proc_error}")
        
        # Update model status
        model.status = 'stopped'
        model.save()
        
        return JsonResponse({
            'status': 'success',
            'message': f'Training stopped for model "{model.name}".',
            'model_status': 'stopped'
        })
        
    except Exception as e:
        logging.error(f"Error stopping training for model {model_id}: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })


class ModelDeleteView(LoginRequiredMixin, DeleteView):
    model = MLModel
    template_name = 'ml_manager/model_confirm_delete.html'
    success_url = reverse_lazy('ml_manager:model-list')
    
    def form_valid(self, form):
        """
        Custom deletion logic moved from delete() to form_valid()
        as recommended by Django DeleteView warning.
        """
        try:
            self.object = self.get_object()
            
            # Clean up associated files if needed
            if self.object.model_directory and os.path.exists(self.object.model_directory):
                try:
                    shutil.rmtree(self.object.model_directory)
                except Exception as e:
                    logging.warning(f"Could not delete model directory: {e}")
            
            messages.success(self.request, f'Model "{self.object.name}" has been deleted.')
            
            # Call parent form_valid which will delete the object
            response = super().form_valid(form)
            
            # Return JSON response for AJAX requests
            if self.request.headers.get('Content-Type') == 'application/json' or self.request.META.get('HTTP_ACCEPT') == 'application/json':
                return JsonResponse({'status': 'success', 'redirect': str(self.success_url)})
            
            return response
            
        except Exception as e:
            logging.error(f"Error deleting model: {e}")
            messages.error(self.request, f'Error deleting model: {e}')
            
            if self.request.headers.get('Content-Type') == 'application/json' or self.request.META.get('HTTP_ACCEPT') == 'application/json':
                return JsonResponse({'status': 'error', 'message': str(e)})
            
            return self.form_invalid(form)


@login_required
@require_POST
def batch_delete_models(request):
    """Delete multiple models at once"""
    try:
        # Handle FormData from JavaScript (not JSON)
        model_ids = request.POST.getlist('model_ids')
        
        if not model_ids:
            return JsonResponse({
                'status': 'error',
                'message': 'No models selected for deletion'
            })
        
        deleted_count = 0
        for model_id in model_ids:
            try:
                model = MLModel.objects.get(id=model_id)
                
                # Clean up model directory
                if model.model_directory and os.path.exists(model.model_directory):
                    try:
                        shutil.rmtree(model.model_directory)
                    except Exception as e:
                        logging.warning(f"Could not delete directory for model {model_id}: {e}")
                
                model.delete()
                deleted_count += 1
                
            except MLModel.DoesNotExist:
                logging.warning(f"Model {model_id} not found")
                continue
            except Exception as e:
                logging.error(f"Error deleting model {model_id}: {e}")
                continue
        
        return JsonResponse({
            'status': 'success',
            'message': f'Successfully deleted {deleted_count} model(s)',
            'deleted_count': deleted_count
        })
        
    except Exception as e:
        logging.error(f"Error in batch delete: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })


class ModelInferenceView(LoginRequiredMixin, FormView):
    form_class = EnhancedInferenceForm
    template_name = 'ml_manager/model_inference.html'
    
    def get_success_url(self):
        return reverse_lazy('ml_manager:model-inference', kwargs={'pk': self.kwargs['pk']})
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        self.model = get_object_or_404(MLModel, pk=self.kwargs['pk'])
        context['model'] = self.model
        
        # Get all completed models for the dropdown
        context['registered_models'] = MLModel.objects.filter(status='completed').order_by('-created_at')
        
        # Get latest prediction for this model to show as example
        latest_prediction = Prediction.objects.filter(model=self.model).order_by('-created_at').first()
        context['latest_prediction'] = latest_prediction
        
        return context
    
    def form_valid(self, form):
        logger = logging.getLogger(__name__)
        try:
            start_time = time.time()
            # Get the selected model from the form instead of URL parameter
            selected_model = form.cleaned_data['model_id']
            
            # Get the uploaded image
            image = form.cleaned_data['image']
            
            # Get the chosen crop size
            crop_size = form.cleaned_data['crop_size']
            
            # Create temporary directories for inference
            temp_input_dir = tempfile.mkdtemp(prefix='inference_input_')
            temp_output_dir = tempfile.mkdtemp(prefix='inference_output_')
            
            try:
                # Save uploaded image to temporary input directory
                input_path = os.path.join(temp_input_dir, f'input_{image.name}')
                with open(input_path, 'wb+') as destination:
                    for chunk in image.chunks():
                        destination.write(chunk)
                
                # Determine model weights path
                weights_path = None
                def abs_path(p):
                    if not p:
                        return None
                    return p if os.path.isabs(p) else os.path.join('/app', p)

                # Log model details for debugging
                logging.info(f"=== DEBUGGING MODEL WEIGHTS PATH ===")
                logging.info(f"Selected model: {selected_model.name} (ID: {selected_model.id})")
                logging.info(f"Model weights path: {selected_model.model_weights_path}")
                logging.info(f"Model directory: {selected_model.model_directory}")
                logging.info(f"MLflow run ID: {selected_model.mlflow_run_id}")

                # 1. Direct model_weights_path
                if selected_model.model_weights_path:
                    abs_weights_path = abs_path(selected_model.model_weights_path)
                    logging.info(f"Checking direct weights path: {abs_weights_path}")
                    logging.info(f"Path exists: {os.path.exists(abs_weights_path) if abs_weights_path else False}")
                    if abs_weights_path and os.path.exists(abs_weights_path):
                        weights_path = abs_weights_path
                        logging.info(f"Found weights at direct path: {weights_path}")

                # 2. model_directory/weights/model.pth
                if not weights_path and selected_model.model_directory:
                    model_dir_abs = abs_path(selected_model.model_directory)
                    logging.info(f"Checking model directory: {model_dir_abs}")
                    logging.info(f"Directory exists: {os.path.exists(model_dir_abs) if model_dir_abs else False}")
                    if model_dir_abs and os.path.exists(model_dir_abs):
                        weights_in_dir = os.path.join(model_dir_abs, "weights", "model.pth")
                        logging.info(f"Checking weights in directory: {weights_in_dir}")
                        logging.info(f"Weights file exists: {os.path.exists(weights_in_dir)}")
                        if os.path.exists(weights_in_dir):
                            weights_path = weights_in_dir
                            logging.info(f"Found weights in model directory: {weights_path}")

                # 3. mlruns/{mlflow_run_id}/artifacts/model.pth
                if not weights_path and selected_model.mlflow_run_id:
                    mlflow_artifacts_path = os.path.join(str(settings.BASE_MLRUNS_DIR), selected_model.mlflow_run_id, 'artifacts', 'model.pth')
                    logging.info(f"Checking MLflow artifacts path: {mlflow_artifacts_path}")
                    logging.info(f"MLflow path exists: {os.path.exists(mlflow_artifacts_path)}")
                    if os.path.exists(mlflow_artifacts_path):
                        weights_path = mlflow_artifacts_path
                        logging.info(f"Found weights at MLflow path: {weights_path}")
                    else:
                        # 4. mlruns/{mlflow_run_id}/artifacts/model/data/model.pth
                        mlflow_alt_path = os.path.join(str(settings.BASE_MLRUNS_DIR), selected_model.mlflow_run_id, 'artifacts', 'model', 'data', 'model.pth')
                        logging.info(f"Checking alternative MLflow path: {mlflow_alt_path}")
                        logging.info(f"Alternative MLflow path exists: {os.path.exists(mlflow_alt_path)}")
                        if os.path.exists(mlflow_alt_path):
                            weights_path = mlflow_alt_path
                            logging.info(f"Found weights at alternative MLflow path: {weights_path}")

                # Log all available files in mlruns for debugging
                if not weights_path and selected_model.mlflow_run_id:
                    mlruns_dir = os.path.join(str(settings.BASE_MLRUNS_DIR), selected_model.mlflow_run_id)
                    logging.info(f"Listing contents of {mlruns_dir}:")
                    try:
                        if os.path.exists(mlruns_dir):
                            for root, dirs, files in os.walk(mlruns_dir):
                                for file in files:
                                    full_path = os.path.join(root, file)
                                    logging.info(f"  Found file: {full_path}")
                        else:
                            logging.info(f"  Directory does not exist: {mlruns_dir}")
                    except Exception as e:
                        logging.error(f"Error listing mlruns directory: {e}")

                logging.info(f"Final weights path: {weights_path}")
                logging.info(f"=== END DEBUGGING ===")

                if not weights_path or not os.path.exists(weights_path):
                    error_msg = f"Model weights not found for model {selected_model.name}. Checked paths:\n"
                    error_msg += f"1. Direct path: {selected_model.model_weights_path}\n"
                    error_msg += f"2. Model directory: {selected_model.model_directory}\n"
                    error_msg += f"3. MLflow run: {selected_model.mlflow_run_id}\n"
                    raise ValueError(error_msg)
                
                # Get model configuration
                model_type = 'unet'  # Default
                if selected_model.training_data_info:
                    model_type = selected_model.training_data_info.get('model_type', 'unet')
                
                # Run inference
                run_inference(
                    model_path=weights_path,
                    input_path=input_path,
                    output_dir=temp_output_dir,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    model_type=model_type,
                    crop_size=int(crop_size) if crop_size != 'original' else 256
                )
                
                # Find the output images - specifically look for prediction-only file
                output_files = [f for f in os.listdir(temp_output_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
                
                if not output_files:
                    raise ValueError("No output image was generated")
                
                # Look specifically for the prediction-only file
                pred_only_file = None
                for f in output_files:
                    if f.startswith('pred_only_'):
                        pred_only_file = f
                        break
                
                if not pred_only_file:
                    # Fallback to first output if pred_only not found
                    pred_only_file = output_files[0]
                    logger.warning(f"pred_only file not found, using fallback: {pred_only_file}")
                
                output_path = os.path.join(temp_output_dir, pred_only_file)
                logger.info(f"Using prediction file: {pred_only_file}")
                logger.info(f"Full output path: {output_path}")
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Create Prediction object
                # prediction = Prediction(
                #     model=selected_model,
                #     input_data={'filename': image.name, 'size': image.size},
                #     output_data={'processing_time': processing_time},
                #     processing_time=processing_time
                # )
                prediction = Prediction.objects.create(
                    model=selected_model,
                    input_data={'filename': image.name, 'size': image.size},
                    output_data={'processing_time': processing_time},
                    input_image=f"predictions/input_{image.name}",
                    output_image=f"predictions/pred_only_{image.name}",
                    processing_time=processing_time
                )
                
                # Copy the transformed input and prediction images to media directory
                media_predictions_dir = os.path.join(settings.MEDIA_ROOT, 'predictions')
                os.makedirs(media_predictions_dir, exist_ok=True)
                
                # Copy transformed input image (consistent orientation)
                input_result_path = os.path.join(temp_output_dir, f"input_{image.name}")
                if os.path.exists(input_result_path):
                    shutil.copy2(input_result_path, os.path.join(media_predictions_dir, f"input_{image.name}"))
                
                # Copy prediction only image
                pred_result_path = os.path.join(temp_output_dir, f"pred_only_{image.name}")
                if os.path.exists(pred_result_path):
                    shutil.copy2(pred_result_path, os.path.join(media_predictions_dir, f"pred_only_{image.name}"))

                # Save input image
                with open(input_path, 'rb') as f:
                    prediction.input_image.save(
                        f'input_{int(time.time())}_{image.name}',
                        File(f),
                        save=False
                    )
                
                # Save output image
                with open(output_path, 'rb') as f:
                    prediction.output_image.save(
                        f'output_{int(time.time())}_{os.path.basename(output_path)}',
                        File(f),
                        save=False
                    )
                
                # Save the prediction
                prediction.save()
                
                messages.success(
                    self.request, 
                    f'Inference completed successfully in {processing_time:.2f} seconds!'
                )
                
            finally:
                # Clean up temporary directories
                if os.path.exists(temp_input_dir):
                    shutil.rmtree(temp_input_dir)
                if os.path.exists(temp_output_dir):
                    shutil.rmtree(temp_output_dir)
            
            # Redirect back to the inference page to show results
            return super().form_valid(form)
            
        except Exception as e:
            logging.error(f"Error during inference: {e}")
            messages.error(self.request, f'Error during inference: {e}')
            return self.form_invalid(form)


class SaveAsTemplateView(LoginRequiredMixin, FormView):
    form_class = TrainingTemplateForm
    template_name = 'ml_manager/save_as_template.html'
    success_url = reverse_lazy('ml_manager:template-list')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        self.model = get_object_or_404(MLModel, pk=self.kwargs['pk'])
        context['model'] = self.model
        return context
    
    def get_initial(self):
        """Pre-populate form with model's training configuration"""
        self.model = get_object_or_404(MLModel, pk=self.kwargs['pk'])
        initial = super().get_initial()
        
        if self.model.training_data_info:
            training_info = self.model.training_data_info
            initial.update({
                'name': f"{self.model.name} Template",
                'description': f"Template created from model: {self.model.name}",
                'model_type': training_info.get('model_type', 'unet'),
                'batch_size': training_info.get('batch_size', 32),
                'epochs': self.model.total_epochs,
                'learning_rate': training_info.get('learning_rate', 0.001),
                'validation_split': training_info.get('validation_split', 0.2),
                'resolution': training_info.get('resolution', '256'),
                'device': training_info.get('device', 'auto'),
                'use_random_flip': training_info.get('use_random_flip', True),
                'use_random_rotate': training_info.get('use_random_rotate', True),
                'use_random_scale': training_info.get('use_random_scale', True),
                'use_random_intensity': training_info.get('use_random_intensity', True),
                'crop_size': training_info.get('crop_size', 128),
                'num_workers': training_info.get('num_workers', 4),
                # Learning Rate Scheduler Configuration
                'lr_scheduler': training_info.get('lr_scheduler', 'plateau'),
                'lr_patience': training_info.get('lr_patience', 5),
                'lr_factor': training_info.get('lr_factor', 0.5),
                'lr_step_size': training_info.get('lr_step_size', 10),
                'lr_gamma': training_info.get('lr_gamma', 0.1),
                'min_lr': training_info.get('min_lr', 1e-7),
            })
        
        return initial
    
    def form_valid(self, form):
        try:
            template = form.save()
            messages.success(self.request, f'Template "{template.name}" has been created!')
            return super().form_valid(form)
            
        except Exception as e:
            logging.error(f"Error saving template: {e}")
            messages.error(self.request, f'Error saving template: {e}')
            return self.form_invalid(form)


# Training Template Views
class TrainingTemplateListView(LoginRequiredMixin, ListView):
    model = TrainingTemplate
    template_name = 'ml_manager/template_list.html'
    context_object_name = 'templates'
    paginate_by = 20


class TrainingTemplateCreateView(LoginRequiredMixin, FormView):
    form_class = TrainingTemplateForm
    template_name = 'ml_manager/template_form.html'
    success_url = reverse_lazy('ml_manager:template-list')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Create Training Template'
        context['submit_text'] = 'Create Template'
        return context
    
    def form_valid(self, form):
        try:
            template = form.save()
            messages.success(self.request, f'Template "{template.name}" has been created!')
            return super().form_valid(form)
        except Exception as e:
            logging.error(f"Error creating template: {e}")
            messages.error(self.request, f'Error creating template: {e}')
            return self.form_invalid(form)


class TrainingTemplateDetailView(LoginRequiredMixin, DetailView):
    model = TrainingTemplate
    template_name = 'ml_manager/template_detail.html'
    context_object_name = 'template'


class TrainingTemplateUpdateView(LoginRequiredMixin, FormView):
    form_class = TrainingTemplateForm
    template_name = 'ml_manager/template_form.html'
    success_url = reverse_lazy('ml_manager:template-list')
    
    def get_object(self):
        return get_object_or_404(TrainingTemplate, pk=self.kwargs['pk'])
    
    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['instance'] = self.get_object()
        return kwargs
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = f'Edit Template: {self.get_object().name}'
        context['submit_text'] = 'Update Template'
        return context
    
    def form_valid(self, form):
        try:
            template = form.save()
            messages.success(self.request, f'Template "{template.name}" has been updated!')
            return super().form_valid(form)
        except Exception as e:
            logging.error(f"Error updating template: {e}")
            messages.error(self.request, f'Error updating template: {e}')
            return self.form_invalid(form)


class TrainingTemplateDeleteView(LoginRequiredMixin, DeleteView):
    model = TrainingTemplate
    template_name = 'ml_manager/template_confirm_delete.html'
    success_url = reverse_lazy('ml_manager:template-list')
    
    def delete(self, request, *args, **kwargs):
        try:
            self.object = self.get_object()
            template_name = self.object.name
            self.object.delete()
            messages.success(request, f'Template "{template_name}" has been deleted.')
            return JsonResponse({'status': 'success'})
        except Exception as e:
            logging.error(f"Error deleting template: {e}")
            messages.error(request, f'Error deleting template: {e}')
            return JsonResponse({'status': 'error', 'message': str(e)})


@login_required
def get_template_data(request, template_id):
    """Get template data for AJAX requests"""
    try:
        template = get_object_or_404(TrainingTemplate, id=template_id)
        return JsonResponse({
            'status': 'success',
            'data': template.get_form_data()
        })
    except Exception as e:
        logging.error(f"Error getting template data: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })


# MLflow Registry functions (simplified versions)
@login_required
@require_POST
def register_model_in_registry(request, pk):
    """Register model in MLflow Model Registry"""
    try:
        model = get_object_or_404(MLModel, pk=pk)
        # Simplified - would implement actual MLflow registry logic
        model.is_registered = True
        model.save()
        
        return JsonResponse({
            'status': 'success',
            'message': f'Model "{model.name}" registered in MLflow Registry'
        })
    except Exception as e:
        logging.error(f"Error registering model: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})


@login_required
@require_POST
def transition_model_stage(request, pk):
    """Transition model stage in MLflow Registry"""
    try:
        model = get_object_or_404(MLModel, pk=pk)
        # Simplified - would implement actual stage transition logic
        
        return JsonResponse({
            'status': 'success',
            'message': 'Model stage transitioned successfully'
        })
    except Exception as e:
        logging.error(f"Error transitioning model stage: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})


@login_required
def sync_registry_info(request, pk):
    """Sync model registry information"""
    try:
        model = get_object_or_404(MLModel, pk=pk)
        # Simplified - would implement actual sync logic
        
        return JsonResponse({
            'status': 'success',
            'message': 'Registry information synced successfully'
        })
    except Exception as e:
        logging.error(f"Error syncing registry info: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})


@login_required
def registry_models_list(request):
    """List all registered models"""
    try:
        # Simplified - would implement actual registry listing
        return render(request, 'ml_manager/registry_list.html', {
            'models': []
        })
    except Exception as e:
        logging.error(f"Error listing registry models: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})


@login_required
def get_training_log(request, model_id):
    """Get training logs for a model"""
    try:
        model = get_object_or_404(MLModel, id=model_id)
        logs = []
        
        # Priority 1: Model-specific logs in model directory
        if model.model_directory and os.path.exists(model.model_directory):
            model_log_path = os.path.join(model.model_directory, 'logs', 'training.log')
            if os.path.exists(model_log_path):
                with open(model_log_path, 'r') as f:
                    logs = f.read().splitlines()
        
        # Priority 2: Search for model-specific logs in organized directory structure
        if not logs:
            import glob
            # Search for directories that might contain this model's logs
            search_patterns = [
                f'data/models/organized/*/*/unet-coronary/*{model.id}*',
                f'data/models/organized/*/*/unet-coronary/*{model.id}*',
                f'data/models/{model.id}*/logs/training.log',
                f'data/models/*{model.id}*/logs/training.log'
            ]
            
            for pattern in search_patterns:
                possible_logs = glob.glob(os.path.join(pattern, 'logs', 'training.log'))
                if possible_logs:
                    # Use the most recent log file
                    latest_log = max(possible_logs, key=os.path.getctime)
                    try:
                        with open(latest_log, 'r') as f:
                            logs = f.read().splitlines()
                        break
                    except Exception:
                        continue
            
            # If still not found, try directory search without specific model ID
            if not logs:
                # Get all model directories and try to find the most recent one
                model_dirs = glob.glob('data/models/organized/*/*/unet-coronary/*')
                if model_dirs:
                    # Sort by creation time, get the most recent
                    recent_dirs = sorted(model_dirs, key=os.path.getctime, reverse=True)
                    for recent_dir in recent_dirs[:5]:  # Check top 5 most recent
                        log_path = os.path.join(recent_dir, 'logs', 'training.log')
                        if os.path.exists(log_path):
                            try:
                                with open(log_path, 'r') as f:
                                    candidate_logs = f.read().splitlines()
                                # Check if this log contains references to our model
                                model_ref_found = any(str(model.id) in line or 
                                                    f'model {model.id}' in line.lower() or
                                                    f'model_{model.id}' in line.lower()
                                                    for line in candidate_logs[:50])  # Check first 50 lines
                                if model_ref_found:
                                    logs = candidate_logs
                                    break
                            except Exception:
                                continue
        
        # Priority 3: Global training log (only as last resort, filtered for this model)
        if not logs:
            global_log_path = os.path.join('data', 'models', 'artifacts', 'training.log')
            if os.path.exists(global_log_path):
                with open(global_log_path, 'r') as f:
                    all_logs = f.read().splitlines()
                    # Filter for logs that specifically mention this model
                    model_specific_logs = [
                        line for line in all_logs 
                        if (str(model.id) in line and 
                            ('model' in line.lower() or 'training' in line.lower())) or
                           f'model {model.id}' in line.lower() or
                           f'model_{model.id}' in line.lower() or
                           ('[TRAIN]' in line or '[EPOCH]' in line or '[VAL]' in line or
                            '[METRICS]' in line or '[CONFIG]' in line or '[MODEL]' in line or
                            'ERROR' in line or 'WARNING' in line)
                    ]
                    if model_specific_logs:
                        logs = model_specific_logs
                    else:
                        # If no model-specific content found, don't show global logs
                        logs = [f"No specific training logs found for model {model.id}",
                               f"This model may be new or training logs are not yet available."]
        
        # If still no logs, check alternative locations
        if not logs:
            # Try MLflow artifacts path
            if model.mlflow_run_id:
                mlflow_log_path = os.path.join('mlruns', model.mlflow_run_id, 'artifacts', 'training.log')
                if os.path.exists(mlflow_log_path):
                    with open(mlflow_log_path, 'r') as f:
                        logs = f.read().splitlines()
        
        if not logs:
            logs = ["No training logs found", f"Checked locations:", 
                   f"- Model directory: {model.model_directory}/logs/training.log" if model.model_directory else "- No model directory set",
                   f"- Global log: data/models/artifacts/training.log", 
                   f"- Training volume: logs/training.log",
                   f"- MLflow artifacts: mlruns/{model.mlflow_run_id}/artifacts/training.log" if model.mlflow_run_id else "- No MLflow run ID"]
        
        return JsonResponse({
            'status': 'success',
            'logs': logs
        })
        
    except Exception as e:
        logging.error(f"Error getting training logs: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})


@login_required
def get_training_progress(request, model_id):
    """Get training progress for a model"""
    try:
        model = get_object_or_404(MLModel, id=model_id)
        
        # Check for status changes by comparing with session data
        session_key = f'model_{model_id}_last_status'
        last_known_status = request.session.get(session_key)
        current_status = model.status
        status_changed = last_known_status != current_status
        
        # Store current status for next comparison
        request.session[session_key] = current_status
        
        # Enhanced status transition detection
        status_transition = None
        if status_changed and last_known_status:
            status_transition = f"{last_known_status} → {current_status}"
            logging.info(f"Status transition detected for model {model_id}: {status_transition}")
        
        return JsonResponse({
            'status': 'success',
            'model_status': current_status,
            'previous_status': last_known_status,
            'status_changed': status_changed,
            'status_transition': status_transition,
            'progress': {
                'current_epoch': model.current_epoch or 0,
                'total_epochs': model.total_epochs or 0,
                'current_batch': model.current_batch or 0,
                'total_batches_per_epoch': model.total_batches_per_epoch or 0,
                'percentage': model.progress_percentage,
                'batch_progress_percentage': model.batch_progress_percentage,
            },
            'metrics': {
                'train_loss': model.train_loss,
                'val_loss': model.val_loss,
                'train_dice': model.train_dice,
                'val_dice': model.val_dice,
                'best_val_dice': model.best_val_dice or 0.0,
                # Also include accuracy metrics from performance_metrics if available
                'train_accuracy': model.performance_metrics.get('train_accuracy', 0.0),
                'val_accuracy': model.performance_metrics.get('val_accuracy', 0.0),
                'best_val_accuracy': model.performance_metrics.get('best_val_accuracy', 0.0),
            },
            'timestamp': model.updated_at.isoformat() if hasattr(model, 'updated_at') else None
        })
        
    except Exception as e:
        logging.error(f"Error getting training progress: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})


@login_required
def serve_training_preview_image(request, model_id, filename):
    """Serve training preview images"""
    try:
        model = get_object_or_404(MLModel, id=model_id)
        logging.info(f"Serving image {filename} for model {model_id}, run_id: {model.mlflow_run_id}")
        
        # Find image file using multiple path strategies
        if model.mlflow_run_id:
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(model.mlflow_run_id)
            
            # Try multiple artifact and prediction directory paths to handle different MLflow configurations
            base_paths = [
                # PRIORITY: Check model-specific directory first (most accurate for organized structure)
                os.path.join(model.model_directory, 'predictions') if model.model_directory else None,
                os.path.join(model.model_directory, 'artifacts') if model.model_directory else None,
                # Fallback: Check organized model directory structure by run_id (legacy)
                os.path.join(settings.BASE_ORGANIZED_MODELS_DIR, run.info.run_id, 'predictions'),
                os.path.join(settings.BASE_ORGANIZED_MODELS_DIR, run.info.run_id, 'artifacts'),
                # Direct run ID path (current MLflow structure)
                os.path.join(settings.BASE_MLRUNS_DIR, run.info.run_id, 'artifacts'),
                # Legacy experiment-based paths  
                os.path.join(settings.BASE_MLRUNS_DIR, run.info.experiment_id, run.info.run_id, 'artifacts'),
                os.path.join(settings.BASE_MLRUNS_DIR, '0', run.info.run_id, 'artifacts'),
                os.path.join(settings.BASE_MLRUNS_DIR, '1', run.info.run_id, 'artifacts'),
                os.path.join(settings.BASE_MLRUNS_DIR, str(run.info.experiment_id), run.info.run_id, 'artifacts'),
                # Fallback to legacy mlruns structure
                os.path.join('mlruns', run.info.run_id, 'artifacts'),
                os.path.join('mlruns', run.info.experiment_id, run.info.run_id, 'artifacts'),
                os.path.join('mlruns', '0', run.info.run_id, 'artifacts'),
                os.path.join('mlruns', '1', run.info.run_id, 'artifacts'),
                os.path.join('mlruns', str(run.info.experiment_id), run.info.run_id, 'artifacts'),
                # Additional paths for data/mlflow and data/mlruns
                os.path.join(settings.BASE_MLRUNS_DIR, run.info.run_id, 'artifacts'),
            ]
            
            # Filter out None paths
            base_paths = [path for path in base_paths if path is not None]
            
            logging.info(f"Searching in base paths: {base_paths}")
            
            # Search in each base path
            for base_path in base_paths:
                if not os.path.exists(base_path):
                    logging.debug(f"Base path does not exist: {base_path}")
                    continue
                    
                # Try direct path first
                direct_path = os.path.join(base_path, filename)
                logging.debug(f"Trying direct path: {direct_path}")
                if os.path.exists(direct_path):
                    logging.info(f"Found image at direct path: {direct_path}")
                    with open(direct_path, 'rb') as f:
                        response = HttpResponse(f.read(), content_type='image/png')
                        response['Content-Disposition'] = f'inline; filename="{filename}"'
                        return response
                
                # Search in subdirectories (predictions/, visualizations/, etc.)
                logging.debug(f"Walking directory tree from: {base_path}")
                for root, dirs, files in os.walk(base_path):
                    if filename in files:
                        image_path = os.path.join(root, filename)
                        logging.info(f"Found image at: {image_path}")
                        with open(image_path, 'rb') as f:
                            response = HttpResponse(f.read(), content_type='image/png')
                            response['Content-Disposition'] = f'inline; filename="{filename}"'
                            return response
        
        # Return 404 if image not found
        logging.warning(f"Training preview image not found: {filename} for model {model_id}")
        return HttpResponse('Image not found', status=404)
        
    except Exception as e:
        logging.error(f"Error serving training preview image: {e}")
        return HttpResponse('Error loading image', status=500)


class ModelLogsView(LoginRequiredMixin, DetailView):
    """View for displaying and filtering training logs with AJAX support"""
    model = MLModel
    template_name = 'ml_manager/model_logs.html'
    context_object_name = 'model'
    
    def get(self, request, *args, **kwargs):
        self.object = self.get_object()
        
        # Get filter parameters
        log_type = request.GET.get('type', 'all')  # all, epochs, metrics, errors
        search = request.GET.get('search', '')
        
        # Read training logs
        logs = []
        log_file_path = None
        if self.object.model_directory:
            log_file_path = os.path.join(self.object.model_directory, 'training.log')
        elif self.object.mlflow_run_id:
            log_file_path = os.path.join(str(settings.BASE_MLRUNS_DIR), self.object.mlflow_run_id, 'artifacts', 'training_logs', 'training.log')
        if log_file_path and os.path.exists(log_file_path):
            try:
                with open(log_file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        # Exclude DEBUG logs from all views (case-insensitive)
                        if 'debug' in line.lower():
                            continue
                        if 'Not Found: /ml/model/' in line and '/progress/' in line:
                            continue
                        # Only include important logs for 'all' view
                        important = (
                            'ERROR' in line or 'Exception' in line or 'WARNING' in line or 'WARN' in line or
                            '[EPOCH]' in line or '[TRAIN]' in line or '[VAL]' in line or '[METRICS]' in line or '[STATS]' in line or '[MODEL]' in line or '[CONFIG]' in line
                        )
                        if not important and log_type == 'all':
                            continue
                        if log_type == 'epochs' and '[EPOCH]' not in line:
                            continue
                        elif log_type == 'metrics' and not any(word in line.lower() for word in ['loss', 'dice', 'accuracy', '[metrics]', '[stats]']):
                            continue
                        elif log_type == 'errors' and not any(word in line for word in ['ERROR', 'Exception', 'error']):
                            continue
                        elif log_type == 'batches' and '[TRAIN]' not in line and '[VAL]' not in line:
                            continue
                        if search and search.lower() not in line.lower():
                            continue
                        logs.append({
                            'line_number': line_num,
                            'content': line,
                            'timestamp': self._extract_timestamp(line),
                            'level': self._extract_log_level(line)
                        })
            except Exception as e:
                logs.append({
                    'line_number': 1,
                    'content': f'Error reading log file: {e}',
                    'timestamp': None,
                    'level': 'ERROR'
                })
        
        # Handle AJAX requests with ETag support
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            import hashlib
            from django.utils.http import http_date
            from django.http import HttpResponseNotModified
            
            # Generate ETag based on log content and filters
            etag_data = f"{len(logs)}:{log_type}:{search}:{self.object.updated_at.isoformat()}"
            if logs:
                # Include hash of last few log entries for more granular change detection
                last_logs = logs[-min(5, len(logs)):]
                last_content = ''.join([log['content'] for log in last_logs])
                etag_data += f":{hashlib.md5(last_content.encode()).hexdigest()[:8]}"
            
            etag = hashlib.md5(etag_data.encode()).hexdigest()
            
            # Check if client has current version
            client_etag = request.headers.get('If-None-Match')
            if client_etag and client_etag.strip('"') == etag:
                return HttpResponseNotModified()
            
            response_data = {
                'logs': logs,
                'total_lines': len(logs),
                'log_file_path': log_file_path,
                'filters': {
                    'type': log_type,
                    'search': search
                }
            }
            
            response = JsonResponse(response_data)
            response['ETag'] = f'"{etag}"'
            response['Last-Modified'] = http_date(self.object.updated_at.timestamp())
            response['Cache-Control'] = 'no-cache, must-revalidate'
            
            return response
        
        context = self.get_context_data()
        context.update({
            'logs': logs,
            'log_type': log_type,
            'search': search,
            'log_file_path': log_file_path
        })
        return self.render_to_response(context)
    
    def _extract_timestamp(self, line):
        """Extract timestamp from log line"""
        import re
        timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
        match = re.search(timestamp_pattern, line)
        return match.group(1) if match else None
    
    def _extract_log_level(self, line):
        """Extract log level from log line"""
        if 'ERROR' in line:
            return 'ERROR'
        elif 'WARNING' in line or 'WARN' in line:
            return 'WARNING'
        elif 'INFO' in line:
            return 'INFO'
        elif 'DEBUG' in line:
            return 'DEBUG'
        elif '[EPOCH]' in line:
            return 'EPOCH'
        elif '[TRAIN]' in line or '[VAL]' in line:
            return 'TRAINING'
        elif '[METRICS]' in line or '[STATS]' in line:
            return 'METRICS'
        else:
            return 'INFO'
    
    def _extract_runtime_device_from_logs(self):
        """Extract the actual runtime device from training logs"""
        try:
            # First try to get logs from the current model training
            if hasattr(self, 'object') and self.object:
                training_logs = self._get_training_logs()
                if training_logs:
                    # Look for device information in logs
                    for log_line in training_logs:
                        line_content = log_line if isinstance(log_line, str) else str(log_line)
                        
                        # Look for the specific log pattern from train.py
                        if '[TRAINING] Using device:' in line_content:
                            # Extract device from log line like: "[TRAINING] Using device: cuda"
                            import re
                            device_match = re.search(r'\[TRAINING\] Using device:\s*(\w+)', line_content)
                            if device_match:
                                return device_match.group(1).lower()
                        
                        # Alternative patterns to catch device information
                        elif 'Device:' in line_content and any(dev in line_content.lower() for dev in ['cuda', 'cpu', 'mps']):
                            # Extract from patterns like "Device: cuda" or "Device: cpu"
                            device_match = re.search(r'Device:\s*(\w+)', line_content, re.IGNORECASE)
                            if device_match:
                                return device_match.group(1).lower()
                        
                        # Also check for CUDA availability logs
                        elif 'cuda.is_available()' in line_content.lower():
                            if 'true' in line_content.lower() or 'available' in line_content.lower():
                                return 'cuda'
                            elif 'false' in line_content.lower() or 'not available' in line_content.lower():
                                return 'cpu'
                            
            return None
        except Exception as e:
            # Log the error but don't fail the whole view
            import logging
            logging.warning(f"Could not extract runtime device from logs: {e}")
            return None

    def _get_training_logs(self):
        """Get training logs for this model with enhanced model-specific prioritization"""
        try:
            model_specific_logs = []
            
            # Priority 1: Model-specific logs in model directory
            if self.object.model_directory and os.path.exists(self.object.model_directory):
                # Try multiple log locations within model directory
                model_log_paths = [
                    os.path.join(self.object.model_directory, 'logs', 'training.log'),
                    os.path.join(self.object.model_directory, 'training.log'),
                    os.path.join(self.object.model_directory, 'training_logs', 'training.log')
                ]
                
                for log_path in model_log_paths:
                    if os.path.exists(log_path):
                        try:
                            with open(log_path, 'r') as f:
                                model_specific_logs = f.read().splitlines()
                            break
                        except Exception as e:
                            continue
            
            # Priority 2: Search in organized directory structure
            if not model_specific_logs:
                import glob
                search_patterns = [
                    f'data/models/organized/*/*/unet-coronary/*{self.object.id}*',
                    f'data/models/{self.object.id}*',
                    f'data/models/*{self.object.id}*'
                ]
                
                for pattern in search_patterns:
                    possible_dirs = glob.glob(pattern)
                    for dir_path in possible_dirs:
                        log_path = os.path.join(dir_path, 'logs', 'training.log')
                        if os.path.exists(log_path):
                            try:
                                with open(log_path, 'r') as f:
                                    model_specific_logs = f.read().splitlines()
                                break
                            except Exception:
                                continue
                    if model_specific_logs:
                        break
                
                # If still not found, try to find the most recent model directory
                if not model_specific_logs:
                    model_dirs = glob.glob('data/models/organized/*/*/unet-coronary/*')
                    if model_dirs:
                        recent_dirs = sorted(model_dirs, key=os.path.getctime, reverse=True)
                        for recent_dir in recent_dirs[:3]:  # Check top 3 most recent
                            log_path = os.path.join(recent_dir, 'logs', 'training.log')
                            if os.path.exists(log_path):
                                try:
                                    with open(log_path, 'r') as f:
                                        candidate_logs = f.read().splitlines()
                                    # Check if this log mentions our specific model
                                    model_ref_found = any(str(self.object.id) in line for line in candidate_logs[:50])
                                    if model_ref_found:
                                        model_specific_logs = candidate_logs
                                        break
                                except Exception:
                                    continue
            
            # Priority 3: MLflow artifacts if available
            if not model_specific_logs and self.object.mlflow_run_id:
                mlflow_log_paths = [
                    os.path.join(str(settings.BASE_MLRUNS_DIR), self.object.mlflow_run_id, 'artifacts', 'training_logs', 'training.log'),
                    os.path.join('mlruns', self.object.mlflow_run_id, 'artifacts', 'training_logs', 'training.log'),
                    f'artifacts/training_logs/training_{self.object.id}.log'
                ]
                
                for log_path in mlflow_log_paths:
                    if os.path.exists(log_path):
                        try:
                            with open(log_path, 'r') as f:
                                model_specific_logs = f.read().splitlines()
                                break
                        except Exception as e:
                            continue
            
            # Priority 4: Global log only if model-specific content found
            if not model_specific_logs:
                global_log_path = os.path.join('data', 'models', 'artifacts', 'training.log')
                if os.path.exists(global_log_path):
                    try:
                        with open(global_log_path, 'r') as f:
                            all_logs = f.read().splitlines()
                        # Filter for logs that specifically mention this model
                        filtered_logs = [
                            line for line in all_logs 
                            if (str(self.object.id) in line and 
                                ('model' in line.lower() or 'training' in line.lower())) or
                               f'model {self.object.id}' in line.lower() or
                               f'model_{self.object.id}' in line.lower()
                        ]
                        if filtered_logs:
                            model_specific_logs = filtered_logs
                    except Exception as e:
                        pass
            
            # Return model-specific logs or empty list if none found
            return model_specific_logs if model_specific_logs else []
            
        except Exception as e:
            import logging
            logging.warning(f"Could not load training logs for model {self.object.id}: {e}")
            return []


@login_required
def mlflow_redirect_view(request):
    """Redirect to MLflow UI (use correct host, not mlflow:5000)"""
    try:
        # Try to get MLflow UI URL from settings
        mlflow_ui_url = getattr(settings, 'MLFLOW_UI_URL', None)
        if not mlflow_ui_url:
            # Fallback to tracking URI, but replace 0.0.0.0 with localhost for browser
            mlflow_uri = getattr(settings, 'MLFLOW_TRACKING_URI', 'http://localhost:5000')
            mlflow_ui_url = mlflow_uri.replace('0.0.0.0', 'localhost').replace('127.0.0.1', 'localhost')
        return redirect(mlflow_ui_url)
    except Exception as e:
        messages.error(request, f"Could not redirect to MLflow: {e}")
        return redirect('ml_manager:model-list')

def apply_semantic_colormap(mask_array, encoding=None, colors=None):
    """Apply color mapping to semantic segmentation mask"""
    import numpy as np
    from PIL import Image
    
    if encoding is None:
        # Default ARCADE encoding
        encoding = {
            "background": 0,
            "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
            "6": 6, "7": 7, "8": 8, "9": 9, "9a": 10,
            "10": 11, "10a": 12, "11": 13, "12": 14, "12a": 15,
            "13": 16, "14": 17, "14a": 18, "15": 19, "16": 20,
            "16a": 21, "16b": 22, "16c": 23, "12b": 24, "14b": 25,
            "stenosis": 26
        }
    
    if colors is None:
        # Enhanced ARCADE colors - more distinct colors for better visualization
        colors = np.array([
            [0, 0, 0],        # background - black
            [255, 0, 0],      # segment 1 - red
            [0, 255, 0],      # segment 2 - green
            [0, 0, 255],      # segment 3 - blue
            [255, 255, 0],    # segment 4 - yellow
            [255, 0, 255],    # segment 5 - magenta
            [0, 255, 255],    # segment 6 - cyan
            [255, 165, 0],    # segment 7 - orange
            [128, 0, 128],    # segment 8 - purple
            [255, 192, 203],  # segment 9 - pink
            [173, 255, 47],   # segment 9a - green yellow
            [30, 144, 255],   # segment 10 - dodger blue
            [255, 20, 147],   # segment 10a - deep pink
            [0, 250, 154],    # segment 11 - medium spring green
            [255, 69, 0],     # segment 12 - red orange
            [72, 61, 139],    # segment 12a - dark slate blue
            [255, 215, 0],    # segment 13 - gold
            [220, 20, 60],    # segment 14 - crimson
            [124, 252, 0],    # segment 14a - lawn green
            [186, 85, 211],   # segment 15 - medium orchid
            [138, 43, 226],   # segment 16 - blue violet
            [255, 105, 180],  # segment 16a - hot pink
            [255, 140, 0],    # segment 16b - dark orange
            [184, 134, 11],   # segment 16c - dark goldenrod
            [70, 130, 180],   # segment 12b - steel blue
            [0, 206, 209],    # segment 14b - dark turquoise
            [255, 99, 71]     # stenosis - tomato red
        ])
    
    # If mask is 2D, convert to color
    if len(mask_array.shape) == 2:
        height, width = mask_array.shape
        color_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Map each unique value to its color
        unique_values = np.unique(mask_array)
        logger = logging.getLogger(__name__)
        logger.info(f"Applying semantic colormap to mask with unique values: {unique_values}")
        
        for val in unique_values:
            if val < len(colors):
                color_mask[mask_array == val] = colors[val]
                logger.info(f"Mapped class {val} to color {colors[val]}")
            else:
                # Default to white for unknown values
                color_mask[mask_array == val] = [255, 255, 255]
                logger.warning(f"Unknown class {val}, using white color")
        
        return color_mask
    
    # If mask is already 3D (RGB), return as is
    elif len(mask_array.shape) == 3 and mask_array.shape[2] == 3:
        return mask_array
    
    # If mask is multi-channel (one-hot), convert to single channel first
    elif len(mask_array.shape) == 3:
        # Convert one-hot to single channel
        single_channel = np.argmax(mask_array, axis=2)
        return apply_semantic_colormap(single_channel, encoding, colors)
    
    return mask_array

def detect_dataset_type(data_path):
    """Detect the type of dataset and return info about it"""
    import json
    
    dataset_info = {
        'total_samples': 'Unknown',
        'structure': 'Unknown'
    }
    
    if not os.path.exists(data_path):
        return 'not_found', dataset_info
    
    # Check for ARCADE COCO structure
    images_dir = os.path.join(data_path, 'images')
    annotations_dir = os.path.join(data_path, 'annotations')
    
    if os.path.exists(images_dir) and os.path.exists(annotations_dir):
        # Count images
        img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        dataset_info['total_samples'] = len(img_files)
        
        # Check for COCO annotation files
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
        if annotation_files:
            # Try to read first annotation file to detect ARCADE
            try:
                ann_file = os.path.join(annotations_dir, annotation_files[0])
                with open(ann_file, 'r') as f:
                    coco_data = json.load(f)
                
                # Check if it has ARCADE-specific categories
                if 'categories' in coco_data:
                    categories = [cat['name'] for cat in coco_data['categories']]
                    arcade_segments = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
                    
                    if any(seg in categories for seg in arcade_segments):
                        dataset_info['structure'] = 'ARCADE COCO'
                        dataset_info['categories'] = len(categories)
                        return 'arcade_coco', dataset_info
                
                dataset_info['structure'] = 'COCO'
                return 'coco_style', dataset_info
                
            except Exception:
                pass
        
        return 'coco_style', dataset_info
    # Check for standard coronary structure
    imgs_dir = os.path.join(data_path, 'imgs')
    masks_dir = os.path.join(data_path, 'masks')
    
    if os.path.exists(imgs_dir) and os.path.exists(masks_dir):
        img_files = [f for f in os.listdir(imgs_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        dataset_info['total_samples'] = len(img_files)
        dataset_info['structure'] = 'Standard (imgs/masks)'
        return 'coronary_standard', dataset_info
    
    # Check for MONAI style
    images_dir = os.path.join(data_path, 'images')
    labels_dir = os.path.join(data_path, 'labels')
    
    if os.path.exists(images_dir) and os.path.exists(labels_dir):
        img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        dataset_info['total_samples'] = len(img_files)
        dataset_info['structure'] = 'MONAI (images/labels)'
        return 'monai_style', dataset_info
    
    return 'unknown', dataset_info

def dataset_preview_view(request):
    """Show dataset preview with sample images and masks"""
    import logging
    import uuid
    import numpy as np
    from PIL import Image
    
    logger = logging.getLogger(__name__)
    context = {}
    error_message = None
    samples = []  # Initialize samples list
    
    # Dataset type choices including all 6 ARCADE dataset types
    dataset_type_choices = [
        ('auto', 'Auto-detect'),
        ('coronary_standard', 'Coronary Standard (imgs/masks)'),
        ('monai_style', 'MONAI Style (images/labels)'),
        ('coco_style', 'COCO Style (images/annotations)'),
        ('arcade_binary_segmentation', 'ARCADE: Binary Segmentation (image → binary mask)'),
        ('arcade_semantic_segmentation', 'ARCADE: Semantic Segmentation (image → 27-class mask)'),
        ('arcade_artery_classification', 'ARCADE: Artery Classification (binary mask → left/right)'),
        ('arcade_semantic_seg_binary', 'ARCADE: Semantic from Binary (binary mask → 26-class mask)'),
        ('arcade_stenosis_detection', 'ARCADE: Stenosis Detection (image → bounding boxes)'),
        ('arcade_stenosis_segmentation', 'ARCADE: Stenosis Segmentation (image → stenosis mask)'),
    ]
    context['dataset_type_choices'] = dataset_type_choices
    
    # Set default values
    if request.method == 'GET':
        context['data_path'] = '/app/data/datasets/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset/seg_train'
        context['dataset_type'] = 'auto'
        context['detected_type'] = None
        context['dataset_info'] = None
        context['total_samples'] = None
    else:
        context['data_path'] = request.POST.get('data_path', '/app/data/datasets/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset/seg_train')
        context['dataset_type'] = request.POST.get('dataset_type', 'auto')
    
    if request.method == 'POST':
        data_path = request.POST.get('data_path', '').strip()
        dataset_type = request.POST.get('dataset_type', 'auto')
        
        if not data_path:
            error_message = "Please provide a dataset path"
        elif not os.path.exists(data_path):
            error_message = f"Dataset path does not exist: {data_path}"
        else:
            try:
                # Detect dataset type using custom function
                detected_type, dataset_info = detect_dataset_type(data_path)
                logger.info(f"Detected dataset type: {detected_type}, Selected type: {dataset_type}")
                
                context['detected_type'] = detected_type
                context['dataset_info'] = dataset_info
                context['total_samples'] = dataset_info.get('total_samples', dataset_info.get('images', 'Unknown'))
                
                # Generate sample images
                samples = []
                 # Handle standard coronary datasets
                if detected_type in ['coronary_standard', 'monai_style'] or dataset_type in ['coronary_standard', 'monai_style']:
                    try:
                        if detected_type == 'coronary_standard' or dataset_type == 'coronary_standard':
                            imgs_dir = os.path.join(data_path, 'imgs')
                            masks_dir = os.path.join(data_path, 'masks')
                        else:  # monai_style
                            imgs_dir = os.path.join(data_path, 'images')
                            masks_dir = os.path.join(data_path, 'labels')
                        
                        if os.path.exists(imgs_dir) and os.path.exists(masks_dir):
                            img_files = [f for f in os.listdir(imgs_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                            mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                            
                            img_files.sort()
                            mask_files.sort()
                            
                            max_samples = min(6, len(img_files), len(mask_files))
                            
                            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp', 'dataset_preview')
                            os.makedirs(temp_dir, exist_ok=True)
                            
                            for i in range(max_samples):
                                try:
                                    sample_id = str(uuid.uuid4())
                                    
                                    # Load original files
                                    img_file = img_files[i]
                                    # Find corresponding mask file
                                    mask_file = None
                                    img_name = os.path.splitext(img_file)[0]
                                    
                                    # Try different mask naming conventions
                                    for ext in ['.png', '.jpg', '.jpeg']:
                                        candidate = img_name + ext
                                        if candidate in mask_files:
                                            mask_file = candidate
                                            break
                                    
                                    if not mask_file and mask_files:
                                        # Fallback to positional matching
                                        if i < len(mask_files):
                                            mask_file = mask_files[i]
                                    
                                    if not mask_file:
                                        continue
                                    
                                    img_path = os.path.join(imgs_dir, img_file)
                                    mask_path = os.path.join(masks_dir, mask_file)
                                    
                                    # Load and analyze images
                                    img = Image.open(img_path)
                                    mask = Image.open(mask_path)
                                    
                                    # Convert to numpy for analysis
                                    img_array = np.array(img)
                                    mask_array = np.array(mask)
                                    
                                    logger.info(f"[GENERIC COCO SECTION] Processing mask: dtype={mask_array.dtype}, shape={mask_array.shape}, min={mask_array.min()}, max={mask_array.max()}")
                                    
                                    # Convert mask to grayscale if needed
                                    if len(mask_array.shape) == 3:
                                        mask_array = mask_array[:,:,0]  # Take first channel
                                        
                                    # BINARY MASK SCALING FIX - ensure proper 0-255 range for visibility
                                    if mask_array.max() <= 1.0:
                                        mask_array = (mask_array * 255).astype(np.uint8)
                                        # Apply full contrast for binary masks
                                        mask_array = np.where(mask_array > 0, 255, 0).astype(np.uint8)
                                        logger.info(f"[GENERIC COCO SECTION] FIXED binary mask to full contrast: min={mask_array.min()}, max={mask_array.max()}")
                                    else:
                                        logger.info(f"[GENERIC COCO SECTION] Mask already in good range: min={mask_array.min()}, max={mask_array.max()}")
                                    
                                    # Copy images to temp directory for web display
                                    img_copy_path = os.path.join(temp_dir, f'image_{sample_id}.jpg')
                                    mask_copy_path = os.path.join(temp_dir, f'mask_{sample_id}.png')
                                    
                                    # Save image copy
                                    if img.mode != 'RGB':
                                        img = img.convert('RGB')
                                    img.save(img_copy_path, 'JPEG')
                                    
                                    # Save mask copy (convert to grayscale if needed) 
                                    # Apply the scaled mask_array back to PIL Image for saving
                                    if len(mask_array.shape) == 2:
                                        # Use the scaled mask_array for proper visibility
                                        mask_for_save = Image.fromarray(mask_array, mode='L')
                                    else:
                                        # Fallback to original conversion if needed
                                        if len(np.array(mask).shape) == 3:
                                            mask_for_save = mask.convert('L')
                                        else:
                                            mask_for_save = mask
                                    mask_for_save.save(mask_copy_path, 'PNG')
                                    logger.info(f"[STANDARD DATASET] Saved scaled mask to {mask_copy_path}")
                                    
                                    # Create dynamic URLs using Django settings
                                    img_url = f'{settings.MEDIA_URL}temp/dataset_preview/image_{sample_id}.jpg'
                                    mask_url = f'{settings.MEDIA_URL}temp/dataset_preview/mask_{sample_id}.png'
                                    
                                    # BINARY MASK SCALING FIX - This is the missing piece for standard datasets!
                                    # Apply the same scaling logic as in ARCADE sections
                                    logger.info(f"[STANDARD DATASET] Mask before scaling: dtype={mask_array.dtype}, min={mask_array.min()}, max={mask_array.max()}")
                                    if mask_array.max() <= 1.0:
                                        mask_array = (mask_array * 255).astype(np.uint8)
                                        # Apply full contrast for binary masks
                                        mask_array = np.where(mask_array > 0, 255, 0).astype(np.uint8)
                                        logger.info(f"[STANDARD DATASET] FIXED binary mask to full contrast: min={mask_array.min()}, max={mask_array.max()}")
                                    else:
                                        logger.info(f"[STANDARD DATASET] Mask already in good range: min={mask_array.min()}, max={mask_array.max()}")
                                    
                                    # Calculate mask coverage
                                    if len(mask_array.shape) == 2:
                                        foreground_pixels = np.sum(mask_array > 0)
                                        total_pixels = mask_array.size
                                        mask_coverage = (foreground_pixels / total_pixels) * 100
                                    else:
                                        mask_coverage = 0.0
                                    
                                    # Create analysis text
                                    analysis_text = f'Standard dataset: {int(np.sum(mask_array > 0))} foreground pixels ({mask_coverage:.1f}% coverage)'
                                    
                                    samples.append({
                                        'index': i,
                                        'filename': img_file,
                                        'image_url': img_url,
                                        'mask_url': mask_url,
                                        'image_shape': img_array.shape,
                                        'mask_shape': mask_array.shape,
                                        'image_min': float(img_array.min()) if hasattr(img_array, 'min') else 'N/A',
                                        'image_max': float(img_array.max()) if hasattr(img_array, 'max') else 'N/A',
                                        'mask_min': float(mask_array.min()) if hasattr(mask_array, 'min') else 'N/A',
                                        'mask_max': float(mask_array.max()) if hasattr(mask_array, 'max') else 'N/A',
                                        'mask_coverage': mask_coverage,
                                        'analysis': analysis_text
                                    })
                                except Exception as e:
                                    logger.error(f"Error processing sample {i}: {e}")
                                    continue
                        else:
                            error_message = f"Expected directories not found. Looking for: {imgs_dir}, {masks_dir}"
                    
                    except Exception as e:
                        error_message = f"Error loading dataset: {str(e)}"
                        logger.error(f"Dataset error: {e}", exc_info=True)
                
                # Handle specific ARCADE dataset types
                elif dataset_type.startswith('arcade_'):
                    try:
                        from ml.datasets.torch_arcade_loader import (
                            ARCADEBinarySegmentation, ARCADESemanticSegmentation, 
                            ARCADEArteryClassification, ARCADESemanticSegmentationBinary,
                            ARCADEStenosisDetection, ARCADEStenosisSegmentation,
                            COCO_AVAILABLE
                        )
                        
                        if not COCO_AVAILABLE:
                            error_message = "pycocotools not available. Install with: pip install pycocotools"
                        else:
                            # Map dataset type to ARCADE class
                            arcade_class_map = {
                                'arcade_binary_segmentation': ('ARCADEBinarySegmentation', ARCADEBinarySegmentation),
                                'arcade_semantic_segmentation': ('ARCADESemanticSegmentation', ARCADESemanticSegmentation),
                                'arcade_artery_classification': ('ARCADEArteryClassification', ARCADEArteryClassification),
                                'arcade_semantic_seg_binary': ('ARCADESemanticSegmentationBinary', ARCADESemanticSegmentationBinary),
                                'arcade_stenosis_detection': ('ARCADEStenosisDetection', ARCADEStenosisDetection),
                                'arcade_stenosis_segmentation': ('ARCADEStenosisSegmentation', ARCADEStenosisSegmentation),
                            }
                            
                            if dataset_type not in arcade_class_map:
                                error_message = f"Unknown ARCADE dataset type: {dataset_type}"
                            else:
                                class_name, arcade_class = arcade_class_map[dataset_type]
                                logger.info(f"Using ARCADE class: {class_name}")
                                
                                # Determine image_set from path structure
                                image_set = "train"  # default
                                if "val" in data_path.lower():
                                    image_set = "val"
                                elif "test" in data_path.lower():
                                    image_set = "test"
                                
                                # Get parent directory (should contain the full ARCADE structure)
                                arcade_root = data_path
                                while arcade_root and not os.path.exists(os.path.join(arcade_root, "arcade_challenge_datasets")):
                                    parent = os.path.dirname(arcade_root)
                                    if parent == arcade_root:  # reached filesystem root
                                        break
                                    arcade_root = parent
                                
                                if os.path.exists(os.path.join(arcade_root, "arcade_challenge_datasets")):
                                    if os.path.basename(arcade_root) == "arcade_challenge_datasets":
                                        arcade_root = os.path.dirname(arcade_root)
                                
                                if os.path.exists(os.path.join(arcade_root, "arcade_challenge_datasets", "dataset_phase_1")):
                                    logger.info(f"Creating {class_name} dataset from: {arcade_root}")
                                    
                                    # Create the specific ARCADE dataset
                                    arcade_dataset = arcade_class(
                                        root=arcade_root,
                                        image_set=image_set,
                                        download=False
                                    )
                                    
                                    # Generate samples with appropriate processing for each type
                                    max_samples = min(6, len(arcade_dataset))
                                    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp', 'dataset_preview')
                                    os.makedirs(temp_dir, exist_ok=True)
                                    
                                    for i in range(max_samples):
                                        try:
                                            logger.info(f"[ARCADE] Starting sample {i} processing...")
                                            sample_id = str(uuid.uuid4())
                                            
                                            # Get data from ARCADE dataset
                                            item = arcade_dataset[i]
                                            
                                            if dataset_type == 'arcade_artery_classification':
                                                # (binary_mask, classification_label)
                                                mask_input, class_label = item
                                                
                                                # Convert binary mask to proper RGB image for display
                                                if isinstance(mask_input, torch.Tensor):
                                                    mask_array = mask_input.squeeze().numpy()
                                                    logger.info(f"Artery classification mask range: {mask_array.min():.3f} - {mask_array.max():.3f}, dtype: {mask_array.dtype}")
                                                    
                                                    # Proper scaling logic
                                                    if mask_array.max() <= 1.0 and mask_array.dtype in [np.float32, np.float64]:
                                                        # Scale 0-1 to 0-255
                                                        mask_array = (mask_array * 255).astype(np.uint8)
                                                        logger.info(f"Scaled artery mask to 0-255")
                                                    elif mask_array.max() > 1.0:
                                                        # Already in 0-255 range
                                                        mask_array = mask_array.astype(np.uint8)
                                                    else:
                                                        # Integer 0-1 values, scale them
                                                        mask_array = (mask_array * 255).astype(np.uint8)
                                                    
                                                    # Apply binary contrast (0 or 255 only)
                                                    mask_array = np.where(mask_array > 0, 255, 0).astype(np.uint8)
                                                    logger.info(f"Final artery mask range: {mask_array.min()} - {mask_array.max()}")
                                                    
                                                    mask_img = Image.fromarray(mask_array, mode='L').convert('RGB')
                                                else:
                                                    # Handle PIL Image input
                                                    mask_array = np.array(mask_input)
                                                    if len(mask_array.shape) > 2:
                                                        mask_array = mask_array[:,:,0]  # Take first channel
                                                    
                                                    # Apply same scaling logic
                                                    if mask_array.max() <= 1.0 and mask_array.dtype in [np.float32, np.float64]:
                                                        mask_array = (mask_array * 255).astype(np.uint8)
                                                    elif mask_array.max() > 1.0:
                                                        mask_array = mask_array.astype(np.uint8)
                                                    else:
                                                        mask_array = (mask_array * 255).astype(np.uint8)
                                                    
                                                    mask_array = np.where(mask_array > 0, 255, 0).astype(np.uint8)
                                                    mask_img = Image.fromarray(mask_array, mode='L').convert('RGB')
                                                
                                                # Save binary mask as input image
                                                img_copy_path = os.path.join(temp_dir, f'image_{sample_id}.jpg')
                                                mask_img.save(img_copy_path, 'JPEG')
                                                
                                                # Create classification result visualization with colored background
                                                import matplotlib.pyplot as plt
                                                from matplotlib.backends.backend_agg import FigureCanvasAgg
                                                
                                                fig, ax = plt.subplots(figsize=(6, 3))
                                                color = 'lightcoral' if class_label == 0 else 'lightblue'
                                                label_text = f"{'RIGHT' if class_label == 0 else 'LEFT'} ARTERY"
                                                
                                                ax.text(0.5, 0.5, label_text, ha='center', va='center', 
                                                       fontsize=20, fontweight='bold', color='black')
                                                ax.set_facecolor(color)
                                                ax.set_xlim(0, 1)
                                                ax.set_ylim(0, 1)
                                                ax.axis('off')
                                                
                                                # Convert to PIL Image
                                                canvas = FigureCanvasAgg(fig)
                                                canvas.draw()
                                                buf = canvas.buffer_rgba()
                                                label_array = np.asarray(buf).copy()
                                                label_array = label_array[:, :, :3]  # Remove alpha
                                                label_img = Image.fromarray(label_array)
                                                plt.close(fig)
                                                
                                                # Save classification visualization as mask
                                                mask_copy_path = os.path.join(temp_dir, f'mask_{sample_id}.png')
                                                label_img.save(mask_copy_path, 'PNG')
                                                
                                                # Create dynamic URLs using Django settings
                                                img_url = f'{settings.MEDIA_URL}temp/dataset_preview/image_{sample_id}.jpg'
                                                mask_url = f'{settings.MEDIA_URL}temp/dataset_preview/mask_{sample_id}.png'
                                                
                                                # Get statistics
                                                img_array = np.array(mask_img.convert('RGB'))
                                                label_array = np.array(label_img)
                                                
                                                # Create analysis text
                                                label_text = f"Classification: {'Right' if class_label == 0 else 'Left'} artery (class {class_label})"
                                                
                                                samples.append({
                                                    'index': i,
                                                    'filename': f'arcade_{class_name.lower()}_{i}.png',
                                                    'image_url': img_url,
                                                    'mask_url': mask_url,
                                                    'image_shape': img_array.shape,
                                                    'mask_shape': f'Classification: {class_label}',
                                                    'image_min': float(img_array.min()),
                                                    'image_max': float(img_array.max()),
                                                    'mask_min': float(label_array.min()),
                                                    'mask_max': float(label_array.max()),
                                                    'mask_coverage': 0.0,
                                                    'analysis': label_text,
                                                    'mask_type': 'Classification Label'
                                                })
                                                
                                            elif dataset_type == 'arcade_stenosis_detection':
                                                # (image, coco_annotations)
                                                img, annotations = item
                                                
                                                # Convert image
                                                if isinstance(img, torch.Tensor):
                                                    img_array = img.permute(1, 2, 0).numpy()
                                                    if img_array.max() <= 1.0:
                                                        img_array = (img_array * 255).astype(np.uint8)
                                                    img_pil = Image.fromarray(img_array)
                                                else:
                                                    img_pil = img
                                                
                                                # Save original image
                                                img_copy_path = os.path.join(temp_dir, f'image_{sample_id}.jpg')
                                                img_pil.convert('RGB').save(img_copy_path, 'JPEG')
                                                
                                                # Create bounding box visualization
                                                import matplotlib.pyplot as plt
                                                import matplotlib.patches as patches
                                                from matplotlib.backends.backend_agg import FigureCanvasAgg
                                                
                                                # Create figure with original image and bounding boxes
                                                fig, ax = plt.subplots(figsize=(8, 8))
                                                ax.imshow(np.array(img_pil), cmap='gray' if len(np.array(img_pil).shape) == 2 else None)
                                                
                                                # Draw bounding boxes if annotations exist
                                                num_boxes = 0
                                                if annotations and len(annotations) > 0:
                                                    for ann in annotations:
                                                        if 'bbox' in ann:
                                                            x, y, w, h = ann['bbox']
                                                            rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                                                                   edgecolor='red', facecolor='none')
                                                            ax.add_patch(rect)
                                                            num_boxes += 1
                                                            
                                                            # Add label if available
                                                            if 'category_id' in ann:
                                                                ax.text(x, y-5, f"Stenosis {ann['category_id']}", 
                                                                       color='red', fontsize=12, fontweight='bold',
                                                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                                                
                                                ax.set_title(f"Stenosis Detection: {num_boxes} annotations", 
                                                           fontsize=14, fontweight='bold')
                                                ax.axis('off')
                                                
                                                # Convert matplotlib figure to PIL Image
                                                canvas = FigureCanvasAgg(fig)
                                                canvas.draw()
                                                buf = canvas.buffer_rgba()
                                                mask_array = np.asarray(buf).copy()
                                                mask_array = mask_array[:, :, :3]  # Remove alpha channel
                                                ann_img = Image.fromarray(mask_array)
                                                plt.close(fig)
                                                
                                                # Save annotation visualization as mask
                                                mask_copy_path = os.path.join(temp_dir, f'mask_{sample_id}.png')
                                                ann_img.save(mask_copy_path, 'PNG')
                                                
                                                # Create dynamic URLs using Django settings
                                                img_url = f'{settings.MEDIA_URL}temp/dataset_preview/image_{sample_id}.jpg'
                                                mask_url = f'{settings.MEDIA_URL}temp/dataset_preview/mask_{sample_id}.png'
                                                
                                                # Get image stats and create detailed analysis
                                                img_array = np.array(img_pil)
                                                ann_array = np.array(ann_img)
                                                
                                                # Count annotations and analyze
                                                num_boxes = 0
                                                bbox_areas = []
                                                if annotations and len(annotations) > 0:
                                                    for ann in annotations:
                                                        if 'bbox' in ann:
                                                            x, y, w, h = ann['bbox']
                                                            bbox_areas.append(w * h)
                                                            num_boxes += 1
                                                            
                                                # Create detailed analysis
                                                if num_boxes > 0:
                                                    avg_area = np.mean(bbox_areas) if bbox_areas else 0
                                                    ann_text = f"Detection: {num_boxes} stenosis annotations, avg area: {avg_area:.1f}px²"
                                                else:
                                                    ann_text = "Detection: No stenosis annotations found"
                                                
                                                samples.append({
                                                    'index': i,
                                                    'filename': f'arcade_{class_name.lower()}_{i}.png',
                                                    'image_url': img_url,
                                                    'mask_url': mask_url,
                                                    'image_shape': img_array.shape,
                                                    'mask_shape': f'COCO annotations: {num_boxes}',
                                                    'image_min': float(img_array.min()),
                                                    'image_max': float(img_array.max()),
                                                    'mask_min': 'Bounding Boxes',
                                                    'mask_max': 'Visualization',
                                                    'mask_coverage': num_boxes,
                                                    'analysis': ann_text,
                                                    'mask_type': 'Bounding Box Visualization'
                                                })
                                                
                                            else:
                                                # Standard image-mask pairs for other types
                                                img, mask = item
                                                
                                                # Convert image tensor to PIL
                                                if isinstance(img, torch.Tensor):
                                                    img_array = img.permute(1, 2, 0).numpy() if img.dim() == 3 else img.numpy()
                                                    if img_array.max() <= 1.0:
                                                        img_array = (img_array * 255).astype(np.uint8)
                                                    if len(img_array.shape) == 3:
                                                        img_pil = Image.fromarray(img_array)
                                                    else:
                                                        img_pil = Image.fromarray(img_array, mode='L')
                                                else:
                                                    img_pil = img.convert('RGB') if hasattr(img, 'convert') else img
                                                
                                                # Convert mask tensor to PIL - Enhanced for semantic segmentation
                                                if isinstance(mask, torch.Tensor):
                                                    mask_array = mask.squeeze().numpy()
                                                    
                                                    # Handle different mask types
                                                    if dataset_type in ['arcade_semantic_segmentation', 'arcade_semantic_seg_binary']:
                                                        # Multi-class semantic mask - debug output
                                                        logger.info(f"Semantic mask shape: {mask_array.shape}, dtype: {mask_array.dtype}, min: {mask_array.min()}, max: {mask_array.max()}")
                                                        
                                                        # Handle one-hot encoded masks (H, W, C)
                                                        if len(mask_array.shape) == 3 and mask_array.shape[2] > 1:
                                                            logger.info(f"Converting one-hot mask with {mask_array.shape[2]} classes")
                                                            mask_array = np.argmax(mask_array, axis=2)
                                                        
                                                        # Ensure we have valid class indices
                                                        unique_classes = np.unique(mask_array)
                                                        logger.info(f"Unique classes in semantic mask: {unique_classes}")
                                                        
                                                        # Apply semantic colormap
                                                        mask_colored = apply_semantic_colormap(mask_array)
                                                        mask_pil = Image.fromarray(mask_colored.astype(np.uint8))
                                                        mask_type_text = 'Semantic (Multi-colored)'
                                                        
                                                        # Analysis
                                                        unique_segments = np.unique(mask_array)
                                                        num_segments = len(unique_segments[unique_segments > 0])
                                                        total_pixels = mask_array.size
                                                        foreground_pixels = np.sum(mask_array > 0)
                                                        coverage = (foreground_pixels / total_pixels) * 100
                                                        analysis_text = f'Semantic mask: {num_segments} coronary segments, {coverage:.1f}% coverage'
                                                    else:
                                                        # Binary mask - Enhanced visibility
                                                        logger.info(f"Binary mask range: {mask_array.min():.3f} - {mask_array.max():.3f}, dtype: {mask_array.dtype}")
                                                        
                                                        # Check if values are already 0-255 or need scaling
                                                        if mask_array.max() <= 1.0 and mask_array.dtype in [np.float32, np.float64]:
                                                            # Scale 0-1 values to 0-255
                                                            mask_array = (mask_array * 255).astype(np.uint8)
                                                            logger.info(f"Scaled binary mask to 0-255 range")
                                                        elif mask_array.max() > 1.0:
                                                            # Already in 0-255 range
                                                            mask_array = mask_array.astype(np.uint8)
                                                        else:
                                                            # Integer 0-1 values, scale them
                                                            mask_array = (mask_array * 255).astype(np.uint8)
                                                        
                                                        # For binary tasks, ensure full contrast (0 or 255)
                                                        if dataset_type in ['arcade_binary_segmentation', 'arcade_stenosis_segmentation']:
                                                            mask_array = np.where(mask_array > 0, 255, 0).astype(np.uint8)
                                                            logger.info(f"Applied binary contrast enhancement")
                                                        
                                                        mask_pil = Image.fromarray(mask_array, mode='L')
                                                        mask_type_text = 'Binary (Grayscale)'
                                                        
                                                        # Analysis
                                                        foreground_pixels = np.sum(mask_array > 128)
                                                        total_pixels = mask_array.size
                                                        coverage = (foreground_pixels / total_pixels) * 100
                                                        analysis_text = f'Binary mask: {coverage:.1f}% coverage, range: {mask_array.min()}-{mask_array.max()}'
                                                
                                                elif isinstance(mask, np.ndarray):
                                                    # Handle numpy arrays directly
                                                    mask_array = mask
                                                    
                                                    if dataset_type in ['arcade_semantic_segmentation', 'arcade_semantic_seg_binary']:
                                                        logger.info(f"Numpy semantic mask shape: {mask_array.shape}")
                                                        # Handle one-hot format
                                                        if len(mask_array.shape) == 3 and mask_array.shape[2] > 1:
                                                            mask_array = np.argmax(mask_array, axis=2)
                                                        
                                                        mask_colored = apply_semantic_colormap(mask_array)
                                                        mask_pil = Image.fromarray(mask_colored.astype(np.uint8))
                                                        mask_type_text = 'Semantic (Multi-colored)'
                                                        
                                                        unique_segments = np.unique(mask_array)
                                                        num_segments = len(unique_segments[unique_segments > 0])
                                                        analysis_text = f'Semantic mask: {num_segments} segments'
                                                    else:
                                                        # Binary numpy mask
                                                        if len(mask_array.shape) > 2:
                                                            mask_array = mask_array.squeeze()
                                                        
                                                        logger.info(f"Binary numpy mask range: {mask_array.min():.3f} - {mask_array.max():.3f}, dtype: {mask_array.dtype}")
                                                        
                                                        # Check if values need scaling
                                                        if mask_array.max() <= 1.0 and mask_array.dtype in [np.float32, np.float64]:
                                                            mask_array = (mask_array * 255).astype(np.uint8)
                                                            logger.info(f"Scaled numpy binary mask to 0-255")
                                                        elif mask_array.max() > 1.0:
                                                            mask_array = mask_array.astype(np.uint8)
                                                        else:
                                                            mask_array = (mask_array * 255).astype(np.uint8)
                                                        
                                                        # Apply binary contrast for binary tasks
                                                        if dataset_type in ['arcade_binary_segmentation', 'arcade_stenosis_segmentation']:
                                                            mask_array = np.where(mask_array > 0, 255, 0).astype(np.uint8)
                                                        
                                                        mask_pil = Image.fromarray(mask_array, mode='L')
                                                        mask_type_text = 'Binary (Grayscale)'
                                                        
                                                        foreground_pixels = np.sum(mask_array > 128)
                                                        total_pixels = mask_array.size
                                                        coverage = (foreground_pixels / total_pixels) * 100
                                                        analysis_text = f'Binary mask: {coverage:.1f}% coverage, range: {mask_array.min()}-{mask_array.max()}'
                                                else:
                                                    # PIL Image or other format
                                                    mask_pil = mask
                                                    mask_type_text = 'Unknown'
                                                    analysis_text = 'Mask analysis unavailable'
                                                
                                                # Save images
                                                img_copy_path = os.path.join(temp_dir, f'image_{sample_id}.jpg')
                                                mask_copy_path = os.path.join(temp_dir, f'mask_{sample_id}.png')
                                                
                                                img_pil.save(img_copy_path, 'JPEG')
                                                mask_pil.save(mask_copy_path, 'PNG')
                                                
                                                # Create dynamic URLs using Django settings
                                                img_url = f'{settings.MEDIA_URL}temp/dataset_preview/image_{sample_id}.jpg'
                                                mask_url = f'{settings.MEDIA_URL}temp/dataset_preview/mask_{sample_id}.png'
                                                
                                                # Get image and mask statistics - use final processed arrays
                                                img_array = np.array(img_pil)
                                                mask_display_array = np.array(mask_pil)
                                                
                                                # UNIVERSAL BINARY MASK FIX - ensure statistics show 0-255 range
                                                # This is the final fix to ensure mask statistics are displayed correctly
                                                logger.info(f"[FINAL FIX] Mask display array before fix: min={mask_display_array.min()}, max={mask_display_array.max()}, dtype={mask_display_array.dtype}")
                                                if len(mask_display_array.shape) == 2 and mask_display_array.max() <= 1.0:
                                                    # This mask has 0-1 range, scale to 0-255 for proper statistics display
                                                    mask_display_array = (mask_display_array * 255).astype(np.uint8)
                                                    # Apply full contrast for binary masks
                                                    mask_display_array = np.where(mask_display_array > 0, 255, 0).astype(np.uint8)
                                                    logger.info(f"[FINAL FIX] FIXED mask display range: min={mask_display_array.min()}, max={mask_display_array.max()}")
                                                
                                                # Calculate coverage
                                                if len(mask_display_array.shape) == 2:
                                                    foreground_pixels = np.sum(mask_display_array > 128)
                                                    total_pixels = mask_display_array.size
                                                    coverage = (foreground_pixels / total_pixels) * 100
                                                else:
                                                    coverage = 0.0
                                                
                                                samples.append({
                                                    'index': i,
                                                    'filename': f'arcade_{class_name.lower()}_{i}.png',
                                                    'image_url': img_url,
                                                    'mask_url': mask_url,
                                                    'image_shape': img_array.shape,
                                                    'mask_shape': mask_display_array.shape,
                                                    'image_min': float(img_array.min()),
                                                    'image_max': float(img_array.max()),
                                                    'mask_min': float(mask_display_array.min()) if len(mask_display_array.shape) <= 3 else 'N/A',
                                                    'mask_max': float(mask_display_array.max()) if len(mask_display_array.shape) <= 3 else 'N/A',
                                                    'mask_coverage': coverage,
                                                    'analysis': analysis_text,
                                                    'mask_type': mask_type_text
                                                })
                                            
                                            logger.info(f"Generated {class_name} sample {i}")
                                            
                                        except Exception as e:
                                            logger.error(f"Error processing {class_name} sample {i}: {e}")
                                            continue
                                    
                                    context['dataset_format'] = f'ARCADE {class_name}'
                                    context['total_samples'] = len(arcade_dataset)
                                    
                                else:
                                    error_message = f"ARCADE dataset structure not found at: {arcade_root}"
                    
                    except Exception as e:
                        error_message = f"Error loading ARCADE dataset: {str(e)}"
                        logger.error(f"ARCADE dataset error: {e}", exc_info=True)
                
                elif dataset_type == 'coco_style' or (detected_type == 'unknown' and dataset_type == 'auto'):
                    # Handle COCO-style datasets using ARCADE implementation
                    try:
                        images_dir = os.path.join(data_path, 'images')
                        annotations_dir = os.path.join(data_path, 'annotations')
                        
                        if os.path.exists(images_dir) and os.path.exists(annotations_dir):
                            logger.info(f"COCO-style dataset detected in: {data_path}")
                            
                            img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                            annotation_files = [f for f in os.listdir(annotations_dir) if f.lower().endswith('.json')]
                            
                            img_files.sort()
                            
                            max_samples = min(6, len(img_files))
                            
                            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp', 'dataset_preview')
                            os.makedirs(temp_dir, exist_ok=True)
                            
                            # Try to use ARCADE dataset implementation for real mask generation
                            try:
                                from ml.datasets.torch_arcade_loader import ARCADEBinarySegmentation, ARCADESemanticSegmentation, COCO_AVAILABLE
                                
                                # Check if this is semantic segmentation dataset
                                use_semantic = False
                                if dataset_info.get('structure') == 'ARCADE COCO':
                                    # Check if we have multi-class annotations
                                    try:
                                        import json
                                        ann_file = os.path.join(annotations_dir, annotation_files[0])
                                        with open(ann_file, 'r') as f:
                                            coco_data = json.load(f)
                                        
                                        # Check for coronary artery segment categories (ARCADE specific)
                                        if 'categories' in coco_data:
                                            category_names = [cat.get('name', '') for cat in coco_data['categories']]
                                            # Look for coronary artery segment names
                                            coronary_segments = [name for name in category_names if any(seg in str(name) for seg in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'])]
                                            if len(coronary_segments) > 2:
                                                use_semantic = True
                                                logger.info(f"Using semantic segmentation with color mapping for {len(coronary_segments)} coronary segments")
                                    except Exception as e:
                                        logger.warning(f"Could not parse COCO annotations for semantic detection: {e}")
                                
                                if COCO_AVAILABLE:
                                    logger.info(f"ARCADE dataset setup: use_semantic={use_semantic}, dataset_type={dataset_type}")
                                    logger.info(f"Using ARCADE{'Semantic' if use_semantic else 'Binary'}Segmentation for real mask generation")
                                    
                                    # Try to instantiate ARCADE dataset - determine image_set from path structure
                                    image_set = "train"  # default
                                    if "val" in data_path.lower():
                                        image_set = "val"
                                    elif "test" in data_path.lower():
                                        image_set = "test"
                                    
                                    # Get parent directory (should contain the full ARCADE structure)
                                    arcade_root = data_path
                                    # Try to find the root dataset directory by going up the path
                                    # Look for the parent directory that contains 'arcade_challenge_datasets'
                                    while arcade_root and not os.path.exists(os.path.join(arcade_root, "arcade_challenge_datasets")):
                                        parent = os.path.dirname(arcade_root)
                                        if parent == arcade_root:  # reached filesystem root
                                            break
                                        arcade_root = parent
                                    
                                    # If we found the arcade_challenge_datasets directory, the root should be one level up
                                    if os.path.exists(os.path.join(arcade_root, "arcade_challenge_datasets")):
                                        # Check if arcade_root already points to the parent of arcade_challenge_datasets
                                        if os.path.basename(arcade_root) == "arcade_challenge_datasets":
                                            arcade_root = os.path.dirname(arcade_root)
                                    
                                    if os.path.exists(os.path.join(arcade_root, "arcade_challenge_datasets", "dataset_phase_1")):
                                        # Full ARCADE dataset structure found
                                        logger.info(f"Full ARCADE dataset structure found at: {arcade_root}")
                                        
                                        # Choose the appropriate dataset class
                                        if use_semantic:
                                            arcade_dataset = ARCADESemanticSegmentation(
                                                root=arcade_root,
                                                image_set=image_set,
                                                download=False
                                            )
                                        else:
                                            arcade_dataset = ARCADEBinarySegmentation(
                                                root=arcade_root,
                                                image_set=image_set,
                                                download=False
                                            )
                                        
                                        # Generate samples with real masks
                                        for i in range(min(max_samples, len(arcade_dataset))):
                                            try:
                                                logger.info(f"[ARCADE] Starting sample {i} processing...")
                                                sample_id = str(uuid.uuid4())
                                                
                                                # Get image and mask from ARCADE dataset
                                                img_tensor, mask_tensor = arcade_dataset[i]
                                                
                                                # DEBUG: Check types and values at the beginning
                                                logger.info(f"[ARCADE LOOP] Sample {i}: img_tensor type={type(img_tensor)}, mask_tensor type={type(mask_tensor)}")
                                                if hasattr(mask_tensor, 'shape'):
                                                    logger.info(f"[ARCADE LOOP] mask_tensor shape={mask_tensor.shape}")
                                                if hasattr(mask_tensor, 'min') and hasattr(mask_tensor, 'max'):
                                                    logger.info(f"[ARCADE LOOP] mask_tensor range={mask_tensor.min():.3f}-{mask_tensor.max():.3f}")
                                                
                                                # Convert tensors to PIL Images for processing
                                                try:
                                                    if isinstance(img_tensor, torch.Tensor):
                                                        # Convert tensor to PIL Image
                                                        if img_tensor.dim() == 3 and img_tensor.shape[0] == 3:  # CHW format
                                                            img_array = img_tensor.permute(1, 2, 0).numpy()
                                                        elif img_tensor.dim() == 3:  # HWC format
                                                            img_array = img_tensor.numpy()
                                                        else:
                                                            img_array = img_tensor.numpy()
                                                        
                                                        # Ensure proper value range
                                                        if img_array.max() <= 1.0:
                                                            img_array = (img_array * 255).astype(np.uint8)
                                                        else:
                                                            img_array = img_array.astype(np.uint8)
                                                        
                                                        if len(img_array.shape) == 3:
                                                            img = Image.fromarray(img_array)
                                                        else:
                                                            img = Image.fromarray(img_array, mode='L')
                                                    else:
                                                        img = img_tensor  # Already PIL Image
                                                    
                                                    # Handle mask conversion
                                                    if isinstance(mask_tensor, torch.Tensor):
                                                        # Convert mask tensor - for semantic it's (H, W, 27)
                                                        mask_array = mask_tensor.numpy()
                                                        
                                                        if use_semantic:
                                                            # Convert one-hot to single channel for color mapping
                                                            if len(mask_array.shape) == 3 and mask_array.shape[2] > 1:
                                                                mask_array = np.argmax(mask_array, axis=2)
                                                            
                                                            # Apply semantic color mapping
                                                            mask_array_colored = apply_semantic_colormap(mask_array)
                                                            mask = Image.fromarray(mask_array_colored.astype(np.uint8))
                                                        else:
                                                            # For binary segmentation, use grayscale
                                                            if len(mask_array.shape) > 2:
                                                                mask_array = mask_array.squeeze()
                                                            
                                                            # Check if values are already 0-255 or 0-1
                                                            logger.info(f"[OLD SECTION] Binary mask before scaling: max={mask_array.max()}, min={mask_array.min()}, dtype={mask_array.dtype}")
                                                            if mask_array.max() <= 1.0:
                                                                mask_array = (mask_array * 255).astype(np.uint8)
                                                                logger.info(f"[OLD SECTION] Scaled binary mask: max={mask_array.max()}, min={mask_array.min()}")
                                                            else:
                                                                logger.info(f"[OLD SECTION] Binary mask already in good range")
                                                                mask_array = mask_array.astype(np.uint8)
                                                            mask = Image.fromarray(mask_array, mode='L')
                                                    
                                                    elif isinstance(mask_tensor, np.ndarray):
                                                        # Direct numpy array
                                                        mask_array = mask_tensor
                                                        
                                                        if use_semantic:
                                                            # Convert one-hot to single channel for color mapping
                                                            if len(mask_array.shape) == 3 and mask_array.shape[2] > 1:
                                                                mask_array = np.argmax(mask_array, axis=2)
                                                            
                                                            # Apply semantic color mapping
                                                            mask_array_colored = apply_semantic_colormap(mask_array)
                                                            mask = Image.fromarray(mask_array_colored.astype(np.uint8))
                                                        else:
                                                            # For binary segmentation
                                                            if len(mask_array.shape) > 2:
                                                                mask_array = mask_array.squeeze()
                                                            
                                                            logger.info(f"Binary mask ARCADE processing: shape={mask_array.shape}, dtype={mask_array.dtype}, range={mask_array.min():.3f}-{mask_array.max():.3f}")
                                                            
                                                            if mask_array.max() <= 1.0 and mask_array.dtype in [np.float32, np.float64]:
                                                                # Scale 0-1 to 0-255 and apply full contrast
                                                                mask_array = (mask_array * 255).astype(np.uint8)
                                                                mask_array = np.where(mask_array > 0, 255, 0).astype(np.uint8)
                                                                logger.info(f"Scaled binary mask from float to 0-255 with full contrast")
                                                            elif mask_array.max() > 1.0:
                                                                mask_array = mask_array.astype(np.uint8)
                                                                # Apply full contrast for any non-zero values
                                                                mask_array = np.where(mask_array > 0, 255, 0).astype(np.uint8)
                                                                logger.info(f"Applied full contrast to existing uint8 mask")
                                                            
                                                            logger.info(f"Final binary mask range: {mask_array.min()} - {mask_array.max()}")
                                                            mask = Image.fromarray(mask_array, mode='L')
                                                    else:
                                                        # Handle PIL Image - check if it needs scaling
                                                        mask_array = np.array(mask_tensor)
                                                        
                                                        if use_semantic:
                                                            # Apply semantic color mapping
                                                            if len(mask_array.shape) == 3 and mask_array.shape[2] > 1:
                                                                mask_array = np.argmax(mask_array, axis=2)
                                                            mask_array_colored = apply_semantic_colormap(mask_array)
                                                            mask = Image.fromarray(mask_array_colored.astype(np.uint8))
                                                        else:
                                                            # For binary segmentation
                                                            logger.info(f"Binary mask PIL processing: shape={mask_array.shape}, dtype={mask_array.dtype}, range={mask_array.min():.3f}-{mask_array.max():.3f}")
                                                            
                                                            if mask_array.max() <= 1.0 and mask_array.dtype in [np.float32, np.float64]:
                                                                # Scale float values 0-1 to 0-255 and apply full contrast
                                                                mask_array = (mask_array * 255).astype(np.uint8)
                                                                mask_array = np.where(mask_array > 0, 255, 0).astype(np.uint8)
                                                                mask = Image.fromarray(mask_array, mode='L')
                                                                logger.info(f"Scaled PIL mask from float to 0-255 with full contrast")
                                                            else:
                                                                mask = mask_tensor  # Already proper PIL Image
                                                
                                                except Exception as tensor_error:
                                                    logger.error(f"Tensor conversion error for sample {i}: {tensor_error}")
                                                    import traceback
                                                    logger.error(f"Full traceback: {traceback.format_exc()}")
                                                    continue
                                                
                                                # Save images for web display
                                                img_copy_path = os.path.join(temp_dir, f'image_{sample_id}.jpg')
                                                mask_copy_path = os.path.join(temp_dir, f'mask_{sample_id}.png')
                                                
                                                # Convert to RGB for JPEG saving
                                                if img.mode != 'RGB':
                                                    img = img.convert('RGB')
                                                img.save(img_copy_path, 'JPEG')
                                                
                                                # Save mask - handle both colored and grayscale masks
                                                if use_semantic and mask.mode == 'RGB':
                                                    # Save colored semantic mask as RGB PNG
                                                    mask.save(mask_copy_path, 'PNG')
                                                else:
                                                    # Save grayscale mask - ensure proper scaling
                                                    mask_array_for_save = np.array(mask)
                                                    if len(mask_array_for_save.shape) == 2 and mask_array_for_save.max() <= 1:
                                                        # Scale binary values 0,1 to 0,255 for proper PNG display
                                                        mask_array_for_save = (mask_array_for_save * 255).astype(np.uint8)
                                                        mask = Image.fromarray(mask_array_for_save, mode='L')
                                                    
                                                    if mask.mode != 'L':
                                                        mask = mask.convert('L')
                                                    mask.save(mask_copy_path, 'PNG')
                                                
                                                # Create dynamic URLs using Django settings
                                                img_url = f'{settings.MEDIA_URL}temp/dataset_preview/image_{sample_id}.jpg'
                                                mask_url = f'{settings.MEDIA_URL}temp/dataset_preview/mask_{sample_id}.png'
                                                
                                                # Get statistics
                                                img_array = np.array(img.convert('RGB'))
                                                
                                                if use_semantic:
                                                    # For semantic segmentation, analyze colored mask
                                                    mask_array_analysis = np.array(mask_tensor.squeeze().numpy() if isinstance(mask_tensor, torch.Tensor) else mask_tensor)
                                                    if len(mask_array_analysis.shape) > 2:
                                                        mask_array_analysis = mask_array_analysis[:,:,0] if len(mask_array_analysis.shape) == 3 else np.argmax(mask_array_analysis, axis=2)
                                                    
                                                    unique_segments = np.unique(mask_array_analysis)
                                                    num_segments = len(unique_segments[unique_segments > 0])  # Exclude background
                                                    total_pixels = mask_array_analysis.size
                                                    foreground_pixels = np.sum(mask_array_analysis > 0)
                                                    coverage_percent = (foreground_pixels / total_pixels) * 100
                                                    
                                                    analysis_text = f'Semantic mask: {num_segments} coronary segments, {foreground_pixels} total foreground pixels ({coverage_percent:.1f}% coverage)'
                                                else:
                                                    # For binary segmentation
                                                    mask_array = np.array(mask)
                                                    if len(mask_array.shape) == 3:
                                                        mask_array = mask_array[:,:,0]  # Take first channel for analysis
                                                    
                                                    # Debug: Check mask characteristics before scaling
                                                    logger.info(f"Binary mask before scaling: dtype={mask_array.dtype}, shape={mask_array.shape}, min={mask_array.min()}, max={mask_array.max()}")
                                                    
                                                    # Fix for binary mask visibility - ensure proper 0-255 range
                                                    if mask_array.max() <= 1.0 and mask_array.dtype in [np.float32, np.float64]:
                                                        mask_array = (mask_array * 255).astype(np.uint8)
                                                        logger.info(f"Scaled binary mask to 0-255 range: {mask_array.min()} - {mask_array.max()}")
                                                    elif mask_array.max() <= 1.0:
                                                        # Even if dtype is not float, scale 0-1 range to 0-255
                                                        mask_array = (mask_array * 255).astype(np.uint8) 
                                                        logger.info(f"Scaled binary mask (non-float) to 0-255 range: {mask_array.min()} - {mask_array.max()}")
                                                    else:
                                                        logger.info(f"Binary mask already in proper range: {mask_array.min()} - {mask_array.max()}")
                                                    
                                                    # Apply full contrast for binary masks
                                                    mask_array = np.where(mask_array > 0, 255, 0).astype(np.uint8)
                                                    logger.info(f"Applied binary contrast: {mask_array.min()} - {mask_array.max()}")
                                                    
                                                    # Calculate mask coverage for sparse masks
                                                    total_pixels = mask_array.size
                                                    foreground_pixels = np.sum(mask_array > 128)  # Consider values > 128 as foreground
                                                    coverage_percent = (foreground_pixels / total_pixels) * 100
                                                    
                                                    analysis_text = f'Binary mask: {foreground_pixels} foreground pixels ({coverage_percent:.1f}% coverage)'
                                                
                                                samples.append({
                                                    'index': i,
                                                    'filename': f'arcade_sample_{i}.png',
                                                    'image_url': img_url,
                                                    'mask_url': mask_url,
                                                    'image_shape': img_array.shape,
                                                    'mask_shape': mask_array_analysis.shape if use_semantic else mask_array.shape,
                                                    'image_min': float(img_array.min()),
                                                    'image_max': float(img_array.max()),
                                                    'mask_min': float(mask_array_analysis.min()) if use_semantic else float(mask_array.min()),
                                                    'mask_max': float(mask_array_analysis.max()) if use_semantic else float(mask_array.max()),
                                                    'mask_coverage': coverage_percent,
                                                    'analysis': analysis_text,
                                                    'mask_type': 'Semantic (Multi-colored)' if use_semantic else 'Binary (Grayscale)'
                                                })
                                                
                                                logger.info(f"Generated ARCADE sample {i} with real mask")
                                                
                                            except Exception as e:
                                                logger.error(f"Error processing ARCADE sample {i}: {e}")
                                                continue
                                        
                                        context['dataset_format'] = f'ARCADE ({"Semantic" if use_semantic else "Binary"} Segmentation)'
                                        context['annotation_files'] = annotation_files
                                        
                                    else:
                                        # Fallback to basic COCO processing without ARCADE structure
                                        logger.warning("ARCADE structure not found, falling back to basic COCO display")
                                        raise Exception("ARCADE structure not available")
                                        
                                else:
                                    logger.warning("pycocotools not available, falling back to basic image display")
                                    raise Exception("pycocotools not available")
                                    
                            except Exception as arcade_error:
                                logger.warning(f"ARCADE processing failed: {arcade_error}, falling back to basic COCO display")
                                
                                # Fallback to basic COCO processing without mask generation
                                for i in range(max_samples):
                                    try:
                                        sample_id = str(uuid.uuid4())
                                        
                                        img_file = img_files[i]
                                        img_path = os.path.join(images_dir, img_file)
                                        
                                        # Load and analyze image
                                        img = Image.open(img_path)
                                        img_array = np.array(img)
                                        
                                        # Copy image to temp directory for web display
                                        img_copy_path = os.path.join(temp_dir, f'image_{sample_id}.jpg')
                                        
                                        # Save image copy
                                        if img.mode != 'RGB':
                                            img = img.convert('RGB')
                                        img.save(img_copy_path, 'JPEG')
                                        
                                        # Create dynamic URL using Django settings
                                        img_url = f'{settings.MEDIA_URL}temp/dataset_preview/image_{sample_id}.jpg'
                                        
                                        samples.append({
                                            'index': i,
                                            'filename': img_file,
                                            'image_url': img_url,
                                            'mask_url': None,  # No mask for basic COCO preview
                                            'image_shape': img_array.shape,
                                            'mask_shape': 'COCO annotations (no parsing)',
                                            'image_min': float(img_array.min()) if hasattr(img_array, 'min') else 'N/A',
                                            'image_max': float(img_array.max()) if hasattr(img_array, 'max') else 'N/A',
                                            'mask_min': 'COCO format',
                                            'mask_max': 'COCO format',
                                        })
                                    except Exception as e:
                                        logger.error(f"Error processing COCO sample {i}: {e}")
                                        continue
                                
                                context['dataset_format'] = 'COCO (Basic)'
                                context['annotation_files'] = annotation_files
                        
                        else:
                            error_message = f"COCO-style directories not found. Looking for: {images_dir}, {annotations_dir}"
                    
                    except Exception as e:
                        error_message = f"Error loading COCO dataset: {str(e)}"
                        logger.error(f"COCO dataset error: {e}", exc_info=True)
                
                elif detected_type == 'unknown' or dataset_type == 'auto':
                    # Try to manually explore directory structure for unknown datasets
                    try:
                        logger.info(f"Exploring unknown dataset structure in: {data_path}")
                        
                        # Check for COCO-style dataset (images/ + annotations/)
                        images_dir = os.path.join(data_path, 'images')
                        annotations_dir = os.path.join(data_path, 'annotations')
                        
                        if os.path.exists(images_dir) and os.path.exists(annotations_dir):
                            # COCO-style dataset detected
                            logger.info(f"COCO-style dataset detected in: {data_path}")
                            
                            img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                            annotation_files = [f for f in os.listdir(annotations_dir) if f.lower().endswith('.json')]
                            
                            img_files.sort()
                            
                            max_samples = min(6, len(img_files))
                            
                            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp', 'dataset_preview')
                            os.makedirs(temp_dir, exist_ok=True)
                            
                            # For COCO datasets, we'll show images without masks for now
                            # since parsing COCO annotations is complex
                            for i in range(max_samples):
                                try:
                                    sample_id = str(uuid.uuid4())
                                    
                                    img_file = img_files[i]
                                    img_path = os.path.join(images_dir, img_file)
                                    
                                    # Load and analyze image
                                    img = Image.open(img_path)
                                    img_array = np.array(img)
                                    
                                    # Copy image to temp directory for web display
                                    img_copy_path = os.path.join(temp_dir, f'image_{sample_id}.jpg')
                                    
                                    # Save image copy
                                    if img.mode != 'RGB':
                                        img = img.convert('RGB')
                                    img.save(img_copy_path, 'JPEG')
                                    
                                    # Create dynamic URL using Django settings
                                    img_url = f'{settings.MEDIA_URL}temp/dataset_preview/image_{sample_id}.jpg'
                                    
                                    samples.append({
                                        'index': i,
                                        'filename': img_file,
                                        'image_url': img_url,
                                        'mask_url': None,  # No mask for COCO preview yet
                                        'image_shape': img_array.shape,
                                        'mask_shape': 'COCO annotations',
                                        'image_min': float(img_array.min()) if hasattr(img_array, 'min') else 'N/A',
                                        'image_max': float(img_array.max()) if hasattr(img_array, 'max') else 'N/A',
                                        'mask_min': 'COCO format',
                                        'mask_max': 'COCO format',
                                    })
                                except Exception as e:
                                    logger.error(f"Error processing COCO sample {i}: {e}")

                                    continue
                            
                            context['dataset_format'] = 'COCO'
                            context['annotation_files'] = annotation_files
                            
                        else:
                            # Try to find any image directories
                            possible_dirs = []
                            for root, dirs, files in os.walk(data_path):
                                for d in dirs:
                                    dir_path = os.path.join(root, d)
                                    try:
                                        image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                                        if len(image_files) > 0:
                                            possible_dirs.append({
                                                'path': dir_path,
                                                'relative': os.path.relpath(dir_path, data_path),
                                                'count': len(image_files)
                                            })
                                    except:
                                        continue
                            
                            if possible_dirs:
                                context['possible_dirs'] = possible_dirs
                                error_message = f"Dataset structure is unclear. Found image directories: {[d['relative'] for d in possible_dirs[:5]]}. Please specify the exact paths to images and masks directories."
                            else:
                                error_message = f"No image files found in dataset directory: {data_path}"
                    
                    except Exception as e:
                        error_message = f"Error exploring dataset: {str(e)}"
                        logger.error(f"Dataset exploration error: {e}", exc_info=True)
                
                else:
                    error_message = f"Unsupported dataset type: {detected_type}. Currently supporting: coronary_standard, monai_style, or use auto-detect for unknown structures."
                
                context['samples'] = samples
                
            except Exception as e:
                error_message = f"Error analyzing dataset: {str(e)}"
                logger.error(f"Dataset analysis error: {e}", exc_info=True)
    
    # Always add error_message and samples to context (can be None or empty)
    context['error_message'] = error_message
    context['samples'] = samples
    
    return render(request, 'ml_manager/dataset_preview.html', context)

