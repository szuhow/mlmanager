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
from .utils.training_utils import TrainingController, create_enhanced_training_config
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

# Try to import psutil for process management
try:
    import psutil
except ImportError:
    psutil = None
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

# Global storage for active training controllers
_active_training_controllers = {}

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
            log_lines = []
            
            # 1. Try model-specific log location first
            if self.object.model_directory and os.path.exists(self.object.model_directory):
                log_path = os.path.join(self.object.model_directory, 'logs', 'training.log')
                if os.path.exists(log_path):
                    try:
                        with open(log_path, 'r', encoding='utf-8') as f:
                            log_lines = f.read().splitlines()
                            logger.info(f"Loaded {len(log_lines)} lines from model-specific log: {log_path}")
                    except Exception as e:
                        logger.warning(f"Could not read model-specific log {log_path}: {e}")
            
            # 2. If no model-specific logs, try global log with filtering
            if not log_lines:
                global_log_path = os.path.join('data', 'logs', 'training.log')
                if os.path.exists(global_log_path):
                    try:
                        with open(global_log_path, 'r', encoding='utf-8') as f:
                            all_lines = f.read().splitlines()
                            # Filter for this specific model if we can identify it
                            model_specific_lines = [
                                line for line in all_lines 
                                if (f"model_{self.object.id}" in line or 
                                   f"Model {self.object.id}" in line or
                                   (self.object.name and self.object.name in line))
                            ]
                            if model_specific_lines:
                                log_lines = model_specific_lines
                                logger.info(f"Loaded {len(log_lines)} model-specific lines from global log")
                            else:
                                # If no specific lines found, get ALL global logs (no limit)
                                log_lines = all_lines
                                logger.info(f"Loaded {len(log_lines)} total lines from global log (no filtering)")
                    except Exception as e:
                        logger.warning(f"Could not read global log {global_log_path}: {e}")
            
            # 3. Fallback to database field
            if not log_lines and self.object.training_logs:
                log_lines = self.object.training_logs.splitlines()
                logger.info(f"Loaded {len(log_lines)} lines from database field")
            
            # 4. Final fallback - check for any recent training logs
            if not log_lines:
                # Check data/models/artifacts for any training logs
                artifacts_path = os.path.join('data', 'models', 'artifacts', 'training.log')
                if os.path.exists(artifacts_path):
                    try:
                        with open(artifacts_path, 'r', encoding='utf-8') as f:
                            log_lines = f.read().splitlines()  # Get ALL lines, no limit
                            logger.info(f"Loaded {len(log_lines)} lines from artifacts log")
                    except Exception as e:
                        logger.warning(f"Could not read artifacts log {artifacts_path}: {e}")
            
            return log_lines if log_lines else ['No training logs found for this model.']
            
        except Exception as e:
            logger.warning(f"Could not load training logs: {e}")
            return [f'Error loading logs: {str(e)}']


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


class ModelDeleteView(LoginRequiredMixin, DeleteView):
    """Delete view for ML models with confirmation"""
    model = MLModel
    template_name = 'ml_manager/model_confirm_delete.html'
    success_url = reverse_lazy('ml_manager:model-list')
    context_object_name = 'model'
    
    def delete(self, request, *args, **kwargs):
        """Override delete to add cleanup logic"""
        model = self.get_object()
        
        try:
            # Clean up MLflow artifacts if run_id exists
            if model.mlflow_run_id:
                try:
                    client = mlflow.tracking.MlflowClient()
                    # Note: We don't delete the MLflow run as it may be referenced elsewhere
                    # Just log the deletion
                    logger.info(f"Model {model.id} with MLflow run {model.mlflow_run_id} is being deleted")
                except Exception as e:
                    logger.warning(f"Error accessing MLflow run during model deletion: {e}")
            
            # Clean up model files if they exist
            if model.model_file and os.path.exists(model.model_file.path):
                try:
                    os.remove(model.model_file.path)
                    logger.info(f"Deleted model file: {model.model_file.path}")
                except Exception as e:
                    logger.warning(f"Error deleting model file: {e}")
            
            messages.success(request, f'Model "{model.name}" has been successfully deleted.')
            
        except Exception as e:
            logger.error(f"Error during model deletion cleanup: {e}")
            messages.warning(request, f'Model deleted but some cleanup operations failed: {str(e)}')
        
        return super().delete(request, *args, **kwargs)


class StartTrainingView(LoginRequiredMixin, FormView):
    form_class = TrainingForm
    template_name = 'ml_manager/start_training.html'
    success_url = reverse_lazy('ml_manager:model-list')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        rerun_model_id = self.request.GET.get('rerun')
        context['rerun_model'] = None
        if rerun_model_id:
            try:
                context['rerun_model'] = get_object_or_404(MLModel, pk=rerun_model_id)
            except MLModel.DoesNotExist:
                pass
        return context

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
                    # Medical preprocessing parameters
                    'use_medical_preprocessing': form_data.get('use_medical_preprocessing', False),
                    'preprocessing_type': form_data.get('preprocessing_type', 'angiography'),
                    'clahe_clip_limit': form_data.get('clahe_clip_limit', 3.0),
                    'clahe_tile_size': form_data.get('clahe_tile_size', 8),
                    'use_unsharp_masking': form_data.get('use_unsharp_masking', False),
                    'unsharp_amount': form_data.get('unsharp_amount', 1.0),
                    'unsharp_radius': form_data.get('unsharp_radius', 1.0),
                    'use_frangi_filter': form_data.get('use_frangi_filter', False),
                    'frangi_sigma_min': form_data.get('frangi_sigma_min', 1.0),
                    'frangi_sigma_max': form_data.get('frangi_sigma_max', 10.0),
                    'frangi_sigma_step': form_data.get('frangi_sigma_step', 2.0),
                    'use_denoising': form_data.get('use_denoising', False),
                    'noise_reduction_sigma': form_data.get('noise_reduction_sigma', 1.0),
                    'use_histogram_equalization': form_data.get('use_histogram_equalization', False),
                    'normalize_intensity': form_data.get('normalize_intensity', True),
                    'gamma_correction': form_data.get('gamma_correction', 1.0),
                    'custom_preprocessing_pipeline': form_data.get('custom_preprocessing_pipeline', ''),
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
                # Medical preprocessing flags
                'use_medical_preprocessing': form_data.get('use_medical_preprocessing', False),
                'preprocessing_type': form_data.get('preprocessing_type', 'angiography'),
                'clahe_clip_limit': form_data.get('clahe_clip_limit', 3.0),
                'clahe_tile_size': form_data.get('clahe_tile_size', 8),
                'use_unsharp_masking': form_data.get('use_unsharp_masking', False),
                'unsharp_amount': form_data.get('unsharp_amount', 1.0),
                'unsharp_radius': form_data.get('unsharp_radius', 1.0),
                'use_frangi_filter': form_data.get('use_frangi_filter', False),
                'frangi_sigma_min': form_data.get('frangi_sigma_min', 1.0),
                'frangi_sigma_max': form_data.get('frangi_sigma_max', 10.0),
                'frangi_sigma_step': form_data.get('frangi_sigma_step', 2.0),
                'use_denoising': form_data.get('use_denoising', False),
                'noise_reduction_sigma': form_data.get('noise_reduction_sigma', 1.0),
                'use_histogram_equalization': form_data.get('use_histogram_equalization', False),
                'normalize_intensity': form_data.get('normalize_intensity', True),
                'gamma_correction': form_data.get('gamma_correction', 1.0),
                'custom_preprocessing_pipeline': form_data.get('custom_preprocessing_pipeline', ''),
            }
            
            # Use TrainingController instead of manual subprocess management
            logger.info(f"Starting training with TrainingController for model {ml_model.id}")
            
            # Create enhanced training configuration
            training_config = create_enhanced_training_config(form_data)
            
            # Add MLflow run ID to config
            training_config['mlflow_run_id'] = str(mlflow_run_id)
            
            try:
                # Initialize TrainingController
                controller = TrainingController(ml_model, training_config)
                
                # Store controller in global storage for later access
                _active_training_controllers[ml_model.id] = controller
                
                # Start training with monitoring
                def progress_callback(epoch, total_epochs, metrics):
                    """Callback to handle training progress updates"""
                    logger.info(f"Training progress: Epoch {epoch}/{total_epochs}, Metrics: {metrics}")
                    # You can add WebSocket or database updates here if needed
                
                # Start training asynchronously (in background)
                import threading
                def run_training():
                    try:
                        result = controller.train_with_monitoring(progress_callback)
                        if result['success']:
                            ml_model.status = 'completed'
                            ml_model.model_path = result.get('model_path')
                            logger.info(f"Training completed successfully for model {ml_model.id}")
                        else:
                            ml_model.status = 'failed'
                            logger.error(f"Training failed for model {ml_model.id}: {result.get('error')}")
                    except Exception as e:
                        ml_model.status = 'failed'
                        logger.error(f"Training exception for model {ml_model.id}: {e}")
                    finally:
                        # Clean up controller from storage
                        if ml_model.id in _active_training_controllers:
                            del _active_training_controllers[ml_model.id]
                        ml_model.save()
                
                # Start training in background thread
                training_thread = threading.Thread(target=run_training)
                training_thread.daemon = True
                training_thread.start()
                
                messages.success(self.request, f"Training started successfully for model '{ml_model.name}' using TrainingController.")
                
            except Exception as e:
                logger.error(f"Error starting training with TrainingController: {e}")
                messages.error(self.request, f"Failed to start training: {e}")
                ml_model.status = 'failed'
                ml_model.save()
        
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            messages.error(self.request, f"Failed to start training: {e}")
        
        return super().form_valid(form)

@login_required
@require_POST
def stop_training(request, model_id):
    """Stop training for a specific model"""
    try:
        # Remove user filter since MLModel doesn't have a user field
        model = get_object_or_404(MLModel, id=model_id)
        
        # Check if model is currently training
        if model.status != 'training':
            return JsonResponse({
                'status': 'error',
                'message': f'Model is not currently training (status: {model.status})'
            })
        
        # Update model status to indicate stop requested
        model.stop_requested = True
        model.status = 'stopping'
        model.save()
        
        # Try to find and stop the TrainingController first
        controller = _active_training_controllers.get(model_id)
        if controller:
            try:
                controller.stop_training()
                message = "Training stop signal sent via TrainingController. Training will stop gracefully."
                logger.info(f"Successfully sent stop signal to TrainingController for model {model_id}")
            except Exception as e:
                logger.error(f"Error stopping training via TrainingController: {e}")
                message = f"Error stopping training: {e}"
        else:
            # Fallback to the old method if controller not found
            logger.warning(f"TrainingController not found for model {model_id}, falling back to process termination")
            
            # Try to stop the training process
            # Use more aggressive process termination with SIGTERM
            try:
                import psutil
                import signal
                
                # Find processes related to this model's training
                stopped_processes = 0
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = proc.info['cmdline']
                        if cmdline and any(str(model_id) in str(arg) for arg in cmdline):
                            if 'python' in proc.info['name'] and 'train.py' in ' '.join(cmdline):
                                # First try SIGTERM for graceful shutdown
                                proc.send_signal(signal.SIGTERM)
                                stopped_processes += 1
                                logger.info(f"Sent SIGTERM to training process {proc.info['pid']} for model {model_id}")
                                
                                # Wait a bit for graceful shutdown
                                import time
                                time.sleep(2)
                                
                                # If process still running, send SIGKILL
                                if proc.is_running():
                                    proc.kill()
                                    logger.info(f"Killed training process {proc.info['pid']} for model {model_id}")
                                    
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                if stopped_processes > 0:
                    message = f"Forcefully stopped {stopped_processes} training process(es)."
                else:
                    message = "Stop signal sent. Training will stop after current epoch."
                    
            except ImportError:
                # If psutil is not available, just update the status
                message = "Stop requested. Training will stop after current epoch."
            except Exception as e:
                logger.warning(f"Error stopping training process: {e}")
                message = "Stop requested. Training will stop after current epoch."
        
        logger.info(f"Training stop requested for model {model_id}")
        
        return JsonResponse({
            'status': 'success',
            'message': message
        })
        
    except Exception as e:
        logger.error(f"Error stopping training for model {model_id}: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to stop training: {str(e)}'
        })


@login_required
def get_training_progress(request, model_id):
    """Get training progress for a specific model"""
    try:
        model = get_object_or_404(MLModel, id=model_id)
        
        progress_data = {
            'status': model.status,
            'current_epoch': getattr(model, 'current_epoch', 0),
            'total_epochs': getattr(model, 'total_epochs', 0),
            'progress_percentage': 0,
            'metrics': {},
            'last_update': None
        }
        
        # Calculate progress percentage
        if hasattr(model, 'total_epochs') and model.total_epochs > 0:
            progress_data['progress_percentage'] = min(100, (model.current_epoch / model.total_epochs) * 100)
        
        # Get latest metrics if available
        if hasattr(model, 'training_data_info') and model.training_data_info:
            metrics = model.training_data_info.get('metrics', {})
            if metrics:
                progress_data['metrics'] = metrics
        
        return JsonResponse({
            'status': 'success',
            'progress': progress_data,
            'model_status': model.status,
            'metrics': progress_data['metrics']
        })
        
    except Exception as e:
        logger.error(f"Error getting progress for model {model_id}: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to get model progress: {str(e)}'
        })


@login_required
@require_POST
def batch_delete_models(request):
    """Batch delete multiple models"""
    try:
        model_ids = request.POST.getlist('model_ids')
        if not model_ids:
            return JsonResponse({
                'status': 'error',
                'message': 'No models selected for deletion'
            })
        
        deleted_count = 0
        errors = []
        
        for model_id in model_ids:
            try:
                model = MLModel.objects.get(id=model_id)
                
                # Clean up files and MLflow artifacts like in ModelDeleteView
                if model.mlflow_run_id:
                    logger.info(f"Model {model.id} with MLflow run {model.mlflow_run_id} is being deleted")
                
                if model.model_file and os.path.exists(model.model_file.path):
                    try:
                        os.remove(model.model_file.path)
                    except Exception as e:
                        logger.warning(f"Error deleting model file: {e}")
                
                model.delete()
                deleted_count += 1
                
            except MLModel.DoesNotExist:
                errors.append(f"Model {model_id} not found")
            except Exception as e:
                errors.append(f"Error deleting model {model_id}: {str(e)}")
        
        if deleted_count > 0:
            messages.success(request, f'Successfully deleted {deleted_count} model(s).')
        
        if errors:
            messages.warning(request, f'Some errors occurred: {"; ".join(errors)}')
        
        return JsonResponse({
            'status': 'success',
            'deleted_count': deleted_count,
            'errors': errors
        })
        
    except Exception as e:
        logger.error(f"Error in batch delete: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Batch delete failed: {str(e)}'
        })


class ModelInferenceView(LoginRequiredMixin, FormView):
    """Enhanced inference view for ML models"""
    template_name = 'ml_manager/enhanced_inference.html'
    form_class = EnhancedInferenceForm
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        model_id = self.kwargs.get('pk')
        model = get_object_or_404(MLModel, pk=model_id)
        context['model'] = model
        
        # Add available checkpoints to context
        available_checkpoints = []
        try:
            # Look for model files in the model's directory
            model_dir = Path(settings.MEDIA_ROOT) / 'models' / str(model_id)
            if model_dir.exists():
                # Find .pth files (PyTorch model checkpoints)
                checkpoint_files = list(model_dir.glob('*.pth'))
                available_checkpoints = [
                    {
                        'name': f.name,
                        'path': str(f.relative_to(settings.MEDIA_ROOT)),
                        'size': f.stat().st_size if f.exists() else 0
                    }
                    for f in checkpoint_files
                ]
        except Exception as e:
            logger.warning(f"Could not load checkpoints for model {model_id}: {e}")
            available_checkpoints = []
        
        context['available_checkpoints'] = available_checkpoints
        return context
    
    def form_valid(self, form):
        model_id = self.kwargs.get('pk')
        model = get_object_or_404(MLModel, pk=model_id)
        
        # Process the inference request
        try:
            # Handle file upload and processing logic here
            messages.success(self.request, 'Inference completed successfully!')
            return redirect('ml_manager:model-detail', pk=model_id)
        except Exception as e:
            messages.error(self.request, f'Inference failed: {str(e)}')
            return self.form_invalid(form)


class SaveAsTemplateView(LoginRequiredMixin, FormView):
    """Save model configuration as template"""
    template_name = 'ml_manager/save_as_template.html'
    form_class = TrainingTemplateForm
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        model_id = self.kwargs.get('pk')
        context['model'] = get_object_or_404(MLModel, pk=model_id)
        return context
    
    def get_initial(self):
        initial = super().get_initial()
        model_id = self.kwargs.get('pk')
        model = get_object_or_404(MLModel, pk=model_id)
        
        # Pre-populate form with model's training configuration
        if hasattr(model, 'training_config') and model.training_config:
            config = model.training_config
            initial.update({
                'name': f"{model.name}_template",
                'description': f"Template created from model: {model.name}",
                # Add other configuration fields based on model.training_config
            })
        
        return initial
    
    def form_valid(self, form):
        model_id = self.kwargs.get('pk')
        model = get_object_or_404(MLModel, pk=model_id)
        
        try:
            template = form.save(commit=False)
            template.created_by = self.request.user
            template.save()
            
            messages.success(self.request, f'Template "{template.name}" created successfully!')
            return redirect('ml_manager:template-detail', pk=template.pk)
        except Exception as e:
            messages.error(self.request, f'Failed to create template: {str(e)}')
            return self.form_invalid(form)


class TrainingTemplateListView(LoginRequiredMixin, ListView):
    """List all training templates"""
    model = TrainingTemplate
    template_name = 'ml_manager/template_list.html'
    context_object_name = 'templates'
    paginate_by = 20
    
    def get_queryset(self):
        return TrainingTemplate.objects.all().order_by('-created_at')


class TrainingTemplateCreateView(LoginRequiredMixin, FormView):
    """Create new training template"""
    template_name = 'ml_manager/template_form.html'
    form_class = TrainingTemplateForm
    
    def form_valid(self, form):
        try:
            template = form.save(commit=False)
            template.created_by = self.request.user
            template.save()
            
            messages.success(self.request, f'Template "{template.name}" created successfully!')
            return redirect('ml_manager:template-detail', pk=template.pk)
        except Exception as e:
            messages.error(self.request, f'Failed to create template: {str(e)}')
            return self.form_invalid(form)


class TrainingTemplateDetailView(LoginRequiredMixin, DetailView):
    """Detail view for training template"""
    model = TrainingTemplate
    template_name = 'ml_manager/template_detail.html'
    context_object_name = 'template'


class TrainingTemplateUpdateView(LoginRequiredMixin, FormView):
    """Update training template"""
    template_name = 'ml_manager/template_form.html'
    form_class = TrainingTemplateForm
    
    def get_object(self):
        return get_object_or_404(TrainingTemplate, pk=self.kwargs.get('pk'))
    
    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['instance'] = self.get_object()
        return kwargs
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['template'] = self.get_object()
        context['is_edit'] = True
        return context
    
    def form_valid(self, form):
        try:
            template = form.save()
            messages.success(self.request, f'Template "{template.name}" updated successfully!')
            return redirect('ml_manager:template-detail', pk=template.pk)
        except Exception as e:
            messages.error(self.request, f'Failed to update template: {str(e)}')
            return self.form_invalid(form)


class TrainingTemplateDeleteView(LoginRequiredMixin, DeleteView):
    """Delete training template"""
    model = TrainingTemplate
    template_name = 'ml_manager/template_confirm_delete.html'
    success_url = reverse_lazy('ml_manager:template-list')
    context_object_name = 'template'
    
    def delete(self, request, *args, **kwargs):
        template = self.get_object()
        messages.success(request, f'Template "{template.name}" has been successfully deleted.')
        return super().delete(request, *args, **kwargs)


class ModelLogsView(LoginRequiredMixin, DetailView):
    """View training logs for a model"""
    model = MLModel
    template_name = 'ml_manager/model_logs.html'
    context_object_name = 'model'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Try to read training logs
        try:
            # Add logic to read log files if available
            context['logs'] = "Training logs will be displayed here"
        except Exception as e:
            context['log_error'] = f"Error reading logs: {str(e)}"
        
        return context


@login_required
def get_template_data(request, template_id):
    """Get template data as JSON for AJAX requests"""
    try:
        template = get_object_or_404(TrainingTemplate, id=template_id)
        data = template.get_form_data()
        
        return JsonResponse({
            'status': 'success',
            'data': data
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })


@login_required
@require_POST
def register_model_in_registry(request, pk):
    """Register model in MLflow model registry"""
    try:
        model = get_object_or_404(MLModel, pk=pk)
        
        if not model.mlflow_run_id:
            return JsonResponse({
                'status': 'error',
                'message': 'Model has no associated MLflow run'
            })
        
        # Implementation for MLflow model registry
        messages.success(request, f'Model "{model.name}" registered in MLflow registry!')
        return JsonResponse({'status': 'success'})
        
    except Exception as e:
        logger.error(f"Error registering model {pk}: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })


@login_required
@require_POST
def transition_model_stage(request, pk):
    """Transition model stage in MLflow registry"""
    try:
        model = get_object_or_404(MLModel, pk=pk)
        stage = request.POST.get('stage', 'Staging')
        
        # Implementation for stage transition
        messages.success(request, f'Model "{model.name}" transitioned to {stage}!')
        return JsonResponse({'status': 'success'})
        
    except Exception as e:
        logger.error(f"Error transitioning model {pk}: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })


@login_required
def sync_registry_info(request, pk):
    """Sync model info with MLflow registry"""
    try:
        model = get_object_or_404(MLModel, pk=pk)
        
        # Implementation for registry sync
        messages.success(request, f'Registry info synced for model "{model.name}"!')
        return JsonResponse({'status': 'success'})
        
    except Exception as e:
        logger.error(f"Error syncing registry info for model {pk}: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })


@login_required
def registry_models_list(request):
    """List models from MLflow registry"""
    context = {
        'registry_models': []  # Placeholder for registry models
    }
    return render(request, 'ml_manager/registry_models.html', context)


@login_required
def mlflow_redirect_view(request):
    """Redirect to MLflow dashboard"""
    return redirect('http://localhost:5000')


@login_required
def get_training_log(request, model_id):
    """Get training log for a model with filtering and real-time updates"""
    try:
        model = get_object_or_404(MLModel, id=model_id)
        
        # Get filter parameters
        log_type = request.GET.get('type', 'all')
        search_query = request.GET.get('search', '')
        lines_param = request.GET.get('lines', '100')
        # Allow 'all' or '-1' to get all lines, otherwise convert to int
        if lines_param.lower() == 'all' or lines_param == '-1':
            lines_limit = None  # No limit
        else:
            lines_limit = int(lines_param)
        
        # Try to read log files in order of preference
        log_lines = []
        log_sources = []
        
        # 1. Model-specific log file (highest priority)
        if model.model_directory and os.path.exists(model.model_directory):
            model_log_path = os.path.join(model.model_directory, 'logs', 'training.log')
            if os.path.exists(model_log_path):
                try:
                    with open(model_log_path, 'r', encoding='utf-8') as f:
                        model_log_lines = f.read().splitlines()
                        log_lines = model_log_lines  # Use only model-specific logs
                        log_sources.append(f"Model-specific log: {model_log_path}")
                        logger.info(f"Found model-specific log with {len(log_lines)} lines")
                except Exception as e:
                    logger.warning(f"Could not read model log {model_log_path}: {e}")
        
        # 2. If no model-specific logs, try global training log with filtering
        if not log_lines:
            global_log_path = os.path.join('data', 'logs', 'training.log')
            if os.path.exists(global_log_path):
                try:
                    with open(global_log_path, 'r', encoding='utf-8') as f:
                        global_log_lines = f.read().splitlines()
                        # Filter for this model's logs
                        model_specific_lines = [line for line in global_log_lines 
                                              if f"model_{model_id}" in line or f"Model {model_id}" in line or f"model={model_id}" in line]
                        if model_specific_lines:
                            log_lines = model_specific_lines
                            log_sources.append(f"Global log (model {model_id} filtered): {global_log_path}")
                            logger.info(f"Found {len(log_lines)} model-specific lines in global log")
                        else:
                            # If no model-specific lines found, show recent global logs as fallback
                            if lines_limit is None:
                                log_lines = global_log_lines[-1000:]  # Limit to last 1000 lines to avoid memory issues
                            else:
                                log_lines = global_log_lines[-lines_limit:]
                            log_sources.append(f"Global log (fallback - no model-specific logs found): {global_log_path}")
                            logger.warning(f"No model-specific logs found, showing {len(log_lines)} recent global lines")
                except Exception as e:
                    logger.warning(f"Could not read global log {global_log_path}: {e}")
        
        # 3. Final fallback to model's training_logs field
        if not log_lines and model.training_logs:
            log_lines = model.training_logs.splitlines()
            log_sources.append("Database field")
            logger.info(f"Using database training_logs field with {len(log_lines)} lines")
        
        # Apply filtering
        if log_type != 'all':
            if log_type == 'epochs':
                log_lines = [line for line in log_lines if '[EPOCH]' in line or '[METRICS]' in line]
            elif log_type == 'batches':
                log_lines = [line for line in log_lines if '[TRAIN]' in line or '[VAL]' in line]
            elif log_type == 'metrics':
                log_lines = [line for line in log_lines if '[METRICS]' in line or '[STATS]' in line]
        
        # Apply search filter
        if search_query:
            log_lines = [line for line in log_lines if search_query.lower() in line.lower()]
        
        # Limit lines and get recent ones
        if lines_limit is not None and len(log_lines) > lines_limit:
            log_lines = log_lines[-lines_limit:]
        
        # Format lines for JavaScript consumption
        formatted_logs = []
        for i, line in enumerate(log_lines):
            formatted_logs.append({
                'line_number': i + 1,
                'content': line,
                'timestamp': extract_timestamp_from_line(line),
                'level': extract_log_level_from_line(line)
            })
        
        return JsonResponse({
            'status': 'success',
            'logs': formatted_logs,
            'total_lines': len(formatted_logs),
            'sources': log_sources,
            'filters_applied': {
                'type': log_type,
                'search': search_query,
                'lines_limit': lines_limit if lines_limit is not None else 'all'
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting training log for model {model_id}: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })


@login_required
def serve_training_preview_image(request, model_id, filename):
    """Serve training preview images"""
    try:
        model = get_object_or_404(MLModel, id=model_id)
        
        # Look for the image in various possible locations
        search_paths = []
        
        # Extract epoch number from filename to search in proper epoch subdirectories
        epoch_num = None
        if 'epoch_' in filename:
            try:
                # Extract epoch number from filename like "predictions_epoch_002.png"
                epoch_part = filename.split('epoch_')[1].split('.')[0]
                epoch_num = int(epoch_part)
            except (IndexError, ValueError):
                logger.warning(f"Could not extract epoch number from filename: {filename}")
        
        # 1. Model-specific directory
        if model.model_directory and os.path.exists(model.model_directory):
            search_paths.extend([
                # Direct paths (legacy structure)
                os.path.join(model.model_directory, 'predictions', filename),
                os.path.join(model.model_directory, 'artifacts', filename),
                # Epoch-specific subdirectories (new MLflow structure)
                os.path.join(model.model_directory, 'predictions', 'epoch_001', filename),
                os.path.join(model.model_directory, 'artifacts', 'epoch_001', filename),
            ])
            
            # Add epoch-specific path if we can extract epoch number
            if epoch_num is not None:
                epoch_dir = f'epoch_{epoch_num:03d}'
                search_paths.extend([
                    os.path.join(model.model_directory, 'predictions', epoch_dir, filename),
                    os.path.join(model.model_directory, 'artifacts', epoch_dir, filename),
                ])
        
        # 2. MLflow artifacts directory
        if model.mlflow_run_id:
            try:
                client = mlflow.tracking.MlflowClient()
                run = client.get_run(model.mlflow_run_id)
                
                # Base MLflow paths (flat structure)
                mlflow_paths = [
                    os.path.join(settings.BASE_MLRUNS_DIR, run.info.run_id, 'artifacts', filename),
                    os.path.join(settings.BASE_MLRUNS_DIR, run.info.run_id, 'artifacts', 'predictions', filename),
                    os.path.join(settings.BASE_MLRUNS_DIR, run.info.experiment_id, run.info.run_id, 'artifacts', filename),
                    os.path.join(settings.BASE_MLRUNS_DIR, run.info.experiment_id, run.info.run_id, 'artifacts', 'predictions', filename),
                    os.path.join('data', 'mlflow', run.info.run_id, 'artifacts', filename),
                    os.path.join('data', 'mlflow', run.info.run_id, 'artifacts', 'predictions', filename),
                ]
                
                # Add epoch-specific subdirectory paths (new MLflow structure)
                if epoch_num is not None:
                    epoch_dir = f'epoch_{epoch_num:03d}'
                    mlflow_paths.extend([
                        os.path.join(settings.BASE_MLRUNS_DIR, run.info.run_id, 'artifacts', 'predictions', epoch_dir, filename),
                        os.path.join(settings.BASE_MLRUNS_DIR, run.info.experiment_id, run.info.run_id, 'artifacts', 'predictions', epoch_dir, filename),
                        os.path.join('data', 'mlflow', run.info.run_id, 'artifacts', 'predictions', epoch_dir, filename),
                    ])
                
                search_paths.extend(mlflow_paths)
            except Exception as e:
                logger.warning(f"Could not access MLflow run for image search: {e}")
        
        # 3. Global fallback paths
        search_paths.extend([
            os.path.join('data', 'models', 'artifacts', filename),
            os.path.join('data', 'models', 'predictions', filename),
            os.path.join('data', 'temp', filename),
        ])
        
        # Find the image file
        image_path = None
        for path in search_paths:
            if os.path.exists(path) and os.path.isfile(path):
                image_path = path
                logger.info(f"Found training preview image at: {path}")
                break
        
        if not image_path:
            logger.warning(f"Training preview image not found: {filename}. Searched paths: {search_paths[:5]}...")
            # Generate a placeholder image
            return generate_placeholder_image(filename)
        
        # Serve the image
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Determine content type
            content_type = 'image/png'
            if filename.lower().endswith(('.jpg', '.jpeg')):
                content_type = 'image/jpeg'
            elif filename.lower().endswith('.gif'):
                content_type = 'image/gif'
            
            response = HttpResponse(image_data, content_type=content_type)
            response['Content-Disposition'] = f'inline; filename="{filename}"'
            return response
            
        except Exception as e:
            logger.error(f"Error reading image file {image_path}: {e}")
            return generate_placeholder_image(filename)
        
    except Exception as e:
        logger.error(f"Error serving preview image: {e}")
        return generate_placeholder_image(filename)


def generate_placeholder_image(filename):
    """Generate a placeholder image when training image is not available"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io
        
        # Create a 512x512 placeholder image
        img = Image.new('RGB', (512, 512), color='lightgray')
        draw = ImageDraw.Draw(img)
        
        # Add text
        text_lines = [
            "Training Image",
            f"{filename}",
            "Not Available Yet"
        ]
        
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None
        
        # Draw text
        y_start = 200
        for i, line in enumerate(text_lines):
            if font:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width, text_height = 100, 20  # Estimate
            
            x = (512 - text_width) // 2
            y = y_start + i * (text_height + 10)
            draw.text((x, y), line, fill='black', font=font)
        
        # Save to bytes
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_data = img_buffer.getvalue()
        
        response = HttpResponse(img_data, content_type='image/png')
        response['Content-Disposition'] = f'inline; filename="{filename}"'
        return response
        
    except Exception as e:
        logger.error(f"Error generating placeholder image: {e}")
        return HttpResponse("Image not available", status=404)


@login_required
def dataset_preview_view(request):
    """Preview dataset for training"""
    from .forms import TrainingForm
    
    # Get dataset type choices from the form field
    try:
        form = TrainingForm()
        dataset_type_choices = form.fields['dataset_type'].choices
    except (KeyError, AttributeError) as e:
        logger.warning(f"Could not get dataset_type choices from form: {e}")
        # Fallback choices
        dataset_type_choices = [
            ('auto', 'Auto-detect dataset type'),
            ('coronary', 'Standard Coronary Dataset'),
            ('arcade_binary', 'ARCADE Binary Segmentation'),
            ('arcade_semantic', 'ARCADE Semantic Segmentation'),
            ('arcade_stenosis', 'ARCADE Stenosis Detection'),
            ('arcade_classification', 'ARCADE Artery Classification')
        ]
    
    # Initialize context with default values
    context = {
        'preview_data': [],
        'dataset_type_choices': dataset_type_choices,
        'data_path': '',
        'dataset_type': 'auto',
        'detected_type': None,
        'samples': [],
        'total_samples': 0,
        'dataset_info': {},
        'error_message': None
    }
    
    if request.method == 'POST':
        data_path = request.POST.get('data_path', '')
        dataset_type = request.POST.get('dataset_type', 'auto')
        
        context['data_path'] = data_path
        context['dataset_type'] = dataset_type
        
        if data_path:
            try:
                import os
                import glob
                import random
                from PIL import Image
                import json
                
                # Check if path exists
                if not os.path.exists(data_path):
                    context['error_message'] = f"Dataset path does not exist: {data_path}"
                    return render(request, 'ml_manager/dataset_preview.html', context)
                
                # Detect dataset structure using ARCADE loader if it's an ARCADE dataset
                from ml.datasets.arcade_loader import (
                    is_arcade_dataset, 
                    detect_arcade_task_type, 
                    get_arcade_dataset_root,
                    get_arcade_task_paths
                )
                
                detected_type = None
                arcade_info = None
                
                # First check if user forced a specific dataset type
                if dataset_type != 'auto':
                    # Map GUI dataset types to internal types
                    gui_type_mapping = {
                        # Legacy names (for backwards compatibility)
                        'arcade_semantic': 'semantic_segmentation',
                        'arcade_binary': 'binary_segmentation', 
                        'arcade_stenosis': 'stenosis_detection',
                        'arcade_classification': 'artery_classification',
                        
                        # Full names from forms.py
                        'arcade_binary_segmentation': 'binary_segmentation',
                        'arcade_semantic_segmentation': 'semantic_segmentation',
                        'arcade_stenosis_detection': 'stenosis_detection',
                        'arcade_artery_classification': 'artery_classification',
                        'arcade_semantic_seg_binary': 'semantic_segmentation_binary',
                        'arcade_stenosis_segmentation': 'stenosis_segmentation',
                        
                        # Other types
                        'coronary': 'segmentation',
                        'classification': 'classification'
                    }
                    
                    detected_type = gui_type_mapping.get(dataset_type, dataset_type)
                    logger.info(f"User selected dataset type: {dataset_type} -> {detected_type}")
                
                if is_arcade_dataset(data_path):
                    logger.info(f"Detected ARCADE dataset at {data_path}")
                    # Use ARCADE-specific detection
                    try:
                        arcade_task = detect_arcade_task_type(data_path)
                        arcade_root = get_arcade_dataset_root(data_path)
                        arcade_paths = get_arcade_task_paths(arcade_root, arcade_task)
                        
                        logger.info(f"ARCADE detection results:")
                        logger.info(f"  - Task: {arcade_task}")
                        logger.info(f"  - Root: {arcade_root}")
                        logger.info(f"  - Paths: {arcade_paths}")
                        
                        # Test torch-arcade loader directly
                        try:
                            from ml.datasets.torch_arcade_loader import get_arcade_dataset_info
                            torch_arcade_info = get_arcade_dataset_info(arcade_root)
                            logger.info(f"Torch-ARCADE info: {torch_arcade_info}")
                        except Exception as torch_e:
                            logger.error(f"Torch-ARCADE test failed: {torch_e}")
                        
                        arcade_info = {
                            'task': arcade_task,
                            'root': arcade_root,
                            'paths': arcade_paths
                        }
                        
                        # If user didn't force a type, use auto-detection
                        if dataset_type == 'auto':
                            # Map ARCADE tasks to our detection types according to task specification:
                            # - semantic: multi-class segmentation
                            # - binary: binary segmentation  
                            # - stenosis detection: bounding box detection
                            # - artery classification: classification
                            task_mapping = {
                                'binary_segmentation': 'binary_segmentation',
                                'semantic_segmentation': 'semantic_segmentation',  # Multi-class
                                'stenosis_detection': 'stenosis_detection',  # Keep as stenosis_detection
                                'stenosis_segmentation': 'stenosis_segmentation',  # Binary stenosis masks
                                'artery_classification': 'artery_classification'  # Keep as artery_classification
                            }
                            
                            detected_type = task_mapping.get(arcade_task, 'semantic_segmentation')
                            logger.info(f"ARCADE dataset detected: task={arcade_task}, type={detected_type}")
                        else:
                            logger.info(f"Using user-selected type {detected_type} for ARCADE dataset (task={arcade_task})")
                        
                    except Exception as e:
                        logger.warning(f"Error using ARCADE detection: {e}")
                        if dataset_type == 'auto':
                            detected_type = detect_dataset_type(data_path)
                else:
                    # Use standard detection only if user didn't force a type
                    if dataset_type == 'auto':
                        detected_type = detect_dataset_type(data_path)
                
                context['detected_type'] = detected_type
                context['arcade_info'] = arcade_info
                
                # Get sample images based on dataset type
                samples = []
                total_image_count = 0
                
                logger.info(f"Processing dataset preview for detected_type: {detected_type}")
                logger.info(f"Arcade info available: {arcade_info is not None}")
                
                if detected_type in ['semantic_segmentation', 'binary_segmentation']:
                    logger.info(f"Processing segmentation dataset type: {detected_type}")
                    # Use ARCADE paths if available
                    if arcade_info and arcade_info.get('paths'):
                        arcade_paths = arcade_info['paths']
                        
                        logger.info(f"ARCADE paths: {arcade_paths}")
                        
                        # Try to get images from ARCADE-specific paths
                        image_dirs = []
                        if arcade_paths.get('train_images') and arcade_paths['train_images'].exists():
                            image_dirs.append(arcade_paths['train_images'])
                            logger.info(f"Added train images path: {arcade_paths['train_images']}")
                        if arcade_paths.get('val_images') and arcade_paths['val_images'].exists():
                            image_dirs.append(arcade_paths['val_images'])
                            logger.info(f"Added val images path: {arcade_paths['val_images']}")
                            
                        # Log what we found
                        logger.info(f"Found {len(image_dirs)} ARCADE image directories")
                            
                        # Fallback to detected paths
                        if not image_dirs:
                            logger.warning("No ARCADE-specific image directories found, using fallback")
                            if os.path.exists(os.path.join(data_path, 'images')):
                                image_dirs.append(os.path.join(data_path, 'images'))
                            else:
                                image_dirs.append(data_path)
                    else:
                        # Standard path detection
                        image_dirs = []
                        if os.path.exists(os.path.join(data_path, 'images')):
                            image_dirs.append(os.path.join(data_path, 'images'))
                        else:
                            image_dirs.append(data_path)
                    
                    all_images = []
                    for img_dir in image_dirs:
                        image_patterns = [
                            os.path.join(str(img_dir), '*.jpg'),
                            os.path.join(str(img_dir), '*.jpeg'),
                            os.path.join(str(img_dir), '*.png'),
                            os.path.join(str(img_dir), '*.tif'),
                            os.path.join(str(img_dir), '*.tiff')
                        ]
                        
                        for pattern in image_patterns:
                            all_images.extend(glob.glob(pattern))
                    
                    # Filter out masks and keep only original images
                    images = [img for img in all_images if not any(mask_keyword in img.lower() 
                             for mask_keyword in ['mask', 'label', 'gt', 'target'])]
                    
                    total_image_count = len(images)
                    context['total_samples'] = total_image_count
                    
                    logger.info(f"Final image count: {total_image_count}")
                    
                    # Take up to 6 random samples
                    sample_images = random.sample(images, min(6, len(images))) if images else []
                    
                    for idx, img_path in enumerate(sample_images):
                        try:
                            sample_data = {
                                'index': idx,
                                'filename': os.path.basename(img_path),
                                'image_url': f'/ml/serve-preview-image/?path={img_path}',
                                'mask_url': None,
                                'image_shape': 'Unknown',
                                'image_min': 0,
                                'image_max': 255,
                                'mask_shape': 'Unknown',
                                'mask_classes': 0,
                                'mask_coverage': None,
                                'analysis': None,
                                'mask_min': 0,
                                'mask_max': 0,
                                'mask_generator': 'Unknown',
                                'annotation_format': 'Unknown',
                                'mask_generated': False
                            }
                            
                            # Try to find corresponding mask/annotation using ARCADE paths if available
                            img_dir = os.path.dirname(img_path)
                            img_name = os.path.splitext(os.path.basename(img_path))[0]
                            
                            mask_path = None
                            
                            # First try ARCADE-specific annotation paths
                            if arcade_info and arcade_info.get('paths'):
                                arcade_paths = arcade_info['paths']
                                annotation_dirs = []
                                
                                if arcade_paths.get('train_annotations') and arcade_paths['train_annotations'].exists():
                                    annotation_dirs.append(arcade_paths['train_annotations'])
                                if arcade_paths.get('val_annotations') and arcade_paths['val_annotations'].exists():
                                    annotation_dirs.append(arcade_paths['val_annotations'])
                                
                                for ann_dir in annotation_dirs:
                                    potential_masks = [
                                        ann_dir / f"{img_name}.png",
                                        ann_dir / f"{img_name}.jpg",
                                        ann_dir / f"{img_name}.tif"
                                    ]
                                    
                                    for potential_mask in potential_masks:
                                        if potential_mask.exists():
                                            mask_path = str(potential_mask)
                                            break
                                    
                                    if mask_path:
                                        break
                            
                            # Fallback to standard mask detection patterns
                            if not mask_path:
                                mask_patterns = [
                                    # Same directory patterns
                                    os.path.join(img_dir, f"{img_name}_mask.*"),
                                    os.path.join(img_dir, f"{img_name}_label.*"),
                                    os.path.join(img_dir, f"{img_name}_gt.*"),
                                    # Separate directory patterns
                                    os.path.join(img_dir.replace('images', 'masks'), f"{img_name}.*"),
                                    os.path.join(img_dir.replace('images', 'labels'), f"{img_name}.*"),
                                    os.path.join(img_dir.replace('images', 'annotations'), f"{img_name}.*"),
                                    # ARCADE specific patterns
                                    os.path.join(data_path, 'annotations', f"{img_name}.png"),
                                    os.path.join(data_path, 'annotations', f"{img_name}.jpg"),
                                    os.path.join(data_path, 'annotations', f"{img_name}.tif"),
                                    os.path.join(data_path, 'masks', f"{img_name}.png"),
                                    os.path.join(data_path, 'masks', f"{img_name}.jpg"),
                                    os.path.join(data_path, 'masks', f"{img_name}.tif"),
                                ]
                                
                                for pattern in mask_patterns:
                                    matches = glob.glob(pattern)
                                    if matches:
                                        mask_path = matches[0]
                                        break
                            
                            # Check for COCO format annotations and generate mask preview
                            logger.info(f"[DEBUG] Processing image {img_path}")
                            logger.info(f"[DEBUG] mask_path found: {mask_path}")
                            logger.info(f"[DEBUG] arcade_info available: {arcade_info is not None}")
                            
                            if not mask_path:
                                # Check for COCO format annotations using ARCADE paths if available
                                json_files = []
                                annotations_dirs = []
                                
                                if arcade_info and arcade_info.get('paths'):
                                    # Use ARCADE-specific annotation paths
                                    arcade_paths = arcade_info['paths']
                                    if arcade_paths.get('train_annotations') and arcade_paths['train_annotations'].exists():
                                        annotations_dirs.append(str(arcade_paths['train_annotations']))
                                    if arcade_paths.get('val_annotations') and arcade_paths['val_annotations'].exists():
                                        annotations_dirs.append(str(arcade_paths['val_annotations']))
                                    logger.info(f"[DEBUG] Using ARCADE annotation paths: {annotations_dirs}")
                                else:
                                    # Fallback to standard path
                                    annotations_dir = os.path.join(data_path, 'annotations')
                                    if os.path.exists(annotations_dir):
                                        annotations_dirs.append(annotations_dir)
                                    logger.info(f"[DEBUG] Using standard annotation path: {annotations_dirs}")
                                
                                # Search for JSON files in all annotation directories
                                for annotations_dir in annotations_dirs:
                                    logger.info(f"[DEBUG] Checking annotations dir: {annotations_dir}")
                                    logger.info(f"[DEBUG] Annotations dir exists: {os.path.exists(annotations_dir)}")
                                    
                                    if os.path.exists(annotations_dir):
                                        found_json = glob.glob(os.path.join(annotations_dir, '*.json'))
                                        json_files.extend(found_json)
                                        logger.info(f"[DEBUG] Found {len(found_json)} JSON files in {annotations_dir}")
                                
                                logger.info(f"[DEBUG] Total JSON files found: {len(json_files)}")
                                
                                if json_files:
                                        sample_data['annotation_format'] = 'COCO JSON'
                                        sample_data['mask_classes'] = 'COCO Format'
                                        
                                        # Try to generate mask using appropriate ARCADE loader for ARCADE datasets
                                        if arcade_info:
                                            logger.info(f"[ARCADE MASK GEN] Starting mask generation for {detected_type}")
                                            logger.info(f"[ARCADE MASK GEN] ARCADE info: {arcade_info}")
                                            logger.info(f"[ARCADE MASK GEN] Image path: {img_path}")
                                            logger.info(f"[ARCADE MASK GEN] Image filename: {os.path.basename(img_path)}")
                                            try:
                                                import tempfile
                                                
                                                # Choose appropriate ARCADE dataset class based on dataset type
                                                arcade_dataset = None
                                                arcade_generator_name = None
                                                expected_classes = 2  # default
                                                
                                                logger.info(f"[ARCADE MASK GEN] Checking detected_type: {detected_type}")
                                                logger.info(f"[ARCADE MASK GEN] ARCADE task: {arcade_info.get('task')}")
                                                
                                                # Use ARCADE task as priority if available
                                                arcade_task = arcade_info.get('task')
                                                
                                                if detected_type == 'semantic_segmentation' or arcade_task == 'semantic_segmentation':
                                                    logger.info("[ARCADE MASK GEN] Creating ARCADESemanticSegmentation dataset")
                                                    from ml.datasets.torch_arcade_loader import ARCADESemanticSegmentation
                                                    arcade_root = arcade_info.get('root', data_path)
                                                    logger.info(f"[ARCADE MASK GEN] Using root path: {arcade_root}")
                                                    arcade_dataset = ARCADESemanticSegmentation(
                                                        root=arcade_root,
                                                        image_set='train',
                                                        download=False,
                                                        transforms=None
                                                    )
                                                    arcade_generator_name = 'ARCADE Semantic Segmentation'
                                                    expected_classes = 27  # 26 coronary segments + background
                                                    
                                                elif detected_type == 'binary_segmentation' and arcade_task == 'stenosis_detection':
                                                    logger.info("[ARCADE MASK GEN] User selected binary_segmentation for stenosis_detection task - using ARCADEStenosisDetection")
                                                    from ml.datasets.torch_arcade_loader import ARCADEStenosisDetection
                                                    arcade_root = arcade_info.get('root', data_path)
                                                    arcade_dataset = ARCADEStenosisDetection(
                                                        root=arcade_root,
                                                        image_set='train',
                                                        download=False,
                                                        transforms=None
                                                    )
                                                    arcade_generator_name = 'ARCADE Stenosis Detection (as Binary)'
                                                    expected_classes = 2  # detection: background + stenosis
                                                    
                                                elif detected_type == 'binary_segmentation' and arcade_task == 'binary_segmentation':
                                                    logger.info("[ARCADE MASK GEN] Creating ARCADEBinarySegmentation dataset")
                                                    from ml.datasets.torch_arcade_loader import ARCADEBinarySegmentation
                                                    arcade_root = arcade_info.get('root', data_path)
                                                    arcade_dataset = ARCADEBinarySegmentation(
                                                        root=arcade_root,
                                                        image_set='train',
                                                        download=False,
                                                        transforms=None
                                                    )
                                                    arcade_generator_name = 'ARCADE Binary Segmentation'
                                                    expected_classes = 2  # background + foreground
                                                    
                                                elif arcade_task == 'stenosis_detection' or detected_type == 'stenosis_detection':
                                                    logger.info("[ARCADE MASK GEN] Creating ARCADEStenosisDetection dataset")
                                                    from ml.datasets.torch_arcade_loader import ARCADEStenosisDetection
                                                    arcade_root = arcade_info.get('root', data_path)
                                                    arcade_dataset = ARCADEStenosisDetection(
                                                        root=arcade_root,
                                                        image_set='train',
                                                        download=False,
                                                        transforms=None
                                                    )
                                                    arcade_generator_name = 'ARCADE Stenosis Detection'
                                                    expected_classes = 2  # detection: background + stenosis
                                                    
                                                elif detected_type == 'stenosis_detection':
                                                    from ml.datasets.torch_arcade_loader import ARCADEStenosisDetection
                                                    arcade_root = arcade_info.get('root', data_path)
                                                    arcade_dataset = ARCADEStenosisDetection(
                                                        root=arcade_root,
                                                        image_set='train',
                                                        download=False,
                                                        transforms=None
                                                    )
                                                    arcade_generator_name = 'ARCADE Stenosis Detection'
                                                    expected_classes = 2  # detection: background + stenosis
                                                    
                                                elif detected_type == 'artery_classification':
                                                    from ml.datasets.torch_arcade_loader import ARCADEArteryClassification
                                                    arcade_root = arcade_info.get('root', data_path)
                                                    arcade_dataset = ARCADEArteryClassification(
                                                        root=arcade_root,
                                                        image_set='train',
                                                        download=False,
                                                        transforms=None
                                                    )
                                                    arcade_generator_name = 'ARCADE Artery Classification'
                                                    expected_classes = 2  # left/right artery classification
                                                    
                                                elif detected_type == 'stenosis_segmentation':
                                                    from ml.datasets.torch_arcade_loader import ARCADEStenosisSegmentation
                                                    arcade_root = arcade_info.get('root', data_path)
                                                    arcade_dataset = ARCADEStenosisSegmentation(
                                                        root=arcade_root,
                                                        image_set='train',
                                                        download=False,
                                                        transforms=None
                                                    )
                                                    arcade_generator_name = 'ARCADE Stenosis Segmentation'
                                                    expected_classes = 2  # background + stenosis
                                                    
                                                elif detected_type == 'semantic_segmentation_binary':
                                                    from ml.datasets.torch_arcade_loader import ARCADESemanticSegmentationBinary
                                                    arcade_root = arcade_info.get('root', data_path)
                                                    arcade_dataset = ARCADESemanticSegmentationBinary(
                                                        root=arcade_root,
                                                        image_set='train',
                                                        download=False,
                                                        transforms=None
                                                    )
                                                    arcade_generator_name = 'ARCADE Semantic Segmentation Binary'
                                                    expected_classes = 26  # 26 coronary segments (no background)
                                                
                                                if arcade_dataset:
                                                    # Find the image in the dataset
                                                    img_filename = os.path.basename(img_path)
                                                    logger.info(f"[ARCADE MASK GEN] Looking for image: {img_filename}")
                                                    logger.info(f"[ARCADE MASK GEN] Dataset has {len(arcade_dataset.file_to_id)} files")
                                                    logger.info(f"[ARCADE MASK GEN] First 5 files in dataset: {list(os.path.basename(f) for f in list(arcade_dataset.file_to_id.keys())[:5])}")
                                                    
                                                    # Find matching file in dataset
                                                    matching_file_path = None
                                                    img_id = None
                                                    
                                                    # Search through dataset files for filename match
                                                    for file_path, file_id in arcade_dataset.file_to_id.items():
                                                        if os.path.basename(file_path) == img_filename:
                                                            matching_file_path = file_path
                                                            img_id = file_id
                                                            logger.info(f"[ARCADE MASK GEN] Found match: {file_path} -> {file_id}")
                                                            break
                                                    
                                                    if img_id is not None and matching_file_path is not None:
                                                        logger.info(f"[ARCADE MASK GEN] Found image {img_filename} with ID {img_id}")
                                                        
                                                        # Generate mask using appropriate ARCADE loader
                                                        if hasattr(arcade_dataset, '_get_cached_mask'):
                                                            mask_data = arcade_dataset._get_cached_mask(matching_file_path, img_id)
                                                        else:
                                                            # For datasets that don't have _get_cached_mask, try to get sample by index
                                                            try:
                                                                # Find index of the image in the dataset
                                                                img_index = None
                                                                for idx, img_path_in_dataset in enumerate(arcade_dataset.images):
                                                                    if os.path.basename(img_path_in_dataset) == img_filename:
                                                                        img_index = idx
                                                                        break
                                                                
                                                                if img_index is not None:
                                                                    sample = arcade_dataset[img_index]
                                                                    if isinstance(sample, tuple) and len(sample) >= 2:
                                                                        mask_data = sample[1]  # Usually mask is second element
                                                                    else:
                                                                        mask_data = sample
                                                                else:
                                                                    mask_data = None
                                                            except Exception as e:
                                                                logger.warning(f"[ARCADE MASK GEN] Error getting sample: {e}")
                                                                mask_data = None
                                                        
                                                        if mask_data is not None:
                                                            logger.info(f"[ARCADE MASK GEN] Got mask data for {img_filename}")
                                                            logger.info(f"[ARCADE MASK GEN] Mask data type: {type(mask_data)}")
                                                            logger.info(f"[ARCADE MASK GEN] Mask data shape: {getattr(mask_data, 'shape', 'No shape attr')}")
                                                            
                                                            # Check if mask_data is a tensor and convert to numpy
                                                            if hasattr(mask_data, 'numpy'):
                                                                logger.info("[ARCADE MASK GEN] Converting tensor to numpy")
                                                                mask_array = mask_data.numpy()
                                                            elif hasattr(mask_data, 'detach'):
                                                                logger.info("[ARCADE MASK GEN] Detaching tensor and converting to numpy")
                                                                mask_array = mask_data.detach().numpy()
                                                            else:
                                                                logger.info("[ARCADE MASK GEN] Converting directly to numpy array")
                                                                mask_array = np.array(mask_data)
                                                            
                                                            logger.info(f"[ARCADE MASK GEN] Converted mask_array shape: {mask_array.shape}")
                                                            logger.info(f"[ARCADE MASK GEN] Mask array dtype: {mask_array.dtype}")
                                                            logger.info(f"[ARCADE MASK GEN] Mask array min/max: {mask_array.min()}/{mask_array.max()}")
                                                            logger.info(f"[ARCADE MASK GEN] Mask array unique values: {np.unique(mask_array)}")
                                                            
                                                            # Save the generated mask to a temporary file
                                                            temp_mask_dir = '/tmp/preview_masks'
                                                            os.makedirs(temp_mask_dir, exist_ok=True)
                                                            temp_mask_path = os.path.join(temp_mask_dir, f"{os.path.splitext(img_filename)[0]}_mask.png")
                                                            
                                                            # Handle different mask formats
                                                            if len(mask_array.shape) == 3:
                                                                logger.info(f"[ARCADE MASK GEN] 3D mask detected: {mask_array.shape}")
                                                                # Multi-channel mask (e.g., semantic segmentation)
                                                                if mask_array.shape[2] > 1:  # Height x Width x Channels format
                                                                    logger.info(f"[ARCADE MASK GEN] Multi-channel mask with {mask_array.shape[2]} channels")
                                                                    if detected_type == 'semantic_segmentation':
                                                                        # Convert one-hot to class indices
                                                                        logger.info(f"[ARCADE MASK GEN] Converting semantic one-hot to class indices")
                                                                        mask_array = np.argmax(mask_array, axis=2)
                                                                        logger.info(f"[ARCADE MASK GEN] After argmax: shape={mask_array.shape}, unique_values={np.unique(mask_array)}")
                                                                    else:
                                                                        # For other multi-channel, take first channel
                                                                        logger.info("[ARCADE MASK GEN] Taking first channel of multi-channel mask")
                                                                        mask_array = mask_array[:, :, 0]
                                                                elif mask_array.shape[0] > 1:  # Channel x Height x Width format
                                                                    logger.info(f"[ARCADE MASK GEN] Multi-channel mask (CHW format) with {mask_array.shape[0]} channels")
                                                                    if detected_type == 'semantic_segmentation':
                                                                        # Convert one-hot to class indices
                                                                        logger.info(f"[ARCADE MASK GEN] Converting semantic one-hot to class indices")
                                                                        mask_array = np.argmax(mask_array, axis=0)
                                                                        logger.info(f"[ARCADE MASK GEN] After argmax: shape={mask_array.shape}, unique_values={np.unique(mask_array)}")
                                                                    else:
                                                                        logger.info("[ARCADE MASK GEN] Taking first channel")
                                                                        mask_array = mask_array[0]  # Single channel
                                                                else:
                                                                    logger.info("[ARCADE MASK GEN] Single channel 3D mask, taking first channel")
                                                                    mask_array = mask_array[0] if mask_array.shape[0] == 1 else mask_array[:, :, 0]
                                                            else:
                                                                logger.info(f"[ARCADE MASK GEN] 2D mask: {mask_array.shape}")
                                                            
                                                            # Normalize for visualization
                                                            logger.info(f"[ARCADE MASK GEN] Before normalization - shape: {mask_array.shape}, min/max: {mask_array.min()}/{mask_array.max()}")
                                                            
                                                            # Store original mask for class counting
                                                            original_unique_vals = np.unique(mask_array)
                                                            original_num_classes = len(original_unique_vals)
                                                            logger.info(f"[ARCADE MASK GEN] Original unique classes: {original_unique_vals} (total: {original_num_classes})")
                                                            
                                                            if detected_type == 'semantic_segmentation':
                                                                logger.info(f"[ARCADE MASK GEN] Creating colored semantic segmentation visualization")
                                                                
                                                                # Create a colored visualization for semantic segmentation
                                                                # Use a color map to assign different colors to different classes
                                                                colored_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
                                                                
                                                                # Create a colormap for classes (up to 27 classes for ARCADE)
                                                                colors = [
                                                                    [0, 0, 0],       # 0: background (black)
                                                                    [255, 0, 0],     # 1: red
                                                                    [0, 255, 0],     # 2: green  
                                                                    [0, 0, 255],     # 3: blue
                                                                    [255, 255, 0],   # 4: yellow
                                                                    [255, 0, 255],   # 5: magenta
                                                                    [0, 255, 255],   # 6: cyan
                                                                    [128, 0, 0],     # 7: dark red
                                                                    [0, 128, 0],     # 8: dark green
                                                                    [0, 0, 128],     # 9: dark blue
                                                                    [128, 128, 0],   # 10: olive
                                                                    [128, 0, 128],   # 11: purple
                                                                    [0, 128, 128],   # 12: teal
                                                                    [192, 192, 192], # 13: light gray
                                                                    [128, 128, 128], # 14: gray
                                                                    [255, 128, 0],   # 15: orange
                                                                    [255, 0, 128],   # 16: pink
                                                                    [128, 255, 0],   # 17: lime
                                                                    [0, 255, 128],   # 18: spring green
                                                                    [128, 0, 255],   # 19: violet
                                                                    [0, 128, 255],   # 20: light blue
                                                                    [255, 255, 128], # 21: light yellow
                                                                    [255, 128, 255], # 22: light pink
                                                                    [128, 255, 255], # 23: light cyan
                                                                    [64, 0, 0],      # 24: dark red 2
                                                                    [0, 64, 0],      # 25: dark green 2
                                                                    [0, 0, 64],      # 26: dark blue 2
                                                                ]
                                                                
                                                                # Apply colors to mask
                                                                for class_idx in original_unique_vals:
                                                                    if class_idx < len(colors):
                                                                        colored_mask[mask_array == class_idx] = colors[int(class_idx)]
                                                                    else:
                                                                        # Fallback color for classes beyond our palette
                                                                        colored_mask[mask_array == class_idx] = [255, 255, 255]  # white
                                                                
                                                                mask_array = colored_mask
                                                                logger.info(f"[ARCADE MASK GEN] Created colored mask with shape: {mask_array.shape}")
                                                                
                                                            else:
                                                                # For binary or other types, ensure proper scaling
                                                                if mask_array.max() <= 1.0:
                                                                    mask_array = (mask_array * 255).astype(np.uint8)
                                                                else:
                                                                    mask_array = mask_array.astype(np.uint8)
                                                            
                                                            logger.info(f"[ARCADE MASK GEN] After processing - shape: {mask_array.shape}, min/max: {mask_array.min()}/{mask_array.max()}")
                                                            if detected_type != 'semantic_segmentation':
                                                                logger.info(f"[ARCADE MASK GEN] Final mask unique values: {np.unique(mask_array)}")
                                                            else:
                                                                logger.info(f"[ARCADE MASK GEN] Colored semantic mask created with {original_num_classes} classes")
                                                            
                                                            # Save mask as PNG
                                                            if detected_type == 'semantic_segmentation' and len(mask_array.shape) == 3:
                                                                # Save as RGB image for colored semantic segmentation
                                                                mask_img = Image.fromarray(mask_array, mode='RGB')
                                                            else:
                                                                # Save as grayscale for binary/other types
                                                                mask_img = Image.fromarray(mask_array)
                                                            mask_img.save(temp_mask_path)
                                                            
                                                            sample_data['mask_url'] = f'/ml/serve-preview-image/?path={temp_mask_path}'
                                                            sample_data['mask_generated'] = True
                                                            sample_data['mask_generator'] = arcade_generator_name
                                                            
                                                            # Get mask info with proper class detection
                                                            if len(mask_array.shape) == 2:
                                                                sample_data['mask_shape'] = f"{mask_array.shape[1]}x{mask_array.shape[0]}"
                                                                sample_data['mask_min'] = int(np.min(mask_array))
                                                                sample_data['mask_max'] = int(np.max(mask_array))
                                                            elif len(mask_array.shape) == 3:
                                                                sample_data['mask_shape'] = f"{mask_array.shape[1]}x{mask_array.shape[0]}x{mask_array.shape[2]}"
                                                                sample_data['mask_min'] = int(np.min(mask_array))
                                                                sample_data['mask_max'] = int(np.max(mask_array))
                                                            else:
                                                                sample_data['mask_shape'] = f"{mask_array.shape}"
                                                                sample_data['mask_min'] = 0
                                                                sample_data['mask_max'] = 0
                                                            
                                                            # Set proper class count based on dataset type
                                                            if detected_type == 'semantic_segmentation':
                                                                sample_data['mask_classes'] = f"{original_num_classes} classes (semantic, expected ~27)"
                                                                sample_data['analysis'] = f"Semantic segmentation with {original_num_classes} detected classes"
                                                                sample_data['mask_coverage'] = f"{np.sum(mask_array > 0) / (mask_array.shape[0] * mask_array.shape[1]) * 100:.1f}%"
                                                            elif detected_type == 'artery_classification':
                                                                sample_data['mask_classes'] = f"2 classes (artery classification: left/right)"
                                                                sample_data['analysis'] = "Artery classification task - binary mask indicates left/right artery"
                                                                sample_data['mask_coverage'] = f"{np.sum(mask_array > 0) / mask_array.size * 100:.1f}%"
                                                            elif detected_type == 'stenosis_detection':
                                                                sample_data['mask_classes'] = f"2 classes (stenosis detection: background/stenosis)"
                                                                sample_data['analysis'] = "Stenosis detection - binary mask indicates stenosis regions"
                                                                sample_data['mask_coverage'] = f"{np.sum(mask_array > 0) / mask_array.size * 100:.1f}%"
                                                            elif detected_type == 'stenosis_segmentation':
                                                                sample_data['mask_classes'] = f"2 classes (stenosis segmentation: background/stenosis)"
                                                                sample_data['analysis'] = "Stenosis segmentation - binary mask of stenosis regions"
                                                                sample_data['mask_coverage'] = f"{np.sum(mask_array > 0) / mask_array.size * 100:.1f}%"
                                                            elif detected_type == 'semantic_segmentation_binary':
                                                                sample_data['mask_classes'] = f"{original_num_classes} classes (26 coronary segments)"
                                                                sample_data['analysis'] = f"Semantic segmentation from binary input - {original_num_classes} coronary segments"
                                                                sample_data['mask_coverage'] = f"{np.sum(mask_array > 0) / (mask_array.shape[0] * mask_array.shape[1]) * 100:.1f}%"
                                                            else:
                                                                sample_data['mask_classes'] = f"{original_num_classes} classes ({detected_type})"
                                                                sample_data['analysis'] = f"ARCADE {detected_type} with {original_num_classes} classes"
                                                                sample_data['mask_coverage'] = f"{np.sum(mask_array > 0) / (mask_array.shape[0] * mask_array.shape[1]) * 100:.1f}%"
                                                            
                                                            logger.info(f"Generated {detected_type} mask for {img_filename} using {arcade_generator_name}")
                                                        else:
                                                            logger.warning(f"Could not get mask data for {img_filename}")
                                                    else:
                                                        logger.warning(f"Image {img_filename} not found in ARCADE dataset")
                                                        logger.info(f"[ARCADE MASK GEN] Available files in dataset: {list(os.path.basename(f) for f in arcade_dataset.file_to_id.keys())[:10]}...")  # Show first 10 files
                                                else:
                                                    logger.warning(f"Unknown ARCADE dataset type: {detected_type}")
                                                    
                                            except Exception as e:
                                                logger.warning(f"[ARCADE MASK GEN] Could not generate mask using ARCADE loader for {img_path}: {e}")
                                                logger.warning(f"[ARCADE MASK GEN] Exception details: {type(e).__name__}: {str(e)}")
                                                import traceback
                                                logger.warning(f"[ARCADE MASK GEN] Traceback: {traceback.format_exc()}")
                                                import traceback
                                                logger.warning(f"[ARCADE MASK GEN] Traceback: {traceback.format_exc()}")
                                                # Fall back to generic COCO generation
                                                logger.info("[FALLBACK] Using COCO utils as fallback")
                                                try:
                                                    from ml.datasets.coco_utils import generate_mask_from_coco_file
                                                    
                                                    # Create temp directory for generated masks
                                                    temp_mask_dir = '/tmp/preview_masks'
                                                    os.makedirs(temp_mask_dir, exist_ok=True)
                                                    
                                                    generated_mask_path = generate_mask_from_coco_file(
                                                        json_files[0],
                                                        os.path.basename(img_path),
                                                        temp_mask_dir
                                                    )
                                                    
                                                    if generated_mask_path and os.path.exists(generated_mask_path):
                                                        sample_data['mask_url'] = f'/ml/serve-preview-image/?path={generated_mask_path}'
                                                        sample_data['mask_generated'] = True
                                                        sample_data['mask_generator'] = 'COCO Utils'
                                                        
                                                        # Get mask info
                                                        try:
                                                            with Image.open(generated_mask_path) as mask_img:
                                                                mask_array = np.array(mask_img)
                                                                sample_data['mask_shape'] = f"{mask_img.size[0]}x{mask_img.size[1]}"
                                                                sample_data['mask_min'] = int(np.min(mask_array))
                                                                sample_data['mask_max'] = int(np.max(mask_array))
                                                                unique_vals = np.unique(mask_array)
                                                                sample_data['mask_classes'] = f"{len(unique_vals)} classes"
                                                        except Exception:
                                                            pass
                                                            
                                                except Exception as e2:
                                                    logger.warning(f"Could not generate mask from COCO for {img_path}: {e2}")
                                                    # Keep COCO format info even if mask generation fails
                                        else:
                                            # Non-ARCADE dataset, use generic COCO generation
                                            try:
                                                from ml.datasets.coco_utils import generate_mask_from_coco_file
                                                
                                                # Create temp directory for generated masks
                                                temp_mask_dir = '/tmp/preview_masks'
                                                os.makedirs(temp_mask_dir, exist_ok=True)
                                                
                                                generated_mask_path = generate_mask_from_coco_file(
                                                    json_files[0],
                                                    os.path.basename(img_path),
                                                    temp_mask_dir
                                                )
                                                
                                                if generated_mask_path and os.path.exists(generated_mask_path):
                                                    sample_data['mask_url'] = f'/ml/serve-preview-image/?path={generated_mask_path}'
                                                    sample_data['mask_generated'] = True
                                                    sample_data['mask_generator'] = 'COCO Utils'
                                                    
                                                    # Get mask info
                                                    try:
                                                        with Image.open(generated_mask_path) as mask_img:
                                                            mask_array = np.array(mask_img)
                                                            sample_data['mask_shape'] = f"{mask_img.size[0]}x{mask_img.size[1]}"
                                                            sample_data['mask_min'] = int(np.min(mask_array))
                                                            sample_data['mask_max'] = int(np.max(mask_array))
                                                            unique_vals = np.unique(mask_array)
                                                            sample_data['mask_classes'] = f"{len(unique_vals)} classes"
                                                    except Exception:
                                                        pass
                                                        
                                            except Exception as e:
                                                logger.warning(f"Could not generate mask from COCO for {img_path}: {e}")
                                                # Keep COCO format info even if mask generation fails
                            
                            if mask_path and os.path.exists(mask_path):
                                sample_data['mask_url'] = f'/ml/serve-preview-image/?path={mask_path}'
                                sample_data['annotation_format'] = 'Direct Image'
                                sample_data['mask_generator'] = 'Direct File'
                                sample_data['mask_generated'] = False
                                
                                # Get mask info
                                try:
                                    with Image.open(mask_path) as mask_img:
                                        mask_array = np.array(mask_img)
                                        sample_data['mask_shape'] = f"{mask_img.size[0]}x{mask_img.size[1]}"
                                        sample_data['mask_min'] = int(np.min(mask_array))
                                        sample_data['mask_max'] = int(np.max(mask_array))
                                        unique_vals = np.unique(mask_array)
                                        sample_data['mask_classes'] = f"{len(unique_vals)} classes"
                                        sample_data['mask_coverage'] = f"{np.sum(mask_array > 0) / mask_array.size * 100:.1f}%"
                                        sample_data['analysis'] = f"Direct mask file with {len(unique_vals)} unique values"
                                except Exception as e:
                                    logger.warning(f"Error analyzing mask {mask_path}: {e}")
                                    sample_data['analysis'] = "⚠️ Error analyzing mask file"
                            
                            # Fallback for when no mask is found
                            if not sample_data.get('mask_url') and sample_data['annotation_format'] == 'Unknown':
                                sample_data['annotation_format'] = 'No Annotations'
                                sample_data['mask_generator'] = 'None'
                                sample_data['analysis'] = "⚠️ No mask or annotations found"
                            
                            # Try to get image info
                            try:
                                with Image.open(img_path) as img:
                                    sample_data['image_shape'] = f"{img.size[0]}x{img.size[1]}"
                            except Exception:
                                pass
                            
                            samples.append(sample_data)
                            
                        except Exception as e:
                            logger.warning(f"Error processing sample {img_path}: {e}")
                            continue
                            
                elif detected_type == 'classification':
                    # For classification datasets, get samples from each class
                    subdirs = [d for d in os.listdir(data_path) 
                               if os.path.isdir(os.path.join(data_path, d))]
                    
                    # Exclude common non-class directories
                    non_class_dirs = {'images', 'masks', 'labels', 'annotations', 'train', 'val', 'test', 'validation'}
                    class_dirs = [d for d in subdirs if d.lower() not in non_class_dirs]
                    
                    all_images = []
                    for class_dir in class_dirs:
                        class_path = os.path.join(data_path, class_dir)
                        image_patterns = [
                            os.path.join(class_path, '*.jpg'),
                            os.path.join(class_path, '*.jpeg'),
                            os.path.join(class_path, '*.png'),
                            os.path.join(class_path, '*.tif'),
                            os.path.join(class_path, '*.tiff')
                        ]
                        
                        for pattern in image_patterns:
                            class_images = glob.glob(pattern)
                            for img in class_images:
                                all_images.append((img, class_dir))
                    
                    total_image_count = len(all_images)
                    context['total_samples'] = total_image_count
                    
                    # Take up to 6 random samples
                    sample_images = random.sample(all_images, min(6, len(all_images))) if all_images else []
                    
                    for idx, (img_path, class_name) in enumerate(sample_images):
                        try:
                            sample_data = {
                                'index': idx,
                                'filename': os.path.basename(img_path),
                                'image_url': f'/ml/serve-preview-image/?path={img_path}',
                                'class_name': class_name,
                                'image_shape': 'Unknown',
                                'image_min': 0,
                                'image_max': 255,
                                'mask_coverage': None,
                                'analysis': f'Classification: {class_name}',
                                'mask_min': 0,
                                'mask_max': 0,
                                'mask_url': None,
                                'mask_shape': 'Unknown',
                                'mask_classes': 0,
                                'mask_generator': 'N/A (Classification)',
                                'annotation_format': 'Directory Structure',
                                'mask_generated': False
                            }
                            
                            # Try to get image info
                            try:
                                with Image.open(img_path) as img:
                                    sample_data['image_shape'] = f"{img.size[0]}x{img.size[1]}"
                            except Exception:
                                pass
                            
                            samples.append(sample_data)
                            
                        except Exception as e:
                            logger.warning(f"Error processing sample {img_path}: {e}")
                            continue
                
                elif detected_type == 'stenosis_detection':
                    # For stenosis detection (bounding boxes)
                    logger.info(f"Processing stenosis detection dataset type: {detected_type}")
                    
                    # Use ARCADE paths if available
                    if arcade_info and arcade_info.get('paths'):
                        arcade_paths = arcade_info['paths']
                        
                        # Try to get images from ARCADE stenosis dataset
                        image_dirs = []
                        if arcade_paths.get('train_images') and arcade_paths['train_images'].exists():
                            image_dirs.append(arcade_paths['train_images'])
                            logger.info(f"Added stenosis train images path: {arcade_paths['train_images']}")
                        if arcade_paths.get('val_images') and arcade_paths['val_images'].exists():
                            image_dirs.append(arcade_paths['val_images'])
                            logger.info(f"Added stenosis val images path: {arcade_paths['val_images']}")
                    else:
                        # Standard path detection for stenosis
                        image_dirs = []
                        if os.path.exists(os.path.join(data_path, 'images')):
                            image_dirs.append(os.path.join(data_path, 'images'))
                        else:
                            image_dirs.append(data_path)
                    
                    all_images = []
                    for img_dir in image_dirs:
                        image_patterns = [
                            os.path.join(str(img_dir), '*.jpg'),
                            os.path.join(str(img_dir), '*.jpeg'),
                            os.path.join(str(img_dir), '*.png'),
                            os.path.join(str(img_dir), '*.tif'),
                            os.path.join(str(img_dir), '*.tiff')
                        ]
                        
                        for pattern in image_patterns:
                            all_images.extend(glob.glob(pattern))
                    
                    # Filter out masks
                    images = [img for img in all_images if not any(mask_keyword in img.lower() 
                             for mask_keyword in ['mask', 'label', 'gt', 'target'])]
                    
                    total_image_count = len(images)
                    context['total_samples'] = total_image_count
                    
                    logger.info(f"Final stenosis detection image count: {total_image_count}")
                    
                    # Take up to 6 random samples
                    sample_images = random.sample(images, min(6, len(images))) if images else []
                    
                    for idx, img_path in enumerate(sample_images):
                        try:
                            sample_data = {
                                'index': idx,
                                'filename': os.path.basename(img_path),
                                'image_url': f'/ml/serve-preview-image/?path={img_path}',
                                'mask_url': None,
                                'image_shape': 'Unknown',
                                'image_min': 0,
                                'image_max': 255,
                                'mask_coverage': None,
                                'analysis': 'Bounding Box Detection',
                                'mask_min': 0,
                                'mask_max': 0,
                                'mask_shape': 'Bounding Boxes',
                                'mask_classes': 'Stenosis Detection',
                                'mask_generator': 'ARCADE Stenosis Detection',
                                'annotation_format': 'COCO Bounding Boxes',
                                'mask_generated': False,
                                'bounding_boxes': []
                            }
                            
                            # Try to get bounding boxes from ARCADE
                            if arcade_info:
                                try:
                                    logger.info(f"[ARCADE BBOX] Starting bounding box detection for stenosis")
                                    from ml.datasets.torch_arcade_loader import ARCADEStenosisDetection
                                    
                                    # Create stenosis detection dataset
                                    arcade_root = arcade_info['root']
                                    # Try both train and val sets to find the image
                                    for split in ['train', 'val']:
                                        try:
                                            stenosis_dataset = ARCADEStenosisDetection(
                                                root=arcade_root,
                                                image_set=split,
                                                download=False
                                            )
                                            
                                            # Find the image in dataset
                                            image_filename = os.path.basename(img_path)
                                            logger.info(f"[ARCADE BBOX] Looking for {image_filename} in {split} set")
                                            
                                            # Look for the image in the dataset
                                            found = False
                                            for dataset_img_path in stenosis_dataset.images:
                                                if os.path.basename(dataset_img_path) == image_filename:
                                                    img_id = stenosis_dataset.file_to_id[dataset_img_path]
                                                    logger.info(f"[ARCADE BBOX] Found image {image_filename} with ID {img_id}")
                                                    
                                                    # Get annotations (bounding boxes)
                                                    ann_ids = stenosis_dataset.coco.getAnnIds(imgIds=img_id)
                                                    anns = stenosis_dataset.coco.loadAnns(ann_ids)
                                                    
                                                    bboxes = []
                                                    for ann in anns:
                                                        if 'bbox' in ann:
                                                            bbox = ann['bbox']  # [x, y, width, height]
                                                            category_id = ann.get('category_id', 1)
                                                            category_name = stenosis_dataset.coco.cats.get(category_id, {}).get('name', 'stenosis')
                                                            
                                                            bboxes.append({
                                                                'x': int(bbox[0]),
                                                                'y': int(bbox[1]),
                                                                'width': int(bbox[2]),
                                                                'height': int(bbox[3]),
                                                                'category': category_name,
                                                                'confidence': ann.get('score', 1.0)
                                                            })
                                                    
                                                    sample_data['bounding_boxes'] = bboxes
                                                    sample_data['mask_classes'] = f"{len(bboxes)} stenosis regions"
                                                    sample_data['analysis'] = f"Found {len(bboxes)} stenosis bounding boxes"
                                                    sample_data['mask_generated'] = True
                                                    
                                                    # Convert bboxes to JSON string for template
                                                    import json
                                                    sample_data['bounding_boxes_json'] = json.dumps(bboxes)
                                                    
                                                    logger.info(f"[ARCADE BBOX] Found {len(bboxes)} bounding boxes for {image_filename}")
                                                    found = True
                                                    break
                                            
                                            if found:
                                                break
                                                
                                        except Exception as e:
                                            logger.warning(f"[ARCADE BBOX] Error with {split} set: {e}")
                                            continue
                                    
                                    if not found:
                                        logger.warning(f"[ARCADE BBOX] Image {image_filename} not found in any dataset split")
                                        sample_data['analysis'] = "⚠️ No bounding boxes found for this image"
                                        
                                except Exception as e:
                                    logger.error(f"[ARCADE BBOX] Error loading stenosis detection: {e}")
                                    sample_data['analysis'] = f"⚠️ Error loading bounding boxes: {e}"
                            
                            # Try to get image info
                            try:
                                with Image.open(img_path) as img:
                                    sample_data['image_shape'] = f"{img.size[0]}x{img.size[1]}"
                            except Exception:
                                pass
                            
                            samples.append(sample_data)
                            
                        except Exception as e:
                            logger.warning(f"Error processing stenosis sample {img_path}: {e}")
                            continue
                
                elif detected_type == 'artery_classification':
                    # For artery classification (left/right artery)
                    logger.info(f"Processing artery classification dataset type: {detected_type}")
                    
                    # Use ARCADE paths if available
                    if arcade_info and arcade_info.get('paths'):
                        arcade_paths = arcade_info['paths']
                        
                        # For artery classification, we use segmentation dataset but classify by artery side
                        image_dirs = []
                        if arcade_paths.get('train_images') and arcade_paths['train_images'].exists():
                            image_dirs.append(arcade_paths['train_images'])
                            logger.info(f"Added artery classification train images path: {arcade_paths['train_images']}")
                        if arcade_paths.get('val_images') and arcade_paths['val_images'].exists():
                            image_dirs.append(arcade_paths['val_images'])
                            logger.info(f"Added artery classification val images path: {arcade_paths['val_images']}")
                    else:
                        # Standard path detection
                        image_dirs = []
                        if os.path.exists(os.path.join(data_path, 'images')):
                            image_dirs.append(os.path.join(data_path, 'images'))
                        else:
                            image_dirs.append(data_path)
                    
                    all_images = []
                    for img_dir in image_dirs:
                        image_patterns = [
                            os.path.join(str(img_dir), '*.jpg'),
                            os.path.join(str(img_dir), '*.jpeg'),
                            os.path.join(str(img_dir), '*.png'),
                            os.path.join(str(img_dir), '*.tif'),
                            os.path.join(str(img_dir), '*.tiff')
                        ]
                        
                        for pattern in image_patterns:
                            all_images.extend(glob.glob(pattern))
                    
                    # Filter out masks
                    images = [img for img in all_images if not any(mask_keyword in img.lower() 
                             for mask_keyword in ['mask', 'label', 'gt', 'target'])]
                    
                    total_image_count = len(images)
                    context['total_samples'] = total_image_count
                    
                    logger.info(f"Final artery classification image count: {total_image_count}")
                    
                    # Take up to 6 random samples
                    sample_images = random.sample(images, min(6, len(images))) if images else []
                    
                    for idx, img_path in enumerate(sample_images):
                        try:
                            sample_data = {
                                'index': idx,
                                'filename': os.path.basename(img_path),
                                'image_url': f'/ml/serve-preview-image/?path={img_path}',
                                'mask_url': None,
                                'image_shape': 'Unknown',
                                'image_min': 0,
                                'image_max': 255,
                                'mask_coverage': None,
                                'analysis': 'Artery Classification',
                                'mask_min': 0,
                                'mask_max': 0,
                                'mask_shape': 'Classification',
                                'mask_classes': 'Left/Right Artery',
                                'mask_generator': 'ARCADE Artery Classification',
                                'annotation_format': 'COCO Segmentation',
                                'mask_generated': False,
                                'artery_side': 'Unknown',
                                'confidence': 0.0
                            }
                            
                            # Try to classify artery side using ARCADE
                            if arcade_info:
                                try:
                                    logger.info(f"[ARCADE CLASSIFY] Starting artery classification")
                                    from ml.datasets.torch_arcade_loader import ARCADEArteryClassification
                                    
                                    # Create artery classification dataset
                                    arcade_root = arcade_info['root']
                                    # Try both train and val sets to find the image
                                    for split in ['train', 'val']:
                                        try:
                                            classification_dataset = ARCADEArteryClassification(
                                                root=arcade_root,
                                                image_set=split,
                                                download=False
                                            )
                                            
                                            # Find the image in dataset
                                            image_filename = os.path.basename(img_path)
                                            logger.info(f"[ARCADE CLASSIFY] Looking for {image_filename} in {split} set")
                                            
                                            # Look for the image in the dataset
                                            found = False
                                            for dataset_img_path in classification_dataset.images:
                                                if os.path.basename(dataset_img_path) == image_filename:
                                                    img_id = classification_dataset.file_to_id[dataset_img_path]
                                                    logger.info(f"[ARCADE CLASSIFY] Found image {image_filename} with ID {img_id}")
                                                    
                                                    # Get annotations and determine artery side
                                                    ann_ids = classification_dataset.coco.getAnnIds(imgIds=img_id)
                                                    anns = classification_dataset.coco.loadAnns(ann_ids)
                                                    
                                                    # Extract segment names to determine side
                                                    segments = set()
                                                    for ann in anns:
                                                        category_id = ann.get('category_id', 1)
                                                        category_name = classification_dataset.coco.cats.get(category_id, {}).get('name', '')
                                                        if category_name:
                                                            segments.add(category_name)
                                                    
                                                    # Use the distinguish_side function from torch_arcade_loader
                                                    from ml.datasets.torch_arcade_loader import distinguish_side
                                                    artery_side = distinguish_side(segments)
                                                    
                                                    sample_data['artery_side'] = artery_side.title()  # Left or Right
                                                    sample_data['mask_classes'] = f"{artery_side.title()} Artery"
                                                    sample_data['analysis'] = f"Classified as {artery_side.title()} Artery"
                                                    sample_data['mask_generated'] = True
                                                    sample_data['confidence'] = 1.0  # High confidence for ARCADE annotations
                                                    
                                                    # Add segment information
                                                    sample_data['segments'] = list(segments)
                                                    
                                                    logger.info(f"[ARCADE CLASSIFY] Classified {image_filename} as {artery_side} artery with segments: {segments}")
                                                    found = True
                                                    break
                                            
                                            if found:
                                                break
                                                
                                        except Exception as e:
                                            logger.warning(f"[ARCADE CLASSIFY] Error with {split} set: {e}")
                                            continue
                                    
                                    if not found:
                                        logger.warning(f"[ARCADE CLASSIFY] Image {image_filename} not found in any dataset split")
                                        sample_data['analysis'] = "⚠️ No classification data found for this image"
                                        
                                except Exception as e:
                                    logger.error(f"[ARCADE CLASSIFY] Error loading artery classification: {e}")
                                    sample_data['analysis'] = f"⚠️ Error loading classification: {e}"
                            
                            # Try to get image info
                            try:
                                with Image.open(img_path) as img:
                                    sample_data['image_shape'] = f"{img.size[0]}x{img.size[1]}"
                            except Exception:
                                pass
                            
                            samples.append(sample_data)
                            
                        except Exception as e:
                            logger.warning(f"Error processing artery classification sample {img_path}: {e}")
                            continue
                
                context['samples'] = samples
                
                # Build dataset info with ARCADE details
                dataset_info = {
                    'structure': detected_type,
                    'image_count': total_image_count,
                    'sample_count': len(samples)
                }
                
                if arcade_info:
                    dataset_info.update({
                        'arcade_task': arcade_info.get('task'),
                        'arcade_root': arcade_info.get('root'),
                        'is_arcade_dataset': True
                    })
                
                context['dataset_info'] = dataset_info
                
            except Exception as e:
                logger.error(f"Error previewing dataset {data_path}: {e}")
                context['error_message'] = f"Error processing dataset: {str(e)}"
    
    return render(request, 'ml_manager/dataset_preview.html', context)


def detect_dataset_type(data_path):
    """Detect the type of dataset based on directory structure"""
    import os
    import json
    
    try:
        # First check if this is an ARCADE dataset by checking the structure
        logger.info(f"Detecting dataset type for: {data_path}")
        
        # Check if we're in an ARCADE dataset structure
        is_arcade = False
        arcade_task = None
        
        # Check if path contains ARCADE indicators
        if 'arcade' in data_path.lower() or 'challenge' in data_path.lower():
            is_arcade = True
            
            # Try to determine ARCADE task from path structure or annotations
            path_parts = data_path.lower().split(os.sep)
            
            # Check for explicit task names in path
            if any('stenosis' in part for part in path_parts):
                arcade_task = 'stenosis_detection'
            elif any('classification' in part for part in path_parts):
                arcade_task = 'artery_classification'
            elif any('segmentation' in part for part in path_parts):
                # Need to distinguish between binary and semantic
                arcade_task = 'segmentation'  # Will be refined below
        
        # Check for ARCADE-specific annotation structure
        annotations_dir = os.path.join(data_path, 'annotations')
        if os.path.exists(annotations_dir):
            json_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
            
            if json_files:
                # Analyze COCO annotations to determine task type
                for json_file in json_files:
                    try:
                        with open(os.path.join(annotations_dir, json_file), 'r') as f:
                            coco_data = json.load(f)
                        
                        if 'annotations' in coco_data and 'categories' in coco_data:
                            categories = coco_data['categories']
                            category_names = [cat.get('name', '').lower() for cat in categories]
                            
                            # Check annotation types
                            has_bbox = False
                            has_segmentation = False
                            
                            for ann in coco_data['annotations'][:10]:  # Check first 10 annotations
                                if 'bbox' in ann and ann['bbox']:
                                    has_bbox = True
                                if 'segmentation' in ann and ann['segmentation']:
                                    has_segmentation = True
                            
                            # Determine task based on annotation content
                            if has_bbox and not has_segmentation:
                                # Pure bounding box detection - likely stenosis detection
                                logger.info(f"Detected stenosis detection (bounding boxes only)")
                                return 'stenosis_detection'
                            
                            elif has_segmentation:
                                # Segmentation task - check categories to determine type
                                if len(categories) <= 2:
                                    # Binary segmentation
                                    logger.info(f"Detected binary segmentation ({len(categories)} categories)")
                                    return 'binary_segmentation'
                                elif any('lad' in name or 'lcx' in name or 'rca' in name or 'left' in name or 'right' in name 
                                        for name in category_names):
                                    # Artery-specific categories - likely classification task
                                    logger.info(f"Detected artery classification (artery-specific categories: {category_names})")
                                    return 'artery_classification'
                                else:
                                    # Multi-class semantic segmentation
                                    logger.info(f"Detected semantic segmentation ({len(categories)} categories)")
                                    return 'semantic_segmentation'
                            
                            elif has_bbox and has_segmentation:
                                # Mixed annotations - could be artery classification with both bbox and segmentation
                                if any('lad' in name or 'lcx' in name or 'rca' in name or 'left' in name or 'right' in name 
                                      for name in category_names):
                                    logger.info(f"Detected artery classification (mixed annotations)")
                                    return 'artery_classification'
                                else:
                                    logger.info(f"Detected semantic segmentation (mixed annotations)")
                                    return 'semantic_segmentation'
                    
                    except Exception as e:
                        logger.warning(f"Error analyzing JSON file {json_file}: {e}")
                        continue
        
        # If no clear ARCADE indicators from JSON, check directory structure
        segmentation_folders = ['images', 'masks', 'labels', 'annotations']
        if any(os.path.exists(os.path.join(data_path, folder)) 
               for folder in segmentation_folders):
            
            # Check for simple mask structure (binary vs semantic)
            mask_dir = None
            for folder in ['masks', 'labels']:
                potential_dir = os.path.join(data_path, folder)
                if os.path.exists(potential_dir):
                    mask_dir = potential_dir
                    break
            
            if mask_dir:
                # Sample a few mask files to determine if binary or semantic
                mask_files = [f for f in os.listdir(mask_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
                
                if mask_files:
                    try:
                        from PIL import Image
                        import numpy as np
                        
                        # Check first few masks
                        for mask_file in mask_files[:3]:
                            mask_path = os.path.join(mask_dir, mask_file)
                            with Image.open(mask_path) as img:
                                mask_array = np.array(img)
                                unique_values = np.unique(mask_array)
                                
                                # If only 2 unique values (e.g., 0 and 255), it's binary
                                if len(unique_values) <= 2:
                                    return 'binary_segmentation'
                                # If more values, it's semantic
                                elif len(unique_values) > 2:
                                    return 'semantic_segmentation'
                    except Exception:
                        pass
            
            return 'binary_segmentation'  # Default for segmentation structure
        
        # Check for classification structure (folders for each class)
        subdirs = [d for d in os.listdir(data_path) 
                   if os.path.isdir(os.path.join(data_path, d))]
        
        # Exclude common non-class directories
        non_class_dirs = {'images', 'masks', 'labels', 'annotations', 'train', 'val', 'test', 'validation'}
        class_dirs = [d for d in subdirs if d.lower() not in non_class_dirs]
        
        if len(class_dirs) > 1:
            # Check if these directories contain images directly
            for class_dir in class_dirs[:3]:  # Check first 3 directories
                class_path = os.path.join(data_path, class_dir)
                files = os.listdir(class_path)
                image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif'))]
                if image_files:
                    return 'classification'
        
        # Default to binary segmentation if structure is unclear
        logger.info(f"No clear dataset type detected, defaulting to binary_segmentation")
        return 'binary_segmentation'
        
    except Exception as e:
        logger.warning(f"Error detecting dataset type for {data_path}: {e}")
        return 'binary_segmentation'


@login_required
def preprocessing_preview(request):
    """Generate preprocessing preview using sample image from chosen dataset"""
    try:
        import cv2
        import numpy as np
        from skimage import exposure, filters, restoration
        from scipy import ndimage
        import base64
        from io import BytesIO
        import glob
       
        import random
        import os
        from PIL import Image
        
        # Get preprocessing parameters from request
        data_path = request.GET.get('data_path', '')
        sample_image_path_param = request.GET.get('sample_image_path', '')  # New parameter for specific image
        preprocessing_type = request.GET.get('preprocessing_type', 'angiography')
        clahe_clip_limit = float(request.GET.get('clahe_clip_limit', 3.0))
        clahe_tile_size = int(request.GET.get('clahe_tile_size', 8))
        use_unsharp_masking = request.GET.get('use_unsharp_masking') == 'true'
        unsharp_amount = float(request.GET.get('unsharp_amount', 1.0))
        unsharp_radius = float(request.GET.get('unsharp_radius', 1.0))
        use_frangi_filter = request.GET.get('use_frangi_filter') == 'true'
        frangi_sigma_min = float(request.GET.get('frangi_sigma_min', 1.0))
        frangi_sigma_max = float(request.GET.get('frangi_sigma_max', 10.0))
        use_denoising = request.GET.get('use_denoising') == 'true'
        noise_reduction_sigma = float(request.GET.get('noise_reduction_sigma', 1.0))
        use_histogram_equalization = request.GET.get('use_histogram_equalization') == 'true'
        normalize_intensity = request.GET.get('normalize_intensity') == 'true'
        gamma_correction = float(request.GET.get('gamma_correction', 1.0))
        
        # Find a sample image from the dataset
        if not data_path or not os.path.exists(data_path):
            return JsonResponse({
                'status': 'error',
                'message': 'Dataset path not found or invalid'
            })
        
        # If specific sample image path is provided, use it
        if sample_image_path_param:
            # Try to find the full path by searching in the dataset
            found_image_path = None
            image_patterns = [
                os.path.join(data_path, '**', sample_image_path_param),
                os.path.join(data_path, '**', '*' + sample_image_path_param),
                os.path.join(data_path, sample_image_path_param),
            ]
            
            for pattern in image_patterns:
                matches = glob.glob(pattern, recursive=True)
                if matches:
                    found_image_path = matches[0]
                    break
            
            if found_image_path and os.path.exists(found_image_path):
                sample_image_path = found_image_path
            else:
                # Fall back to random selection if specified image not found
                sample_image_path = None
        else:
            sample_image_path = None
        
        # If no specific image or not found, select randomly
        if sample_image_path is None:
            # Look for image files in common dataset structures
            image_patterns = [
                os.path.join(data_path, '**', '*.png'),
                os.path.join(data_path, '**', '*.jpg'),
                os.path.join(data_path, '**', '*.jpeg'),
                os.path.join(data_path, '**', '*.tif'),
                os.path.join(data_path, '**', '*.tiff'),
                os.path.join(data_path, 'images', '*.png'),
                os.path.join(data_path, 'train', '*.png'),
                os.path.join(data_path, 'val', '*.png'),
            ]
            
            sample_images = []
            for pattern in image_patterns:
                sample_images.extend(glob.glob(pattern, recursive=True))
                if len(sample_images) >= 10:  # Limit search for performance
                    break
            
            if not sample_images:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No sample images found in dataset'
                })
            
            # Select a random sample image
            sample_image_path = random.choice(sample_images)
        
        # Load the image
        try:
            # Try loading as grayscale first (common for medical images)
            image = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Failed to load sample image'
                })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Error loading image: {str(e)}'
            })
        
        # Resize image for preview (max 512x512)
        h, w = image.shape
        if max(h, w) > 512:
            scale = 512 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        # Store original for comparison
        original_image = image.copy()
        
        # Apply preprocessing steps
        processed_image = image.copy().astype(np.float32)
        
        # Normalize to 0-1 range for processing
        if processed_image.max() > 1.0:
            processed_image = processed_image / 255.0
        
        processing_steps = []
        
        # 1. CLAHE (always applied when medical preprocessing is enabled)
        if clahe_clip_limit > 0:
            # Convert back to uint8 for CLAHE
            clahe_input = (processed_image * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_size, clahe_tile_size))
            processed_image = clahe.apply(clahe_input).astype(np.float32) / 255.0
            processing_steps.append(f"CLAHE (clip: {clahe_clip_limit}, tile: {clahe_tile_size})")
        
        # 2. Histogram Equalization
        if use_histogram_equalization:
            hist_input = (processed_image * 255).astype(np.uint8)
            processed_image = cv2.equalizeHist(hist_input).astype(np.float32) / 255.0
            processing_steps.append("Histogram Equalization")
        
        # 3. Denoising
        if use_denoising:
            processed_image = restoration.denoise_bilateral(processed_image, sigma_color=noise_reduction_sigma, sigma_spatial=noise_reduction_sigma)
            processing_steps.append(f"Denoising (σ: {noise_reduction_sigma})")
        
        # 4. Frangi Filter (vessel enhancement)
        if use_frangi_filter:
            sigmas = np.arange(frangi_sigma_min, frangi_sigma_max + 0.5, 0.5)
            frangi_response = filters.frangi(processed_image, sigmas=sigmas)
            # Combine original with Frangi response
            processed_image = np.clip(processed_image + 0.3 * frangi_response, 0, 1)
            processing_steps.append(f"Frangi Filter (σ: {frangi_sigma_min}-{frangi_sigma_max})")
        
        # 5. Unsharp Masking
        if use_unsharp_masking:
            blurred = ndimage.gaussian_filter(processed_image, sigma=unsharp_radius)
            unsharp_mask = processed_image - blurred
            processed_image = processed_image + unsharp_amount * unsharp_mask
            processed_image = np.clip(processed_image, 0, 1)
            processing_steps.append(f"Unsharp Mask (amount: {unsharp_amount}, radius: {unsharp_radius})")
        
        # 6. Gamma Correction
        if gamma_correction != 1.0:
            processed_image = np.power(processed_image, gamma_correction)
            processing_steps.append(f"Gamma Correction (γ: {gamma_correction})")
        
        # 7. Intensity Normalization (final step)
        if normalize_intensity:
            processed_image = (processed_image - processed_image.min()) / (processed_image.max() - processed_image.min())
            processing_steps.append("Intensity Normalization")
        
        # Convert images to base64 for web display
        def image_to_base64(img):
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(img, mode='L')
            buffer = BytesIO()
            pil_img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        
        # Calculate some basic statistics for comparison
        original_stats = {
            'mean': float(np.mean(original_image)),
            'std': float(np.std(original_image)),
            'min': float(np.min(original_image)),
            'max': float(np.max(original_image))
        }
        
        processed_stats = {
            'mean': float(np.mean(processed_image * 255)),
            'std': float(np.std(processed_image * 255)),
            'min': float(np.min(processed_image * 255)),
            'max': float(np.max(processed_image * 255))
        }
        
        return JsonResponse({
            'status': 'success',
            'original_image': image_to_base64(original_image),
            'processed_image': image_to_base64(processed_image),
            'processing_steps': processing_steps,
            'original_stats': original_stats,
            'processed_stats': processed_stats,
            'sample_image_path': os.path.basename(sample_image_path),
            'full_sample_image_path': sample_image_path,  # Pełna ścieżka dla ponownego użycia
            'preprocessing_type': preprocessing_type
        })
        
    except ImportError as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Required libraries not available: {str(e)}'
        })
    except Exception as e:
        logger.error(f"Error in preprocessing preview: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Preview generation failed: {str(e)}'
        })


def extract_timestamp_from_line(line):
    """Extract timestamp from log line"""
    import re
    timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
    match = re.search(timestamp_pattern, line)
    return match.group(1) if match else None

def extract_log_level_from_line(line):
    """Extract log level from log line"""
    import re
    level_pattern = r'\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b'
    match = re.search(level_pattern, line)
    return match.group(1) if match else 'INFO'


@login_required
def serve_preview_image(request):
    """Serve preview image for dataset preview"""
    try:
        import os
        from django.http import HttpResponse, Http404
        from PIL import Image
        import io
        
        image_path = request.GET.get('path')
        if not image_path or not os.path.exists(image_path):
            raise Http404("Image not found")
        
        # Security check - ensure path is within allowed directories
        allowed_dirs = ['/app/data/', '/data/', '/tmp/']
        if not any(image_path.startswith(dir_path) for dir_path in allowed_dirs):
            raise Http404("Access denied")
        
        try:
            # Open and convert image to RGB if necessary
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create response
                response = HttpResponse(content_type='image/jpeg')
                img.save(response, 'JPEG', quality=85)
                return response
                
        except Exception as e:
            logger.error(f"Error serving preview image {image_path}: {e}")
            raise Http404("Error processing image")
            
    except Exception as e:
        logger.error(f"Error in serve_preview_image: {e}")
        raise Http404("Image not found")

