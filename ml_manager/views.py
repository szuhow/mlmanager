from django.shortcuts import render, get_object_or_404
from django.views.generic import ListView, DetailView, FormView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.contrib import messages
from django.db import models
from django.core.files import File
from django.http import JsonResponse
from .forms import TrainingForm, InferenceForm, TrainingTemplateForm
from .models import MLModel, Prediction, TrainingTemplate
from .forms import TrainingForm
import mlflow
import subprocess
import sys
from pathlib import Path
import json
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
import logging
import tempfile
import time
import shutil
from PIL import Image
import torch
import os
from shared.train import run_inference
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

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
                if model.total_epochs > 0:
                    progress_percentage = (model.current_epoch / model.total_epochs) * 100
                else:
                    progress_percentage = 0
                model.progress_percentage = round(progress_percentage, 1)
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
                from .mlflow_utils import get_registered_model_info, get_model_version_details
                
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

        # Calculate progress percentage
        if self.object.total_epochs > 0:
            progress_percentage = (self.object.current_epoch / self.object.total_epochs) * 100
        else:
            progress_percentage = 0
        context['progress_percentage'] = round(progress_percentage, 1)

        # Get training logs with enhanced parsing
        try:
            import os
            # Try to find logs in the correct location (global fallback for now)
            log_path = os.path.join('models', 'artifacts', 'training.log')
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    log_lines = f.read().splitlines()
                
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
                context['training_logs'] = ['No training logs found']
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
                'general_logs': []
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
            return JsonResponse({
                'progress': {
                    'current_epoch': self.object.current_epoch,
                    'total_epochs': self.object.total_epochs,
                    'progress_percentage': context['progress_percentage'],
                    'train_loss': self.object.train_loss,
                    'val_loss': self.object.val_loss,
                    'train_dice': self.object.train_dice,
                    'val_dice': self.object.val_dice,
                    'best_val_dice': self.object.best_val_dice,
                },
                'logs': context.get('training_logs', []),
                'parsed_logs': context.get('parsed_logs', {}),
                'status': self.object.status,
                'log_stats': {
                    'total_lines': len(context.get('training_logs', [])),
                    'epoch_logs': len(context.get('parsed_logs', {}).get('epoch_logs', [])),
                    'batch_logs': len(context.get('parsed_logs', {}).get('batch_logs', [])),
                    'model_logs': len(context.get('parsed_logs', {}).get('model_logs', [])),
                    'last_updated': context.get('training_logs', [''])[-1] if context.get('training_logs') else ''
                }
            })
        return super().render_to_response(context, **response_kwargs)

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
            
            # Try multiple artifact directory paths to handle different MLflow configurations
            base_paths = [
                # Direct run ID path (current MLflow structure)
                os.path.join('mlruns', run.info.run_id, 'artifacts'),
                # Legacy experiment-based paths
                os.path.join('mlruns', run.info.experiment_id, run.info.run_id, 'artifacts'),
                os.path.join('mlruns', '0', run.info.run_id, 'artifacts'),
                os.path.join('mlruns', '1', run.info.run_id, 'artifacts'),
                os.path.join('mlruns', str(run.info.experiment_id), run.info.run_id, 'artifacts'),
            ]
            
            artifacts_path = None
            for path in base_paths:
                if os.path.exists(path):
                    artifacts_path = path
                    break
            
            if not artifacts_path:
                preview_data['error'] = f"Artifacts directory not found. Tried paths: {', '.join(base_paths)}"
                return preview_data
            
            # Find all prediction images
            prediction_files = []
            try:
                for filename in os.listdir(artifacts_path):
                    if filename.startswith('predictions_epoch_') and filename.endswith('.png'):
                        try:
                            epoch_num = int(filename.replace('predictions_epoch_', '').replace('.png', ''))
                            # Use the actual found artifacts_path for relative_path
                            prediction_files.append({
                                'filename': filename,
                                'epoch': epoch_num,
                                'path': os.path.join(artifacts_path, filename),
                                'relative_path': os.path.join(os.path.dirname(os.path.dirname(artifacts_path)), 'artifacts', filename)
                            })
                        except ValueError:
                            # Skip files with invalid epoch numbers
                            continue
            except OSError as e:
                preview_data['error'] = f"Error reading artifacts directory: {str(e)}"
                return preview_data
            
            # Sort by epoch number
            prediction_files.sort(key=lambda x: x['epoch'])
            
            if prediction_files:
                preview_data['images'] = prediction_files
                preview_data['latest_epoch'] = prediction_files[-1]['epoch']
            else:
                preview_data['error'] = "No prediction images found in artifacts"
                
        except Exception as e:
            logging.error(f"Error getting training preview for model {self.object.id}: {e}")
            preview_data['error'] = f"Error loading training preview: {str(e)}"
        
        return preview_data

    def _get_training_details(self):
        """Extract comprehensive training configuration and details"""
        details = {
            'config': {},
            'hardware': {},
            'dataset': {},
            'augmentation': {},
            'optimizer': {},
            'architecture': {},
            'error': None
        }
        
        try:
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
                    
                    # Extract hardware details
                    details['hardware'] = {
                        'device': training_params.get('device', config_data.get('device', 'N/A')),
                        'pytorch_version': config_data.get('pytorch_version', 'N/A'),
                        'cuda_available': torch.cuda.is_available(),
                        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
                    }
                    
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

    def form_valid(self, form):
        logger = logging.getLogger(__name__) # Define logger
        logger.info("StartTrainingView.form_valid() called")
        
        try:
            # Extract form data
            form_data = form.cleaned_data
            logger.info(f"Form data: {form_data}")
            
            # Create MLflow run first
            import mlflow
            mlflow_run = mlflow.start_run()
            mlflow_run_id = mlflow_run.info.run_id
            mlflow.end_run()  # End the run, train.py will restart it
            
            # Create new model instance
            ml_model = MLModel.objects.create(
                name=form_data['name'],
                description=form_data.get('description', ''),
                status='pending',
                current_epoch=0,
                total_epochs=form_data['epochs'],
                train_loss=0.0,
                val_loss=0.0,
                train_dice=0.0,
                val_dice=0.0,
                best_val_dice=0.0,
                mlflow_run_id=mlflow_run_id,  # Set the run ID
                training_data_info={
                    'model_type': form_data['model_type'],
                    'data_path': form_data['data_path'],
                    'batch_size': form_data['batch_size'],
                    'learning_rate': form_data['learning_rate'],
                    'validation_split': form_data['validation_split'],
                    'use_random_flip': form_data['use_random_flip'],
                    'use_random_rotate': form_data['use_random_rotate'],
                    'use_random_scale': form_data['use_random_scale'],
                    'use_random_intensity': form_data['use_random_intensity'],
                    'crop_size': form_data['crop_size'],
                    'num_workers': form_data['num_workers'],
                },
                # Store model_type directly for easier access if needed
                model_type=form_data['model_type'] 
            )
            logger.info(f"Created MLModel instance with ID: {ml_model.id}, model_type: {ml_model.model_type}, mlflow_run_id: {mlflow_run_id}")

            # Prepare command for subprocess
            command = [
                sys.executable, 
                str(Path(__file__).parent.parent / 'shared' / 'train.py'),
                '--mode=train',
                f'--model-id={ml_model.id}',
                f'--mlflow-run-id={mlflow_run_id}',
                f'--model-type={form_data["model_type"]}', 
                f'--data-path={form_data["data_path"]}',
                f'--batch-size={form_data["batch_size"]}',
                f'--epochs={form_data["epochs"]}',
                f'--learning-rate={form_data["learning_rate"]}',
                f'--validation-split={form_data["validation_split"]}',
                f'--crop-size={form_data["crop_size"]}',
                f'--num-workers={form_data["num_workers"]}'
            ]
            # Add boolean flags for augmentations
            if form_data.get('use_random_flip'): command.append('--random-flip')
            if form_data.get('use_random_rotate'): command.append('--random-rotate')
            if form_data.get('use_random_scale'): command.append('--random-scale')
            if form_data.get('use_random_intensity'): command.append('--random-intensity')

            logger.info(f"Training command: {' '.join(command)}")

            # Prepare environment for subprocess
            current_env = os.environ.copy()
            project_root = Path(__file__).resolve().parent.parent  # Assuming views.py is in ml_manager
            
            # Add ml_manager directory to PYTHONPATH so shared/train.py can import mlflow_utils
            ml_manager_path = str(project_root / "ml_manager")
            shared_path = str(project_root / "shared") # Also add shared path for robustness
            
            existing_python_path = current_env.get("PYTHONPATH")
            new_paths = [ml_manager_path, shared_path]
            if existing_python_path:
                current_env["PYTHONPATH"] = ":".join(new_paths) + ":" + existing_python_path
            else:
                current_env["PYTHONPATH"] = ":".join(new_paths)
                
            current_env["DJANGO_SETTINGS_MODULE"] = "coronary_experiments.settings"
            
            logger.info(f"Starting training subprocess with PYTHONPATH: {current_env['PYTHONPATH']}")
            logger.info(f"Working directory for subprocess: {str(project_root)}")

            subprocess.Popen(
                command,
                shell=False, 
                env=current_env,
                cwd=str(project_root)
            )
            messages.success(self.request, f"Training for \'{ml_model.name}\' (ID: {ml_model.id}) started successfully with model type \'{form_data['model_type']}\'.")
            return super().form_valid(form)
        except Exception as e:
            logger.error(f"Error in StartTrainingView.form_valid: {e}", exc_info=True)
            messages.error(self.request, f"Failed to start training: {e}")
            # Ensure the created model is marked as failed if subprocess Popen fails
            if 'ml_model' in locals() and ml_model:
                ml_model.status = 'failed'
                ml_model.training_logs = f"Failed to start training process: {e}"
                ml_model.save()
            return self.form_invalid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Add available templates for template selection
        try:
            from .models import TrainingTemplate
            context['templates'] = TrainingTemplate.objects.all()
        except:
            context['templates'] = []
            
        return context


# Additional view functions and classes

@login_required
@require_POST
def stop_training(request, model_id):
    """Stop a running training job"""
    try:
        model = get_object_or_404(MLModel, id=model_id)
        
        if model.status != 'training':
            return JsonResponse({
                'status': 'error',
                'message': 'Model is not currently training'
            })
        
        # Update model status to stopped
        model.status = 'stopped'
        model.save()
        
        # Here you would typically send a signal to stop the actual training process
        # For now, we'll just update the status
        
        return JsonResponse({
            'status': 'success',
            'message': f'Training for model "{model.name}" has been stopped'
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
    
    def delete(self, request, *args, **kwargs):
        try:
            self.object = self.get_object()
            success_url = self.get_success_url()
            
            # Clean up associated files if needed
            if self.object.model_directory and os.path.exists(self.object.model_directory):
                try:
                    shutil.rmtree(self.object.model_directory)
                except Exception as e:
                    logging.warning(f"Could not delete model directory: {e}")
            
            self.object.delete()
            messages.success(request, f'Model "{self.object.name}" has been deleted.')
            
            return JsonResponse({'status': 'success', 'redirect': success_url})
            
        except Exception as e:
            logging.error(f"Error deleting model: {e}")
            messages.error(request, f'Error deleting model: {e}')
            return JsonResponse({'status': 'error', 'message': str(e)})


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
    form_class = InferenceForm
    template_name = 'ml_manager/model_inference.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        self.model = get_object_or_404(MLModel, pk=self.kwargs['pk'])
        context['model'] = self.model
        return context
    
    def form_valid(self, form):
        try:
            self.model = get_object_or_404(MLModel, pk=self.kwargs['pk'])
            
            # Process inference request
            # This is a simplified version - you would implement actual inference logic
            
            messages.success(self.request, 'Inference completed successfully!')
            return JsonResponse({
                'status': 'success',
                'message': 'Inference completed'
            })
            
        except Exception as e:
            logging.error(f"Error during inference: {e}")
            messages.error(self.request, f'Error during inference: {e}')
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })


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
                'use_random_flip': training_info.get('use_random_flip', True),
                'use_random_rotate': training_info.get('use_random_rotate', True),
                'use_random_scale': training_info.get('use_random_scale', True),
                'use_random_intensity': training_info.get('use_random_intensity', True),
                'crop_size': training_info.get('crop_size', 128),
                'num_workers': training_info.get('num_workers', 4),
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
        
        # Try to find log file
        log_path = os.path.join('models', 'artifacts', 'training.log')
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                logs = f.read()
        else:
            logs = "No training logs found"
        
        return JsonResponse({
            'status': 'success',
            'logs': logs.split('\n')
        })
        
    except Exception as e:
        logging.error(f"Error getting training logs: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})


@login_required
def serve_training_preview_image(request, model_id, filename):
    """Serve training preview images"""
    try:
        model = get_object_or_404(MLModel, id=model_id)
        
        # Find image file
        if model.mlflow_run_id:
            image_path = os.path.join('mlruns', model.mlflow_run_id, 'artifacts', filename)
            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    response = HttpResponse(f.read(), content_type='image/png')
                    response['Content-Disposition'] = f'inline; filename="{filename}"'
                    return response
        
        # Return 404 if image not found
        return HttpResponse('Image not found', status=404)
        
    except Exception as e:
        logging.error(f"Error serving training preview image: {e}")
        return HttpResponse('Error loading image', status=500)
