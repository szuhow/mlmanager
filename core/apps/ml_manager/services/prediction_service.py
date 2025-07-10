"""
ML prediction services with Celery integration.
"""

import os
import tempfile
from pathlib import Path
from PIL import Image
from django.core.files.base import ContentFile
from django.conf import settings

from ..models import MLModel, Prediction


class MLPredictionService:
    """Service for handling ML model predictions with Celery."""
    
    def __init__(self, model_id):
        self.model = MLModel.objects.get(id=model_id)
    
    def predict(self, image_file, save_result=True):
        """
        Run prediction on an image using Celery.
        
        Args:
            image_file: Uploaded image file
            save_result: Whether to save prediction to database
        
        Returns:
            dict: Prediction results
        """
        try:
            # Import Celery task
            from ..tasks.tasks import run_inference_task
            
            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                for chunk in image_file.chunks():
                    temp_file.write(chunk)
                temp_image_path = temp_file.name
            
            # Prepare inference parameters
            inference_params = {
                'model_type': self.model.model_type,
                'output_dir': str(Path(settings.BASE_DIR) / 'data' / 'inference_results'),
                'threshold': 0.5,
                'weights_path': self.model.model_weights_path if self.model.model_weights_path else None
            }
            
            # Start inference task asynchronously
            task = run_inference_task.delay(
                self.model.id, 
                temp_image_path, 
                inference_params
            )
            
            return {
                'success': True,
                'task_id': task.id,
                'model_id': self.model.id,
                'status': 'queued',
                'temp_image_path': temp_image_path
            }
            
        except Exception as e:
            # Cleanup on error
            if 'temp_image_path' in locals():
                try:
                    os.unlink(temp_image_path)
                except:
                    pass
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict_sync(self, image_file, save_result=True):
        """
        Run prediction on an image synchronously (fallback method).
        
        Args:
            image_file: Uploaded image file
            save_result: Whether to save prediction to database
        
        Returns:
            dict: Prediction results
        """
        try:
            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                for chunk in image_file.chunks():
                    temp_file.write(chunk)
                temp_image_path = temp_file.name
            
            # Try to run inference using the training script
            import subprocess
            import sys
            import json
            
            # Prepare inference command
            inference_script = Path(__file__).parent.parent / 'training' / 'train.py'
            command = [
                sys.executable,
                str(inference_script),
                '--mode=predict',
                f'--model-id={self.model.id}',
                f'--input-path={temp_image_path}',
                f'--output-dir={Path(settings.BASE_DIR) / "data" / "inference_results"}',
                f'--model-type={self.model.model_type}',
                '--threshold=0.5',
            ]
            
            if self.model.model_weights_path:
                command.append(f'--weights-path={self.model.model_weights_path}')
            
            # Execute inference
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            # Cleanup temporary file
            os.unlink(temp_image_path)
            
            if process.returncode == 0:
                # Parse output for results
                output_lines = process.stdout.strip().split('\n')
                
                # Look for result information in output
                result_info = {
                    'success': True,
                    'model_id': self.model.id,
                    'status': 'completed',
                    'output': output_lines
                }
                
                if save_result:
                    # Create prediction record
                    prediction = self._save_prediction_result(
                        image_file,
                        None,  # No prediction image for now
                        0.5    # Default confidence
                    )
                    result_info['prediction_id'] = prediction.id
                
                return result_info
            else:
                return {
                    'success': False,
                    'error': process.stderr,
                    'return_code': process.returncode
                }
                
        except Exception as e:
            # Cleanup on error
            if 'temp_image_path' in locals():
                try:
                    os.unlink(temp_image_path)
                except:
                    pass
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_model_path(self):
        """Get the path to the trained model file."""
        if self.model.model_weights_path:
            return self.model.model_weights_path
        
        model_dir = Path(settings.BASE_DIR) / 'data' / 'models' / str(self.model.id)
        model_files = list(model_dir.glob('*.pth'))
        
        if not model_files:
            raise FileNotFoundError(f"No trained model found for model ID {self.model.id}")
        
        # Return the most recent model file
        return max(model_files, key=os.path.getctime)
    
    def _save_prediction_result(self, input_image, prediction_image, confidence_score):
        """Save prediction result to database."""
        prediction = Prediction(
            model=self.model,
            confidence_score=confidence_score
        )
        
        # Save input image
        prediction.input_image.save(
            f'input_{prediction.id}.png',
            ContentFile(input_image.read()),
            save=False
        )
        
        # Save prediction result image
        if prediction_image:
            prediction.result_image.save(
                f'result_{prediction.id}.png',
                ContentFile(prediction_image),
                save=False
            )
        
        prediction.save()
        return prediction
