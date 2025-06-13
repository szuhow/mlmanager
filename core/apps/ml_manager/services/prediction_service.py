"""
ML prediction services.
"""

import os
import tempfile
from pathlib import Path
from PIL import Image
from django.core.files.base import ContentFile
from django.conf import settings

from ..models import MLModel, Prediction


class MLPredictionService:
    """Service for handling ML model predictions."""
    
    def __init__(self, model_id):
        self.model = MLModel.objects.get(id=model_id)
    
    def predict(self, image_file, save_result=True):
        """
        Run prediction on an image.
        
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
            
            # Run inference using ML inference module
            from ml.inference.predict import run_prediction
            
            result = run_prediction(
                model_path=self._get_model_path(),
                image_path=temp_image_path,
                model_type=self.model.model_type
            )
            
            if save_result:
                prediction = self._save_prediction_result(
                    image_file, 
                    result['prediction_image'],
                    result['confidence_score']
                )
                result['prediction_id'] = prediction.id
            
            # Cleanup temporary file
            os.unlink(temp_image_path)
            
            return {
                'success': True,
                'result': result
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
        model_dir = Path(settings.BASE_DIR).parent / 'data' / 'models' / str(self.model.id)
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
