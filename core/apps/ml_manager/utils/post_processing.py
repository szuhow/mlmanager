import cv2
import numpy as np
import torch
from scipy import ndimage
from skimage import measure, morphology
from typing import Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class SegmentationPostProcessor:
    """
    Enhanced post-processing for segmentation models to reduce false positives
    and noise artifacts (small elements).
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        min_component_size: int = 50,
        morphology_kernel_size: int = 3,
        apply_opening: bool = True,
        apply_closing: bool = True,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize post-processor with configurable parameters.
        
        Args:
            threshold: Probability threshold for binary segmentation (0.5-0.8 recommended)
            min_component_size: Minimum size of connected components to keep (pixels)
            morphology_kernel_size: Size of morphological operations kernel
            apply_opening: Whether to apply morphological opening (removes small objects)
            apply_closing: Whether to apply morphological closing (fills holes)
            confidence_threshold: Higher threshold for more conservative segmentation
        """
        self.threshold = threshold
        self.min_component_size = min_component_size
        self.morphology_kernel_size = morphology_kernel_size
        self.apply_opening = apply_opening
        self.apply_closing = apply_closing
        self.confidence_threshold = confidence_threshold
        
        # Create morphological kernel
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (morphology_kernel_size, morphology_kernel_size)
        )
        
    def process(
        self, 
        probability_map: Union[np.ndarray, torch.Tensor], 
        return_stats: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Apply comprehensive post-processing to segmentation output.
        
        Args:
            probability_map: Model output probability map (H, W) or (1, H, W)
            return_stats: Whether to return processing statistics
            
        Returns:
            Processed binary mask and optionally processing statistics
        """
        
        # Convert to numpy if tensor
        if isinstance(probability_map, torch.Tensor):
            prob_map = probability_map.detach().cpu().numpy()
        else:
            prob_map = probability_map.copy()
            
        # Handle different input shapes
        if len(prob_map.shape) == 3:
            prob_map = prob_map.squeeze()
        elif len(prob_map.shape) == 4:
            prob_map = prob_map.squeeze()
            
        original_shape = prob_map.shape
        stats = {
            'original_shape': original_shape,
            'threshold_used': self.threshold,
            'min_component_size': self.min_component_size
        }
        
        # Step 1: Apply threshold with higher confidence threshold for segmentation
        binary_mask = (prob_map >= self.confidence_threshold).astype(np.uint8)
        stats['pixels_after_threshold'] = np.sum(binary_mask)
        
        # Step 2: Apply morphological opening to remove small noise
        if self.apply_opening:
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self.kernel)
            stats['pixels_after_opening'] = np.sum(binary_mask)
            
        # Step 3: Apply morphological closing to fill holes
        if self.apply_closing:
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, self.kernel)
            stats['pixels_after_closing'] = np.sum(binary_mask)
            
        # Step 4: Remove small connected components (most effective for noise removal)
        binary_mask = self._remove_small_components(binary_mask, stats)
        
        # Step 5: Optional additional filtering based on shape properties
        binary_mask = self._filter_by_shape_properties(binary_mask, stats)
        
        stats['final_pixels'] = np.sum(binary_mask)
        stats['reduction_ratio'] = 1 - (stats['final_pixels'] / max(stats['pixels_after_threshold'], 1))
        
        logger.info(f"Post-processing reduced noise by {stats['reduction_ratio']:.2%}")
        
        if return_stats:
            return binary_mask, stats
        return binary_mask
    
    def _remove_small_components(self, binary_mask: np.ndarray, stats: Dict) -> np.ndarray:
        """Remove connected components smaller than threshold."""
        # Label connected components
        labeled_mask = measure.label(binary_mask, connectivity=2)
        component_props = measure.regionprops(labeled_mask)
        
        stats['components_before_filtering'] = len(component_props)
        
        # Create new mask with only large components
        filtered_mask = np.zeros_like(binary_mask)
        kept_components = 0
        
        for prop in component_props:
            if prop.area >= self.min_component_size:
                # Additional filtering: check if component is reasonably shaped
                if self._is_valid_component(prop):
                    filtered_mask[labeled_mask == prop.label] = 1
                    kept_components += 1
                    
        stats['components_after_filtering'] = kept_components
        stats['components_removed'] = stats['components_before_filtering'] - kept_components
        
        return filtered_mask
    
    def _is_valid_component(self, prop) -> bool:
        """Check if a component has valid shape properties for arteries."""
        # Basic shape validation for artery-like structures
        
        # Aspect ratio check: arteries shouldn't be too circular or too elongated
        if prop.minor_axis_length > 0:
            aspect_ratio = prop.major_axis_length / prop.minor_axis_length
            if aspect_ratio > 20:  # Too elongated, likely noise
                return False
                
        # Solidity check: arteries should be reasonably solid (not too fragmented)
        if prop.solidity < 0.3:  # Too fragmented
            return False
            
        # Extent check: component should fill its bounding box reasonably
        if prop.extent < 0.1:  # Too sparse
            return False
            
        return True
    
    def _filter_by_shape_properties(self, binary_mask: np.ndarray, stats: Dict) -> np.ndarray:
        """Additional filtering based on advanced shape properties."""
        # This can be extended with more sophisticated shape analysis
        # For now, apply basic median filtering to smooth edges
        
        # Apply median filter to reduce jagged edges
        filtered_mask = cv2.medianBlur(binary_mask, 3)
        
        stats['shape_filtering_applied'] = True
        return filtered_mask
    
    def adaptive_threshold(self, probability_map: np.ndarray) -> float:
        """
        Calculate adaptive threshold based on image statistics.
        Useful for handling different image qualities.
        """
        # Use Otsu's method on probability map
        prob_uint8 = (probability_map * 255).astype(np.uint8)
        threshold_val, _ = cv2.threshold(prob_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive_threshold = threshold_val / 255.0
        
        # Ensure threshold is within reasonable bounds
        adaptive_threshold = max(0.3, min(0.8, adaptive_threshold))
        
        return adaptive_threshold


class ClassificationPostProcessor:
    """Post-processor for classification outputs."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
    
    def process(self, logits: torch.Tensor, return_stats: bool = False):
        """Process classification outputs."""
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)
        confidence = torch.max(probabilities, dim=-1)[0]
        
        # Apply confidence threshold
        uncertain_predictions = confidence < self.confidence_threshold
        
        stats = {
            'max_confidence': confidence.max().item(),
            'min_confidence': confidence.min().item(),
            'mean_confidence': confidence.mean().item(),
            'uncertain_predictions': uncertain_predictions.sum().item()
        }
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'uncertain': uncertain_predictions
        }
        
        if return_stats:
            return result, stats
        return result


def enhanced_post_processing(
    model_output: Union[np.ndarray, torch.Tensor],
    model_type: str = 'segmentation',
    **kwargs
) -> Dict:
    """
    Main function for enhanced post-processing of model outputs.
    
    Args:
        model_output: Raw model output
        model_type: 'segmentation' or 'classification'
        **kwargs: Additional parameters for post-processors
        
    Returns:
        Dictionary with processed results and metadata
    """
    
    if model_type == 'segmentation':
        processor = SegmentationPostProcessor(**kwargs)
        processed_output, stats = processor.process(model_output, return_stats=True)
        
        return {
            'processed_output': processed_output,
            'type': 'segmentation',
            'stats': stats,
            'post_processing_applied': True
        }
        
    elif model_type == 'classification':
        processor = ClassificationPostProcessor(**kwargs)
        result, stats = processor.process(model_output, return_stats=True)
        
        return {
            'processed_output': result,
            'type': 'classification', 
            'stats': stats,
            'post_processing_applied': True
        }
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
