"""
Advanced Medical Image Preprocessing for Coronary Segmentation
=============================================================

This module provides specialized preprocessing functions for medical images,
particularly coronary angiography images. Includes noise reduction, contrast
enhancement, vessel enhancement, and normalization techniques.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union, Dict, Any
import warnings
from skimage import filters, morphology, measure, exposure
from scipy import ndimage
import matplotlib.pyplot as plt


class MedicalImagePreprocessor:
    """Advanced preprocessor for medical images with coronary-specific enhancements."""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (512, 512),
                 normalize_method: str = 'minmax',
                 enhance_contrast: bool = True,
                 enhance_vessels: bool = True):
        """
        Initialize the medical image preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            normalize_method: Normalization method ('minmax', 'zscore', 'percentile')
            enhance_contrast: Whether to apply contrast enhancement
            enhance_vessels: Whether to apply vessel enhancement
        """
        self.target_size = target_size
        self.normalize_method = normalize_method
        self.enhance_contrast = enhance_contrast
        self.enhance_vessels = enhance_vessels
        
    def preprocess(self, 
                   image: np.ndarray,
                   mask: Optional[np.ndarray] = None,
                   augment: bool = False) -> Dict[str, np.ndarray]:
        """
        Complete preprocessing pipeline for medical images.
        
        Args:
            image: Input image (H, W) or (H, W, C)
            mask: Optional ground truth mask (H, W)
            augment: Whether to apply data augmentation
            
        Returns:
            Dictionary with processed image and mask
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        # Noise reduction
        image = self.denoise(image)
        
        # Contrast enhancement
        if self.enhance_contrast:
            image = self.enhance_contrast_clahe(image)
            
        # Vessel enhancement
        if self.enhance_vessels:
            image = self.enhance_vessels_frangi(image)
            
        # Resize
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            
        # Normalization
        image = self.normalize(image)
        
        # Data augmentation
        if augment and mask is not None:
            image, mask = self.augment_image_mask(image, mask)
            
        result = {'image': image}
        if mask is not None:
            result['mask'] = mask
            
        return result
    
    def denoise(self, image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """
        Apply denoising to medical image.
        
        Args:
            image: Input image
            method: Denoising method ('bilateral', 'gaussian', 'median', 'nlm')
            
        Returns:
            Denoised image
        """
        if method == 'bilateral':
            # Bilateral filter preserves edges while reducing noise
            return cv2.bilateralFilter(image.astype(np.uint8), 9, 75, 75).astype(np.float32)
        
        elif method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
            
        elif method == 'median':
            return cv2.medianBlur(image.astype(np.uint8), 5).astype(np.float32)
            
        elif method == 'nlm':
            # Non-local means denoising (computationally expensive)
            return cv2.fastNlMeansDenoising(image.astype(np.uint8), None, 10, 7, 21).astype(np.float32)
            
        else:
            return image
    
    def enhance_contrast_clahe(self, 
                              image: np.ndarray, 
                              clip_limit: float = 2.0, 
                              tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        Args:
            image: Input image
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of the neighborhood for histogram equalization
            
        Returns:
            Enhanced image
        """
        # Convert to uint8 for CLAHE
        img_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(img_uint8)
        
        # Convert back to float32
        return enhanced.astype(np.float32) / 255.0
    
    def enhance_contrast_unsharp_mask(self, 
                                     image: np.ndarray, 
                                     sigma: float = 1.0, 
                                     strength: float = 1.5) -> np.ndarray:
        """
        Apply unsharp masking for edge enhancement.
        
        Args:
            image: Input image
            sigma: Standard deviation for Gaussian blur
            strength: Strength of the enhancement
            
        Returns:
            Enhanced image
        """
        # Create blurred version
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        
        # Create unsharp mask
        unsharp_mask = image - blurred
        
        # Apply enhancement
        enhanced = image + strength * unsharp_mask
        
        # Clip values to valid range
        return np.clip(enhanced, 0, 1)
    
    def enhance_vessels_frangi(self, 
                              image: np.ndarray,
                              scale_range: Tuple[float, float] = (1, 10),
                              scale_step: float = 2,
                              beta1: float = 0.5,
                              beta2: float = 15) -> np.ndarray:
        """
        Apply Frangi vesselness filter to enhance vessel structures.
        
        Args:
            image: Input image
            scale_range: Range of scales to analyze
            scale_step: Step between scales
            beta1: Sensitivity to plate-like structures
            beta2: Sensitivity to blob-like structures
            
        Returns:
            Vessel-enhanced image
        """
        try:
            # Apply Frangi filter
            vesselness = filters.frangi(
                image,
                sigmas=np.arange(scale_range[0], scale_range[1], scale_step),
                alpha=beta1,
                beta=beta2
            )
            
            # Combine original with vesselness
            alpha = 0.7  # Weight for original image
            enhanced = alpha * image + (1 - alpha) * vesselness
            
            return np.clip(enhanced, 0, 1)
            
        except Exception as e:
            warnings.warn(f"Frangi filter failed: {e}. Using original image.")
            return image
    
    def enhance_vessels_hessian(self, image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """
        Apply Hessian-based vessel enhancement.
        
        Args:
            image: Input image
            sigma: Scale parameter for Hessian computation
            
        Returns:
            Vessel-enhanced image
        """
        try:
            # Compute Hessian eigenvalues
            hxx = ndimage.gaussian_filter(image, sigma, order=[2, 0])
            hxy = ndimage.gaussian_filter(image, sigma, order=[1, 1])
            hyy = ndimage.gaussian_filter(image, sigma, order=[0, 2])
            
            # Compute eigenvalues
            trace = hxx + hyy
            det = hxx * hyy - hxy ** 2
            
            # Eigenvalues
            lambda1 = 0.5 * (trace + np.sqrt(trace**2 - 4*det))
            lambda2 = 0.5 * (trace - np.sqrt(trace**2 - 4*det))
            
            # Vesselness measure (darker vessels)
            vesselness = np.zeros_like(image)
            mask = lambda2 < 0
            vesselness[mask] = np.exp(-lambda1[mask]**2 / (2 * lambda2[mask]**2))
            
            # Combine with original
            alpha = 0.8
            enhanced = alpha * image + (1 - alpha) * vesselness
            
            return np.clip(enhanced, 0, 1)
            
        except Exception as e:
            warnings.warn(f"Hessian vessel enhancement failed: {e}. Using original image.")
            return image
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using specified method.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        if self.normalize_method == 'minmax':
            return (image - image.min()) / (image.max() - image.min() + 1e-8)
            
        elif self.normalize_method == 'zscore':
            return (image - image.mean()) / (image.std() + 1e-8)
            
        elif self.normalize_method == 'percentile':
            p2, p98 = np.percentile(image, (2, 98))
            return np.clip((image - p2) / (p98 - p2 + 1e-8), 0, 1)
            
        else:
            return image
    
    def augment_image_mask(self, 
                          image: np.ndarray, 
                          mask: np.ndarray,
                          rotation_range: float = 15,
                          zoom_range: float = 0.1,
                          brightness_range: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random augmentations to image and mask simultaneously.
        
        Args:
            image: Input image
            mask: Input mask
            rotation_range: Range for random rotation (degrees)
            zoom_range: Range for random zoom
            brightness_range: Range for brightness adjustment
            
        Returns:
            Augmented image and mask
        """
        # Random rotation
        if np.random.random() < 0.5:
            angle = np.random.uniform(-rotation_range, rotation_range)
            center = (image.shape[1] // 2, image.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
        
        # Random zoom
        if np.random.random() < 0.5:
            zoom_factor = 1 + np.random.uniform(-zoom_range, zoom_range)
            h, w = image.shape[:2]
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            
            # Resize
            image_zoomed = cv2.resize(image, (new_w, new_h))
            mask_zoomed = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # Crop or pad to original size
            if zoom_factor > 1:  # Crop
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                image = image_zoomed[start_h:start_h+h, start_w:start_w+w]
                mask = mask_zoomed[start_h:start_h+h, start_w:start_w+w]
            else:  # Pad
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                image = np.pad(image_zoomed, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), mode='reflect')
                mask = np.pad(mask_zoomed, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), mode='constant')
        
        # Random brightness
        if np.random.random() < 0.5:
            brightness_factor = 1 + np.random.uniform(-brightness_range, brightness_range)
            image = np.clip(image * brightness_factor, 0, 1)
        
        # Random horizontal flip
        if np.random.random() < 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        
        return image, mask


# Specialized preprocessing functions for different image types
def preprocess_angiography(image: np.ndarray, 
                          enhance_vessels: bool = True,
                          target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Specialized preprocessing for coronary angiography images.
    
    Args:
        image: Input angiography image
        enhance_vessels: Whether to apply vessel enhancement
        target_size: Target image size
        
    Returns:
        Preprocessed image
    """
    preprocessor = MedicalImagePreprocessor(
        target_size=target_size,
        normalize_method='percentile',
        enhance_contrast=True,
        enhance_vessels=enhance_vessels
    )
    
    result = preprocessor.preprocess(image)
    return result['image']


def preprocess_ct_coronary(image: np.ndarray,
                          window_level: float = 40,
                          window_width: float = 400,
                          target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Specialized preprocessing for coronary CT images.
    
    Args:
        image: Input CT image (Hounsfield units)
        window_level: CT window level
        window_width: CT window width
        target_size: Target image size
        
    Returns:
        Preprocessed image
    """
    # Apply CT windowing
    min_val = window_level - window_width / 2
    max_val = window_level + window_width / 2
    
    image_windowed = np.clip(image, min_val, max_val)
    image_windowed = (image_windowed - min_val) / (max_val - min_val)
    
    # Apply standard preprocessing
    preprocessor = MedicalImagePreprocessor(
        target_size=target_size,
        normalize_method='minmax',
        enhance_contrast=True,
        enhance_vessels=True
    )
    
    result = preprocessor.preprocess(image_windowed)
    return result['image']


def preprocess_oct_coronary(image: np.ndarray,
                           speckle_reduction: bool = True,
                           target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Specialized preprocessing for coronary OCT images.
    
    Args:
        image: Input OCT image
        speckle_reduction: Whether to apply speckle noise reduction
        target_size: Target image size
        
    Returns:
        Preprocessed image
    """
    if speckle_reduction:
        # Apply speckle reduction using bilateral filter
        image = cv2.bilateralFilter(image.astype(np.uint8), 9, 75, 75).astype(np.float32)
    
    preprocessor = MedicalImagePreprocessor(
        target_size=target_size,
        normalize_method='percentile',
        enhance_contrast=True,
        enhance_vessels=False  # OCT has different vessel appearance
    )
    
    result = preprocessor.preprocess(image)
    return result['image']


# Utility functions for preprocessing validation and visualization
def visualize_preprocessing_steps(image: np.ndarray, 
                                 steps: bool = True,
                                 save_path: Optional[str] = None) -> None:
    """
    Visualize preprocessing steps for debugging and validation.
    
    Args:
        image: Input image
        steps: Whether to show individual preprocessing steps
        save_path: Optional path to save the visualization
    """
    preprocessor = MedicalImagePreprocessor()
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if steps:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Denoised
        denoised = preprocessor.denoise(image)
        axes[0, 1].imshow(denoised, cmap='gray')
        axes[0, 1].set_title('Denoised')
        axes[0, 1].axis('off')
        
        # CLAHE enhanced
        clahe_enhanced = preprocessor.enhance_contrast_clahe(denoised)
        axes[0, 2].imshow(clahe_enhanced, cmap='gray')
        axes[0, 2].set_title('CLAHE Enhanced')
        axes[0, 2].axis('off')
        
        # Vessel enhanced
        vessel_enhanced = preprocessor.enhance_vessels_frangi(clahe_enhanced)
        axes[1, 0].imshow(vessel_enhanced, cmap='gray')
        axes[1, 0].set_title('Vessel Enhanced')
        axes[1, 0].axis('off')
        
        # Normalized
        normalized = preprocessor.normalize(vessel_enhanced)
        axes[1, 1].imshow(normalized, cmap='gray')
        axes[1, 1].set_title('Normalized')
        axes[1, 1].axis('off')
        
        # Final result
        result = preprocessor.preprocess(image)
        final_image = result['image']
        axes[1, 2].imshow(final_image, cmap='gray')
        axes[1, 2].set_title('Final Result')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    else:
        # Simple before/after comparison
        result = preprocessor.preprocess(image)
        final_image = result['image']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(final_image, cmap='gray')
        axes[1].set_title('Preprocessed')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def validate_preprocessing(images: list, 
                          masks: Optional[list] = None) -> Dict[str, Any]:
    """
    Validate preprocessing on a batch of images.
    
    Args:
        images: List of input images
        masks: Optional list of corresponding masks
        
    Returns:
        Validation statistics
    """
    preprocessor = MedicalImagePreprocessor()
    
    stats = {
        'input_sizes': [],
        'output_sizes': [],
        'intensity_ranges': [],
        'processing_times': []
    }
    
    import time
    
    for i, image in enumerate(images):
        start_time = time.time()
        
        # Record input statistics
        stats['input_sizes'].append(image.shape)
        
        # Preprocess
        mask = masks[i] if masks else None
        result = preprocessor.preprocess(image, mask)
        
        # Record output statistics
        processed_image = result['image']
        stats['output_sizes'].append(processed_image.shape)
        stats['intensity_ranges'].append((processed_image.min(), processed_image.max()))
        stats['processing_times'].append(time.time() - start_time)
    
    # Compute summary statistics
    stats['avg_processing_time'] = np.mean(stats['processing_times'])
    stats['total_images'] = len(images)
    
    return stats


# Example usage and testing
if __name__ == "__main__":
    # Create a synthetic test image
    test_image = np.random.rand(256, 256) * 255
    test_image = test_image.astype(np.uint8)
    
    # Add some vessel-like structures
    center = (128, 128)
    for angle in np.linspace(0, 2*np.pi, 8):
        x = int(center[0] + 100 * np.cos(angle))
        y = int(center[1] + 100 * np.sin(angle))
        cv2.line(test_image, center, (x, y), 200, 2)
    
    print("🔬 Medical Image Preprocessing Test")
    print("=" * 40)
    
    # Test preprocessing
    preprocessor = MedicalImagePreprocessor(
        target_size=(512, 512),
        normalize_method='percentile',
        enhance_contrast=True,
        enhance_vessels=True
    )
    
    result = preprocessor.preprocess(test_image)
    processed_image = result['image']
    
    print(f"✅ Input shape: {test_image.shape}")
    print(f"✅ Output shape: {processed_image.shape}")
    print(f"✅ Output range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
    
    # Test specialized functions
    angio_processed = preprocess_angiography(test_image)
    print(f"✅ Angiography preprocessing: {angio_processed.shape}")
    
    # Test CT preprocessing (simulate HU values)
    ct_image = test_image.astype(np.float32) * 4 - 1024  # Simulate Hounsfield units
    ct_processed = preprocess_ct_coronary(ct_image)
    print(f"✅ CT preprocessing: {ct_processed.shape}")
    
    print("\n🎉 All preprocessing functions working correctly!")
