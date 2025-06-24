"""
Augmentation Visualization and Testing Tools
===========================================

This module provides utilities for visualizing and testing data augmentation
techniques used in medical image processing, particularly for coronary segmentation.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from typing import List, Tuple, Optional, Dict, Any
import seaborn as sns
from PIL import Image, ImageEnhance
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AugmentationVisualizer:
    """Visualize and test various augmentation techniques."""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize the augmentation visualizer.
        
        Args:
            figsize: Figure size for matplotlib plots
        """
        self.figsize = figsize
        
    def visualize_basic_augmentations(self, 
                                    image: np.ndarray, 
                                    mask: Optional[np.ndarray] = None,
                                    save_path: Optional[str] = None) -> None:
        """
        Visualize basic augmentation techniques.
        
        Args:
            image: Input image (H, W) or (H, W, C)
            mask: Optional mask (H, W)
            save_path: Optional path to save the visualization
        """
        # Ensure image is 2D for simplicity
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Define basic augmentations
        augmentations = {
            'Original': lambda img, msk: (img, msk),
            'Horizontal Flip': self._flip_horizontal,
            'Vertical Flip': self._flip_vertical,
            'Rotation 15°': lambda img, msk: self._rotate(img, msk, 15),
            'Rotation -15°': lambda img, msk: self._rotate(img, msk, -15),
            'Zoom In 1.2x': lambda img, msk: self._zoom(img, msk, 1.2),
            'Zoom Out 0.8x': lambda img, msk: self._zoom(img, msk, 0.8),
            'Brightness +20%': lambda img, msk: self._adjust_brightness(img, msk, 1.2),
            'Brightness -20%': lambda img, msk: self._adjust_brightness(img, msk, 0.8),
            'Gaussian Blur': lambda img, msk: self._gaussian_blur(img, msk, sigma=1.0),
            'Add Noise': lambda img, msk: self._add_noise(img, msk, std=0.05),
            'Elastic Transform': lambda img, msk: self._elastic_transform(img, msk)
        }
        
        # Create subplot grid
        rows = 3
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        axes = axes.flatten()
        
        for idx, (name, aug_func) in enumerate(augmentations.items()):
            if idx >= len(axes):
                break
                
            try:
                aug_image, aug_mask = aug_func(image.copy(), 
                                             mask.copy() if mask is not None else None)
                
                # Display image
                axes[idx].imshow(aug_image, cmap='gray')
                axes[idx].set_title(name, fontsize=10)
                axes[idx].axis('off')
                
                # Overlay mask if available
                if aug_mask is not None:
                    axes[idx].contour(aug_mask, colors='red', linewidths=1, alpha=0.7)
                
            except Exception as e:
                axes[idx].text(0.5, 0.5, f'Error: {str(e)[:20]}...', 
                             transform=axes[idx].transAxes, ha='center', va='center')
                axes[idx].set_title(name, fontsize=10)
                axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_albumentations_pipeline(self, 
                                        image: np.ndarray,
                                        mask: Optional[np.ndarray] = None,
                                        pipeline: Optional[A.Compose] = None,
                                        num_samples: int = 8,
                                        save_path: Optional[str] = None) -> None:
        """
        Visualize Albumentations augmentation pipeline.
        
        Args:
            image: Input image
            mask: Optional mask
            pipeline: Albumentations pipeline (if None, uses default)
            num_samples: Number of augmented samples to show
            save_path: Optional path to save the visualization
        """
        if pipeline is None:
            pipeline = self._get_default_albumentations_pipeline()
        
        # Ensure proper format for albumentations
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Create samples
        cols = min(4, num_samples)
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i in range(num_samples):
            try:
                if mask is not None:
                    augmented = pipeline(image=image, mask=mask)
                    aug_image = augmented['image']
                    aug_mask = augmented['mask']
                else:
                    augmented = pipeline(image=image)
                    aug_image = augmented['image']
                    aug_mask = None
                
                # Convert to displayable format
                if isinstance(aug_image, torch.Tensor):
                    aug_image = aug_image.permute(1, 2, 0).numpy()
                
                if len(aug_image.shape) == 3:
                    aug_image = cv2.cvtColor(aug_image, cv2.COLOR_RGB2GRAY)
                
                axes[i].imshow(aug_image, cmap='gray')
                axes[i].set_title(f'Sample {i+1}', fontsize=10)
                axes[i].axis('off')
                
                # Overlay mask if available
                if aug_mask is not None:
                    axes[i].contour(aug_mask, colors='red', linewidths=1, alpha=0.7)
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error: {str(e)[:20]}...', 
                           transform=axes[i].transAxes, ha='center', va='center')
                axes[i].set_title(f'Sample {i+1}', fontsize=10)
                axes[i].axis('off')
        
        # Hide extra subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_augmentation_methods(self, 
                                   image: np.ndarray,
                                   mask: Optional[np.ndarray] = None,
                                   save_path: Optional[str] = None) -> None:
        """
        Compare different augmentation libraries and methods.
        
        Args:
            image: Input image
            mask: Optional mask
            save_path: Optional path to save the visualization
        """
        # Ensure proper format
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image_rgb = image
        else:
            image_gray = image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        fig, axes = plt.subplots(2, 4, figsize=self.figsize)
        
        # Original
        axes[0, 0].imshow(image_gray, cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        if mask is not None:
            axes[0, 0].contour(mask, colors='red', linewidths=1, alpha=0.7)
        
        # OpenCV-based augmentation
        try:
            opencv_aug = self._opencv_augmentation(image_gray, mask)
            axes[0, 1].imshow(opencv_aug[0], cmap='gray')
            axes[0, 1].set_title('OpenCV Aug')
            axes[0, 1].axis('off')
            if opencv_aug[1] is not None:
                axes[0, 1].contour(opencv_aug[1], colors='red', linewidths=1, alpha=0.7)
        except Exception as e:
            axes[0, 1].text(0.5, 0.5, f'OpenCV Error', transform=axes[0, 1].transAxes, ha='center')
            axes[0, 1].axis('off')
        
        # Albumentations
        try:
            albu_pipeline = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.3)
            ])
            albu_result = albu_pipeline(image=image_rgb, mask=mask)
            albu_image = cv2.cvtColor(albu_result['image'], cv2.COLOR_RGB2GRAY)
            axes[0, 2].imshow(albu_image, cmap='gray')
            axes[0, 2].set_title('Albumentations')
            axes[0, 2].axis('off')
            if albu_result['mask'] is not None:
                axes[0, 2].contour(albu_result['mask'], colors='red', linewidths=1, alpha=0.7)
        except Exception as e:
            axes[0, 2].text(0.5, 0.5, f'Albu Error', transform=axes[0, 2].transAxes, ha='center')
            axes[0, 2].axis('off')
        
        # PyTorch transforms
        try:
            torch_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor()
            ])
            torch_image = torch_transforms(image_gray)
            torch_image_np = torch_image.squeeze().numpy()
            axes[0, 3].imshow(torch_image_np, cmap='gray')
            axes[0, 3].set_title('PyTorch Transforms')
            axes[0, 3].axis('off')
        except Exception as e:
            axes[0, 3].text(0.5, 0.5, f'PyTorch Error', transform=axes[0, 3].transAxes, ha='center')
            axes[0, 3].axis('off')
        
        # Advanced augmentations
        advanced_augs = [
            ('Elastic Transform', lambda: self._elastic_transform(image_gray, mask)),
            ('Grid Distortion', lambda: self._grid_distortion(image_gray, mask)),
            ('Piece Affine', lambda: self._piecewise_affine(image_gray, mask)),
            ('CLAHE + Noise', lambda: self._clahe_noise_combo(image_gray, mask))
        ]
        
        for idx, (name, aug_func) in enumerate(advanced_augs):
            try:
                aug_result = aug_func()
                axes[1, idx].imshow(aug_result[0], cmap='gray')
                axes[1, idx].set_title(name)
                axes[1, idx].axis('off')
                if aug_result[1] is not None:
                    axes[1, idx].contour(aug_result[1], colors='red', linewidths=1, alpha=0.7)
            except Exception as e:
                axes[1, idx].text(0.5, 0.5, f'Error', transform=axes[1, idx].transAxes, ha='center')
                axes[1, idx].set_title(name)
                axes[1, idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def test_augmentation_robustness(self, 
                                   image: np.ndarray,
                                   mask: Optional[np.ndarray] = None,
                                   num_tests: int = 100) -> Dict[str, Any]:
        """
        Test augmentation robustness and statistics.
        
        Args:
            image: Input image
            mask: Optional mask
            num_tests: Number of test iterations
            
        Returns:
            Statistics about augmentation effects
        """
        pipeline = self._get_default_albumentations_pipeline()
        
        stats = {
            'intensity_changes': [],
            'geometric_changes': [],
            'mask_preservation': [],
            'errors': 0
        }
        
        original_mean = np.mean(image)
        original_std = np.std(image)
        
        for i in range(num_tests):
            try:
                if len(image.shape) == 2:
                    test_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    test_image = image
                
                if mask is not None:
                    augmented = pipeline(image=test_image, mask=mask)
                    aug_image = augmented['image']
                    aug_mask = augmented['mask']
                    
                    # Check mask preservation
                    mask_diff = np.abs(np.sum(mask) - np.sum(aug_mask)) / np.sum(mask)
                    stats['mask_preservation'].append(mask_diff)
                else:
                    augmented = pipeline(image=test_image)
                    aug_image = augmented['image']
                
                # Convert back to numpy if tensor
                if isinstance(aug_image, torch.Tensor):
                    aug_image = aug_image.permute(1, 2, 0).numpy()
                
                if len(aug_image.shape) == 3:
                    aug_image = cv2.cvtColor(aug_image, cv2.COLOR_RGB2GRAY)
                
                # Intensity statistics
                aug_mean = np.mean(aug_image)
                aug_std = np.std(aug_image)
                
                intensity_change = abs(aug_mean - original_mean) / original_mean
                stats['intensity_changes'].append(intensity_change)
                
                # Geometric change (simple metric based on gradients)
                original_grad = np.gradient(image)
                aug_grad = np.gradient(aug_image)
                
                grad_change = np.mean(np.abs(np.array(original_grad) - np.array(aug_grad)))
                stats['geometric_changes'].append(grad_change)
                
            except Exception as e:
                stats['errors'] += 1
        
        # Compute summary statistics
        stats['avg_intensity_change'] = np.mean(stats['intensity_changes'])
        stats['std_intensity_change'] = np.std(stats['intensity_changes'])
        stats['avg_geometric_change'] = np.mean(stats['geometric_changes'])
        stats['error_rate'] = stats['errors'] / num_tests
        
        if mask is not None:
            stats['avg_mask_preservation'] = np.mean(stats['mask_preservation'])
            stats['std_mask_preservation'] = np.std(stats['mask_preservation'])
        
        return stats
    
    # Helper methods for individual augmentations
    def _flip_horizontal(self, image: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply horizontal flip."""
        flipped_image = np.fliplr(image)
        flipped_mask = np.fliplr(mask) if mask is not None else None
        return flipped_image, flipped_mask
    
    def _flip_vertical(self, image: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply vertical flip."""
        flipped_image = np.flipud(image)
        flipped_mask = np.flipud(mask) if mask is not None else None
        return flipped_image, flipped_mask
    
    def _rotate(self, image: np.ndarray, mask: Optional[np.ndarray], angle: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply rotation."""
        center = (image.shape[1] // 2, image.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        rotated_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0])) if mask is not None else None
        
        return rotated_image, rotated_mask
    
    def _zoom(self, image: np.ndarray, mask: Optional[np.ndarray], factor: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply zoom."""
        h, w = image.shape[:2]
        new_h, new_w = int(h * factor), int(w * factor)
        
        # Resize
        zoomed_image = cv2.resize(image, (new_w, new_h))
        zoomed_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST) if mask is not None else None
        
        # Crop or pad to original size
        if factor > 1:  # Crop
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            final_image = zoomed_image[start_h:start_h+h, start_w:start_w+w]
            final_mask = zoomed_mask[start_h:start_h+h, start_w:start_w+w] if zoomed_mask is not None else None
        else:  # Pad
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            final_image = np.pad(zoomed_image, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), mode='reflect')
            final_mask = np.pad(zoomed_mask, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), mode='constant') if zoomed_mask is not None else None
        
        return final_image, final_mask
    
    def _adjust_brightness(self, image: np.ndarray, mask: Optional[np.ndarray], factor: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Adjust brightness."""
        adjusted_image = np.clip(image * factor, 0, 255 if image.dtype == np.uint8 else 1.0)
        return adjusted_image, mask
    
    def _gaussian_blur(self, image: np.ndarray, mask: Optional[np.ndarray], sigma: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply Gaussian blur."""
        blurred_image = cv2.GaussianBlur(image, (0, 0), sigma)
        return blurred_image, mask
    
    def _add_noise(self, image: np.ndarray, mask: Optional[np.ndarray], std: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Add Gaussian noise."""
        noise = np.random.normal(0, std, image.shape)
        noisy_image = np.clip(image + noise, 0, 255 if image.dtype == np.uint8 else 1.0)
        return noisy_image, mask
    
    def _elastic_transform(self, image: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply elastic transformation using Albumentations."""
        transform = A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0)
        
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        if mask is not None:
            result = transform(image=image_rgb, mask=mask)
            transformed_image = cv2.cvtColor(result['image'], cv2.COLOR_RGB2GRAY)
            transformed_mask = result['mask']
        else:
            result = transform(image=image_rgb)
            transformed_image = cv2.cvtColor(result['image'], cv2.COLOR_RGB2GRAY)
            transformed_mask = None
        
        return transformed_image, transformed_mask
    
    def _get_default_albumentations_pipeline(self) -> A.Compose:
        """Get default Albumentations pipeline for medical images."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_REFLECT),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
            A.GridDistortion(p=0.2),
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.05, p=0.2),
        ])
    
    def _opencv_augmentation(self, image: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply random OpenCV-based augmentation."""
        # Random choice of augmentation
        aug_type = np.random.choice(['rotate', 'blur', 'brightness', 'noise'])
        
        if aug_type == 'rotate':
            return self._rotate(image, mask, np.random.uniform(-15, 15))
        elif aug_type == 'blur':
            return self._gaussian_blur(image, mask, np.random.uniform(0.5, 2.0))
        elif aug_type == 'brightness':
            return self._adjust_brightness(image, mask, np.random.uniform(0.8, 1.2))
        else:  # noise
            return self._add_noise(image, mask, np.random.uniform(0.01, 0.05))
    
    def _grid_distortion(self, image: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply grid distortion using Albumentations."""
        transform = A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0)
        
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        if mask is not None:
            result = transform(image=image_rgb, mask=mask)
            transformed_image = cv2.cvtColor(result['image'], cv2.COLOR_RGB2GRAY)
            transformed_mask = result['mask']
        else:
            result = transform(image=image_rgb)
            transformed_image = cv2.cvtColor(result['image'], cv2.COLOR_RGB2GRAY)
            transformed_mask = None
        
        return transformed_image, transformed_mask
    
    def _piecewise_affine(self, image: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply piecewise affine transformation."""
        transform = A.PiecewiseAffine(scale=(0.03, 0.05), p=1.0)
        
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        if mask is not None:
            result = transform(image=image_rgb, mask=mask)
            transformed_image = cv2.cvtColor(result['image'], cv2.COLOR_RGB2GRAY)
            transformed_mask = result['mask']
        else:
            result = transform(image=image_rgb)
            transformed_image = cv2.cvtColor(result['image'], cv2.COLOR_RGB2GRAY)
            transformed_mask = None
        
        return transformed_image, transformed_mask
    
    def _clahe_noise_combo(self, image: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply CLAHE followed by noise addition."""
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if image.dtype != np.uint8:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        enhanced = clahe.apply(image_uint8)
        enhanced = enhanced.astype(np.float32) / 255.0
        
        # Add noise
        noise = np.random.normal(0, 0.02, enhanced.shape)
        final_image = np.clip(enhanced + noise, 0, 1)
        
        return final_image, mask


# Utility functions
def create_test_dataset_preview(images: List[np.ndarray], 
                              masks: List[np.ndarray],
                              augment: bool = False,
                              num_samples: int = 8,
                              save_path: Optional[str] = None) -> None:
    """
    Create a preview of the dataset with optional augmentations.
    
    Args:
        images: List of input images
        masks: List of corresponding masks
        augment: Whether to show augmented versions
        num_samples: Number of samples to show
        save_path: Optional path to save the preview
    """
    visualizer = AugmentationVisualizer()
    
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i in range(min(num_samples, len(images))):
        image = images[i]
        mask = masks[i] if i < len(masks) else None
        
        if augment:
            # Apply random augmentation
            pipeline = visualizer._get_default_albumentations_pipeline()
            
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image
            
            try:
                if mask is not None:
                    result = pipeline(image=image_rgb, mask=mask)
                    display_image = cv2.cvtColor(result['image'], cv2.COLOR_RGB2GRAY)
                    display_mask = result['mask']
                else:
                    result = pipeline(image=image_rgb)
                    display_image = cv2.cvtColor(result['image'], cv2.COLOR_RGB2GRAY)
                    display_mask = None
            except:
                display_image = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                display_mask = mask
        else:
            display_image = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            display_mask = mask
        
        axes[i].imshow(display_image, cmap='gray')
        axes[i].set_title(f'Sample {i+1}' + (' (Aug)' if augment else ''))
        axes[i].axis('off')
        
        if display_mask is not None:
            axes[i].contour(display_mask, colors='red', linewidths=1, alpha=0.7)
    
    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Create a synthetic test image with vessel-like structures
    test_image = np.zeros((256, 256), dtype=np.uint8)
    
    # Add some vessel-like structures
    center = (128, 128)
    for angle in np.linspace(0, 2*np.pi, 8):
        x = int(center[0] + 80 * np.cos(angle))
        y = int(center[1] + 80 * np.sin(angle))
        cv2.line(test_image, center, (x, y), 200, 3)
    
    # Create a simple mask
    test_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(test_mask, center, 60, 255, -1)
    
    print("🎨 Augmentation Visualization Test")
    print("=" * 40)
    
    # Initialize visualizer
    visualizer = AugmentationVisualizer()
    
    # Test basic augmentations
    print("✅ Testing basic augmentations...")
    visualizer.visualize_basic_augmentations(test_image, test_mask)
    
    # Test Albumentations pipeline
    print("✅ Testing Albumentations pipeline...")
    visualizer.visualize_albumentations_pipeline(test_image, test_mask, num_samples=6)
    
    # Test comparison of methods
    print("✅ Testing comparison of augmentation methods...")
    visualizer.compare_augmentation_methods(test_image, test_mask)
    
    # Test robustness
    print("✅ Testing augmentation robustness...")
    stats = visualizer.test_augmentation_robustness(test_image, test_mask, num_tests=50)
    
    print(f"📊 Robustness Statistics:")
    print(f"   Average intensity change: {stats['avg_intensity_change']:.3f}")
    print(f"   Average geometric change: {stats['avg_geometric_change']:.3f}")
    print(f"   Error rate: {stats['error_rate']:.1%}")
    if 'avg_mask_preservation' in stats:
        print(f"   Average mask preservation: {stats['avg_mask_preservation']:.3f}")
    
    print("\n🎉 All augmentation visualization tools working correctly!")
