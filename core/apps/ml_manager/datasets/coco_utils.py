#!/usr/bin/env python3
"""
COCO to Mask utilities for ARCADE dataset preview
"""

import json
import numpy as np
from PIL import Image, ImageDraw
import os
from typing import Optional, Tuple, List, Dict, Any

def coco_polygon_to_mask(polygon: List[float], image_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert COCO polygon annotation to binary mask
    
    Args:
        polygon: List of x,y coordinates [x1,y1,x2,y2,...]
        image_size: (width, height) of the image
        
    Returns:
        Binary mask as numpy array
    """
    width, height = image_size
    
    # Create PIL Image for drawing
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    # Convert flat list to list of tuples
    if len(polygon) >= 6:  # At least 3 points (x,y pairs)
        coords = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
        draw.polygon(coords, fill=255)
    
    return np.array(img)

def coco_rle_to_mask(rle: Dict[str, Any], image_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert COCO RLE annotation to binary mask
    
    Args:
        rle: RLE annotation with 'size' and 'counts'
        image_size: (width, height) of the image
        
    Returns:
        Binary mask as numpy array
    """
    try:
        # Try to use pycocotools if available
        from pycocotools import mask as maskUtils
        
        # Ensure RLE is in correct format
        if isinstance(rle.get('counts'), list):
            # Uncompressed RLE
            rle_formatted = {
                'size': rle['size'],
                'counts': rle['counts']
            }
        else:
            # Already compressed
            rle_formatted = rle
        
        mask = maskUtils.decode(rle_formatted)
        return mask * 255  # Convert to 0-255 range
        
    except ImportError:
        # Fallback implementation without pycocotools
        width, height = image_size
        
        if 'counts' in rle and isinstance(rle['counts'], list):
            # Uncompressed RLE - list of counts
            counts = rle['counts']
            mask = np.zeros(width * height, dtype=np.uint8)
            
            idx = 0
            value = 0  # Start with background
            for count in counts:
                mask[idx:idx+count] = value * 255
                idx += count
                value = 1 - value  # Toggle between 0 and 1
                
            return mask.reshape((height, width))
        else:
            # Create empty mask if can't decode
            return np.zeros((height, width), dtype=np.uint8)

def generate_mask_from_coco_annotation(
    annotation: Dict[str, Any], 
    image_size: Tuple[int, int]
) -> Optional[np.ndarray]:
    """
    Generate mask from a single COCO annotation
    
    Args:
        annotation: COCO annotation dictionary
        image_size: (width, height) of the image
        
    Returns:
        Binary mask as numpy array or None if failed
    """
    try:
        segmentation = annotation.get('segmentation')
        
        if not segmentation:
            return None
        
        # Handle different segmentation formats
        if isinstance(segmentation, list):
            # Polygon format
            if len(segmentation) > 0 and isinstance(segmentation[0], list):
                # Multiple polygons
                width, height = image_size
                combined_mask = np.zeros((height, width), dtype=np.uint8)
                
                for polygon in segmentation:
                    if len(polygon) >= 6:  # At least 3 points
                        mask = coco_polygon_to_mask(polygon, image_size)
                        combined_mask = np.maximum(combined_mask, mask)
                
                return combined_mask
            else:
                # Single polygon
                return coco_polygon_to_mask(segmentation, image_size)
                
        elif isinstance(segmentation, dict):
            # RLE format
            return coco_rle_to_mask(segmentation, image_size)
        
        return None
        
    except Exception as e:
        print(f"Error generating mask from annotation: {e}")
        return None

def generate_mask_from_coco_file(
    coco_file_path: str,
    image_filename: str,
    output_dir: str,
    category_id: Optional[int] = None
) -> Optional[str]:
    """
    Generate mask from COCO annotation file for a specific image
    
    Args:
        coco_file_path: Path to COCO JSON file
        image_filename: Name of the image file
        output_dir: Directory to save generated mask
        category_id: Specific category ID to generate mask for (None for all)
        
    Returns:
        Path to generated mask file or None if failed
    """
    try:
        # Load COCO annotations
        with open(coco_file_path, 'r') as f:
            coco_data = json.load(f)
        
        # Find image info
        image_info = None
        for img in coco_data.get('images', []):
            if img['file_name'] == image_filename:
                image_info = img
                break
        
        if not image_info:
            print(f"Image {image_filename} not found in COCO file")
            return None
        
        image_id = image_info['id']
        image_size = (image_info['width'], image_info['height'])
        
        # Find annotations for this image
        annotations = [
            ann for ann in coco_data.get('annotations', [])
            if ann['image_id'] == image_id and 
            (category_id is None or ann.get('category_id') == category_id)
        ]
        
        if not annotations:
            print(f"No annotations found for image {image_filename}")
            return None
        
        # Generate combined mask
        width, height = image_size
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        for annotation in annotations:
            mask = generate_mask_from_coco_annotation(annotation, image_size)
            if mask is not None:
                combined_mask = np.maximum(combined_mask, mask)
        
        if np.sum(combined_mask) == 0:
            print(f"Generated mask is empty for {image_filename}")
            return None
        
        # Save mask
        os.makedirs(output_dir, exist_ok=True)
        mask_filename = os.path.splitext(image_filename)[0] + '_mask.png'
        mask_path = os.path.join(output_dir, mask_filename)
        
        mask_image = Image.fromarray(combined_mask, mode='L')
        mask_image.save(mask_path)
        
        print(f"Generated mask: {mask_path}")
        return mask_path
        
    except Exception as e:
        print(f"Error generating mask from COCO file: {e}")
        return None

def generate_preview_masks_from_coco(
    dataset_path: str,
    sample_images: List[str],
    output_dir: str = '/tmp/preview_masks'
) -> Dict[str, str]:
    """
    Generate preview masks for sample images from COCO dataset
    
    Args:
        dataset_path: Path to dataset containing images and annotations
        sample_images: List of image file paths
        output_dir: Directory to save generated masks
        
    Returns:
        Dictionary mapping image paths to generated mask paths
    """
    mask_mapping = {}
    
    # Find COCO annotation file
    annotations_dir = os.path.join(dataset_path, 'annotations')
    if not os.path.exists(annotations_dir):
        print(f"Annotations directory not found: {annotations_dir}")
        return mask_mapping
    
    json_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
    if not json_files:
        print(f"No JSON files found in {annotations_dir}")
        return mask_mapping
    
    coco_file = os.path.join(annotations_dir, json_files[0])
    print(f"Using COCO file: {coco_file}")
    
    # Generate masks for sample images
    for image_path in sample_images:
        image_filename = os.path.basename(image_path)
        
        try:
            mask_path = generate_mask_from_coco_file(
                coco_file, 
                image_filename, 
                output_dir
            )
            
            if mask_path:
                mask_mapping[image_path] = mask_path
                
        except Exception as e:
            print(f"Error generating mask for {image_filename}: {e}")
    
    return mask_mapping

if __name__ == "__main__":
    # Test the functions
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        print(f"Testing COCO mask generation for: {dataset_path}")
        
        # Test with a sample image
        images_dir = os.path.join(dataset_path, 'images')
        if os.path.exists(images_dir):
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                sample_image = os.path.join(images_dir, image_files[0])
                print(f"Testing with: {sample_image}")
                
                masks = generate_preview_masks_from_coco(dataset_path, [sample_image])
                print(f"Generated masks: {masks}")
            else:
                print("No image files found")
        else:
            print(f"Images directory not found: {images_dir}")
    else:
        print("Usage: python coco_utils.py <dataset_path>")
