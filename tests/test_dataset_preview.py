#!/usr/bin/env python3

import os
import sys
import django

# Add the core directory to the Python path
sys.path.insert(0, '/app/core')

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
django.setup()

# Now import our function
from core.apps.ml_manager.views import detect_dataset_type
import glob

def test_dataset_detection():
    """Test dataset detection on ARCADE dataset"""
    
    base_path = '/app/data/datasets/'
    arcade_path = '/app/data/datasets/arcade_challenge_datasets/dataset_final_phase/test_case_segmentation'
    
    print("=== Dataset Detection Test ===")
    print(f"Base path: {base_path}")
    print(f"Base path exists: {os.path.exists(base_path)}")
    
    if os.path.exists(base_path):
        print(f"Base path contents: {os.listdir(base_path)}")
    
    print(f"\nARCADE path: {arcade_path}")
    print(f"ARCADE path exists: {os.path.exists(arcade_path)}")
    
    if os.path.exists(arcade_path):
        print(f"ARCADE path contents: {os.listdir(arcade_path)}")
        
        # Check images directory
        images_dir = os.path.join(arcade_path, 'images')
        if os.path.exists(images_dir):
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Images found: {len(image_files)}")
            if image_files:
                print(f"Sample images: {image_files[:5]}")
        
        # Check annotations directory
        annotations_dir = os.path.join(arcade_path, 'annotations')
        if os.path.exists(annotations_dir):
            annotation_files = os.listdir(annotations_dir)
            print(f"Annotation files: {annotation_files}")
        
        # Test detection
        try:
            detected_type = detect_dataset_type(arcade_path)
            print(f"\n*** Detected type: {detected_type} ***")
        except Exception as e:
            print(f"Error in detection: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    test_dataset_detection()
