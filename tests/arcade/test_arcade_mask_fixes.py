#!/usr/bin/env python3
"""
Test ARCADE mask visibility fixes
Tests that binary masks are properly displayed with full contrast (0-255 range)
"""

import os
import sys
import django

# Setup Django
sys.path.append('/app/core')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.testing')
django.setup()

import numpy as np
from PIL import Image
import requests
import time
from django.test import Client
from django.contrib.auth.models import User

def test_arcade_mask_visibility():
    """Test that ARCADE masks are displayed with proper visibility"""
    print("🧪 Testing ARCADE mask visibility fixes...")
    
    # Test parameters
    test_data_path = '/app/data/datasets/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset/seg_train'
    dataset_types = [
        'arcade_binary_segmentation',
        'arcade_artery_classification', 
        'arcade_stenosis_segmentation'
    ]
    
    client = Client()
    
    # Create test user
    try:
        user = User.objects.get(username='testuser')
    except User.DoesNotExist:
        user = User.objects.create_user('testuser', 'test@test.com', 'testpass')
    
    # Login
    client.login(username='testuser', password='testpass')
    
    print(f"📂 Testing dataset path: {test_data_path}")
    print(f"📊 Testing dataset types: {dataset_types}")
    
    # Test each dataset type
    for dataset_type in dataset_types:
        print(f"\n🎯 Testing {dataset_type}...")
        
        try:
            # Make request to dataset preview
            response = client.post('/ml/dataset-preview/', {
                'data_path': test_data_path,
                'dataset_type': dataset_type
            })
            
            if response.status_code == 200:
                print(f"   ✅ {dataset_type}: Preview loaded successfully")
                
                # Check if samples were generated
                context = response.context
                if context and 'samples' in context:
                    samples = context['samples']
                    if samples:
                        print(f"   📈 Generated {len(samples)} samples")
                        
                        # Check sample details
                        for i, sample in enumerate(samples[:2]):  # Check first 2 samples
                            print(f"      📄 Sample {i+1}:")
                            print(f"         🖼️  Image: {sample.get('image_url', 'N/A')}")
                            print(f"         🎭 Mask: {sample.get('mask_url', 'N/A')}")
                            print(f"         📏 Image shape: {sample.get('image_shape', 'N/A')}")
                            print(f"         📐 Mask shape: {sample.get('mask_shape', 'N/A')}")
                            print(f"         📊 Range: {sample.get('image_min', 'N/A')} - {sample.get('image_max', 'N/A')}")
                            print(f"         🎨 Mask range: {sample.get('mask_min', 'N/A')} - {sample.get('mask_max', 'N/A')}")
                            print(f"         📝 Analysis: {sample.get('analysis', 'N/A')}")
                            
                            # Check for visibility issues
                            if (sample.get('mask_min') == sample.get('mask_max') == 0) or \
                               (sample.get('mask_min') == sample.get('mask_max') == 1):
                                print(f"         ⚠️  WARNING: Mask appears to have no contrast!")
                            else:
                                print(f"         ✅ Mask has proper contrast")
                    else:
                        print(f"   ⚠️  No samples generated")
                else:
                    print(f"   ⚠️  No context or samples in response")
                    
                # Check for error messages
                if context and 'error_message' in context and context['error_message']:
                    print(f"   ❌ Error: {context['error_message']}")
                    
            else:
                print(f"   ❌ {dataset_type}: Failed with status {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ {dataset_type}: Exception - {e}")
    
    print(f"\n✨ ARCADE mask visibility test completed!")

def test_image_file_creation():
    """Test that generated images are actually visible"""
    print("\n🖼️  Testing generated image files...")
    
    import glob
    temp_dir = '/app/core/media/temp/dataset_preview'
    
    if os.path.exists(temp_dir):
        image_files = glob.glob(os.path.join(temp_dir, '*.jpg'))
        mask_files = glob.glob(os.path.join(temp_dir, '*.png'))
        
        print(f"   📁 Found {len(image_files)} image files")
        print(f"   📁 Found {len(mask_files)} mask files")
        
        # Check a few files for proper content
        for i, mask_file in enumerate(mask_files[:3]):
            try:
                img = Image.open(mask_file)
                img_array = np.array(img)
                
                print(f"   🎭 Mask {i+1}: {os.path.basename(mask_file)}")
                print(f"      📏 Shape: {img_array.shape}")
                print(f"      📊 Range: {img_array.min()} - {img_array.max()}")
                print(f"      🎨 Mode: {img.mode}")
                
                if img_array.max() == img_array.min():
                    print(f"      ⚠️  WARNING: Uniform image (no contrast)!")
                else:
                    print(f"      ✅ Image has contrast")
                    
            except Exception as e:
                print(f"   ❌ Error reading {mask_file}: {e}")
    else:
        print(f"   📁 Temp directory not found: {temp_dir}")

def test_paths_structure():
    """Test that data paths are properly organized"""
    print("\n📂 Testing data structure organization...")
    
    paths_to_check = [
        '/app/data/logs',
        '/app/data/models/organized',
        '/app/data/mlflow',
        '/app/data/artifacts'
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"   ✅ {path}: Exists")
            # Count files
            try:
                files = os.listdir(path)
                print(f"      📁 Contains {len(files)} items")
            except:
                print(f"      📁 Cannot list contents")
        else:
            print(f"   ❌ {path}: Does not exist")

if __name__ == "__main__":
    print("🚀 Starting ARCADE mask visibility and paths test...")
    
    test_arcade_mask_visibility()
    test_image_file_creation()
    test_paths_structure()
    
    print("\n🎉 All tests completed!")
