#!/usr/bin/env python3

import os
import sys
import django

# Add the core directory to the Python path
sys.path.insert(0, '/app/core')

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
django.setup()

from django.test import RequestFactory
from django.contrib.auth.models import User
from core.apps.ml_manager.views import dataset_preview_view

def test_ui_values():
    """Test specific UI values that should be displayed"""
    
    # Create a test user and request
    user = User.objects.get_or_create(username='testuser')[0]
    factory = RequestFactory()
    
    # Test ARCADE dataset path
    arcade_path = '/app/data/datasets/arcade_challenge_datasets/dataset_final_phase/test_case_segmentation'
    
    print("=== Testing UI Values ===")
    print(f"Testing path: {arcade_path}")
    
    # Create a POST request
    request = factory.post('/ml/dataset-preview/', {
        'data_path': arcade_path,
        'dataset_type': 'arcade_binary'
    })
    request.user = user
    
    try:
        response = dataset_preview_view(request)
        content = response.content.decode('utf-8')
        
        print(f"Response status: {response.status_code}")
        
        # Look for specific patterns in the response
        tests = [
            ('Binary Segmentation badge', 'Binary Segmentation'),
            ('Total samples value', 'Total Samples'),
            ('Image count 600', '600'),
            ('ARCADE task info', 'binary_segmentation'),
            ('Sample count 6', 'Sample_Count.*?6'),
            ('Image shape 512x512', '512x512'),
            ('COCO format', 'COCO')
        ]
        
        for test_name, pattern in tests:
            if pattern.lower() in content.lower():
                print(f"✅ {test_name}: FOUND")
            else:
                print(f"❌ {test_name}: NOT FOUND")
        
        # Extract the detected type section
        import re
        detected_match = re.search(r'Detected Type:.*?<span[^>]*>(.*?)</span>', content, re.IGNORECASE | re.DOTALL)
        if detected_match:
            detected_text = detected_match.group(1).strip()
            print(f"\\n🎯 Detected Type Display: '{detected_text}'")
        
        # Extract total samples
        samples_match = re.search(r'Total Samples:.*?<span[^>]*>(.*?)</span>', content, re.IGNORECASE | re.DOTALL)
        if samples_match:
            samples_text = samples_match.group(1).strip()
            print(f"🎯 Total Samples Display: '{samples_text}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_ui_values()
    print(f"\\n{'✅ UI Test OK' if success else '❌ UI Test Failed'}")
