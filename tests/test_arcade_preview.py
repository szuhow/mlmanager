#!/usr/bin/env python3

import os
import sys
import django
import json

# Add the core directory to the Python path
sys.path.insert(0, '/app/core')

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
django.setup()

from django.test import RequestFactory
from django.contrib.auth.models import User
from core.apps.ml_manager.views import dataset_preview_view

def test_arcade_preview():
    """Test ARCADE dataset preview functionality"""
    
    # Create a test user and request
    user = User.objects.get_or_create(username='testuser')[0]
    factory = RequestFactory()
    
    # Test ARCADE dataset path
    arcade_path = '/app/data/datasets/arcade_challenge_datasets/dataset_final_phase/test_case_segmentation'
    
    print("=== Testing ARCADE Dataset Preview ===")
    print(f"Testing path: {arcade_path}")
    print(f"Path exists: {os.path.exists(arcade_path)}")
    
    # Create a POST request
    request = factory.post('/ml/dataset-preview/', {
        'data_path': arcade_path,
        'dataset_type': 'arcade_binary'
    })
    request.user = user
    
    try:
        response = dataset_preview_view(request)
        print(f"Response status: {response.status_code}")
        
        # Try to extract key information from the response
        content = response.content.decode('utf-8')
        
        # Look for key indicators in the HTML
        indicators = {
            'detected_type': 'Detected Type:',
            'total_samples': 'Total Samples:',
            'image_count': 'Image_Count:',
            'sample_count': 'Sample_Count:',
            'arcade_info': 'arcade_task',
            'error_message': 'error'
        }
        
        results = {}
        for key, indicator in indicators.items():
            if indicator.lower() in content.lower():
                results[key] = True
                # Try to extract the value
                lines = content.split('\n')
                for line in lines:
                    if indicator.lower() in line.lower():
                        print(f"Found {key}: {line.strip()}")
                        break
            else:
                results[key] = False
        
        print(f"\\nResults summary: {results}")
        
        # Check if samples were generated
        if 'sample' in content.lower() and 'preview' in content.lower():
            print("✅ Samples appear to be generated")
        else:
            print("❌ No samples detected in response")
            
        return response.status_code == 200 and results.get('detected_type', False)
        
    except Exception as e:
        print(f"❌ Error calling view: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_arcade_preview()
    print(f"\\n{'✅ Test passed' if success else '❌ Test failed'}")
