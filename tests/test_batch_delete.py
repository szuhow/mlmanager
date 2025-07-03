#!/usr/bin/env python3
"""
Test batch deletion functionality
"""

import requests
import sys

# Test batch deletion via API
def test_batch_deletion():
    # Get CSRF token first
    session = requests.Session()
    
    # Get the models page to extract CSRF token
    response = session.get('http://localhost:8000/ml/models/')
    if response.status_code != 200:
        print(f"Failed to access models page: {response.status_code}")
        return False
    
    # Extract CSRF token from response
    csrf_token = None
    for line in response.text.split('\n'):
        if 'csrfmiddlewaretoken' in line and 'value=' in line:
            # Extract token from input field
            start = line.find('value="') + 7
            end = line.find('"', start)
            if start > 6 and end > start:
                csrf_token = line[start:end]
                break
    
    if not csrf_token:
        print("Could not extract CSRF token")
        return False
    
    print(f"CSRF token: {csrf_token[:10]}...")
    
    # Test with model ID 90 (our test model)
    data = {
        'model_ids': ['90'],
        'csrfmiddlewaretoken': csrf_token
    }
    
    response = session.post(
        'http://localhost:8000/ml/models/batch-delete/',
        data=data,
        headers={
            'X-CSRFToken': csrf_token,
            'Referer': 'http://localhost:8000/ml/models/'
        }
    )
    
    print(f"Batch delete response status: {response.status_code}")
    print(f"Response content: {response.text[:500]}")
    
    return response.status_code == 200

if __name__ == '__main__':
    print("Testing batch deletion...")
    success = test_batch_deletion()
    print(f"Test {'PASSED' if success else 'FAILED'}")
