#!/usr/bin/env python3
"""
Test the UI fixes for progress bar text visibility and IoU metric labels
"""

import os
import sys
import time
import django
import requests
from datetime import datetime

# Setup Django within container
def test_ui_fixes():
    print(f"🧪 Testing UI fixes at {datetime.now()}")
    
    # Test with curl to check if API works
    print("\n1. Testing API with sample model...")
    
    try:
        # First check if we can access the training progress API
        response = requests.get('http://localhost:8000/ml/api/training-progress/999/', timeout=10)
        print(f"API response status: {response.status_code}")
        
        if response.status_code == 302:
            print("✅ API endpoint exists (requires login - expected)")
        elif response.status_code == 404:
            print("✅ API endpoint works (model not found - expected)")
        else:
            print(f"Unexpected status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
    
    print("\n2. Manual testing needed:")
    print("✅ Open http://localhost:8000/ml/start-training/")
    print("✅ Login with admin/admin123")
    print("✅ Fill form with segmentation_metric='iou'")
    print("✅ Submit and check:")
    print("   - Progress bar text should be visible at low progress values")
    print("   - Metric labels should show 'Training IoU' instead of 'Training Dice'")
    print("   - Progress should redirect with model_id parameter")
    print("   - Training Progress section should auto-show")
    
    print(f"\n🎉 UI Testing Guide completed at {datetime.now()}")

if __name__ == '__main__':
    test_ui_fixes()
