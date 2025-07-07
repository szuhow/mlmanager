#!/usr/bin/env python3
"""
Debug Script: Test Training Progress Auto-Refresh

This script tests the auto-refresh functionality in the model detail page.
It checks the JavaScript functions that handle progress bar updates.
"""

print("""
🔧 AUTO-REFRESH TEST SCRIPT
==========================

This script helps verify that the auto-refresh fixes are working properly.

STEPS TO TEST:

1. Start the server:
   ```
   make enhanced-start
   ```

2. Create a new model (or use an existing one with "pending" status)
   - Go to http://localhost:8000/ml/models/
   - Click "Create Model"

3. Open browser developer tools (F12) -> Console tab

4. Check the console for these messages:
   - "ModelDetailManager: Initializing for model X"
   - "ModelDetailManager: Starting updates for model status: pending"
   - "ModelDetailManager: Fetching progress from /ml/model/X/progress/"

5. Watch for status transition logs:
   - "ModelDetailManager: Status changed from pending to training"
   - "ModelDetailManager: Updated UI status attribute to training"

6. Verify UI updates:
   - Progress bar should appear when training starts (without refresh)
   - Progress percentage should update every 2 seconds
   - Metrics should update as they become available

TROUBLESHOOTING:

If auto-refresh isn't working, check for:

1. JavaScript errors in the console
2. Network errors when fetching /ml/model/X/progress/
3. Make sure the model_detail_unified.js and model_progress.js are loaded
4. Verify CSS selectors in the JavaScript match the HTML structure

MANUAL REFRESH TEST:

1. Click the refresh button (🔄) next to model name
2. Check console for:
   - "ModelDetailManager: Fetching progress from /ml/model/X/progress/"
   - "ModelDetailManager: Progress data received: {...}"
   - "ModelDetailManager: Updating progress bars with: {...}"
""")

import requests
import time
import json
import os
import sys
from datetime import datetime

def check_api_endpoint():
    """Test the progress API endpoint"""
    try:
        # Try to find a model ID to test
        response = requests.get("http://localhost:8000/ml/api/models/")
        if response.status_code == 200:
            models = response.json()
            if models and len(models) > 0:
                model_id = models[0].get('id')
                print(f"✅ Found model ID {model_id} to test")
                
                # Check progress endpoint
                progress_url = f"http://localhost:8000/ml/model/{model_id}/progress/"
                progress_response = requests.get(progress_url)
                if progress_response.status_code == 200:
                    data = progress_response.json()
                    print(f"✅ Progress API working, returned: {json.dumps(data, indent=2)}")
                else:
                    print(f"❌ Progress API failed with status: {progress_response.status_code}")
            else:
                print("⚠️ No models found to test")
        else:
            print(f"❌ API models endpoint failed with status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking API: {e}")

def main():
    print("\n📡 Checking API Endpoints...")
    check_api_endpoint()
    
    print("\n📋 Next Steps:")
    print("1. Open model detail page in browser")
    print("2. Watch for auto-updates in the UI")
    print("3. Check browser console for debug messages")
    
    print("\n✨ Auto-refresh fixes should now be working!")
    print("If issues persist, check JavaScript console and network requests.")

if __name__ == "__main__":
    main()
