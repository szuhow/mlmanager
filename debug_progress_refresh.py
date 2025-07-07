#!/usr/bin/env python3
"""
Debug Progress Bar Update Issue
"""

print("""
🔧 DEBUG: Progress Bar Auto-Refresh Issue
==========================================

PROBLEM: Progress bars nie odświeżają się automatycznie

STEPS TO DEBUG:

1. Open browser to: http://localhost:8000/ml/model/89/
2. Open Developer Tools (F12) -> Console tab
3. Look for these messages in console:

INITIALIZATION:
✅ "ModelDetailManager: Initializing for model 89"
✅ "Elements found: {updateStatus: true, progressBar: true, ...}"
✅ "ModelDetailManager: Model status is training, not starting auto-updates" 
   OR
✅ "ModelDetailManager: Starting updates for model status: training"

IF MODEL STATUS IS 'training':
✅ "ModelDetailManager: Starting updates for model status: training"
✅ "ModelDetailManager: Fetching progress from /ml/model/89/progress/"
✅ "ModelDetailManager: Progress response status: 200"
✅ "ModelDetailManager: Progress data received: {...}"

MANUAL REFRESH TEST:
1. Click the refresh button (🔄) next to "Training Progress"
2. Should see in console:
   ✅ "ModelDetailManager: Fetching progress from /ml/model/89/progress/"
   ✅ "ModelDetailManager: Progress data received: {...}"
   ✅ "ModelDetailManager: Updating progress bars with: {...}"
   ✅ "Updated main progress bar: ..."

POSSIBLE ISSUES:

1. MODEL STATUS CHECK:
   - If console shows "Model status is completed/failed, not starting auto-updates"
   - The model is not training, so auto-refresh won't start
   - Solution: Start training a new model or change model status to 'training' in DB

2. ELEMENT SELECTORS:
   - If console shows "Elements found: {progressBar: false, ...}"
   - The CSS selectors aren't finding the progress bar elements
   - Check if HTML structure matches selectors

3. API ERRORS:
   - If console shows "Progress response status: 404/500"
   - API endpoint has issues
   - Check Django logs for errors

4. JAVASCRIPT ERRORS:
   - If console shows any JavaScript errors
   - Fix JavaScript syntax/logic errors

QUICK FIX TO TEST:
In browser console, run:
```
window.modelDetailManager.manualRefresh()
```

This should trigger one manual update and show debug logs.

API TEST:
Open in new tab: http://localhost:8000/ml/model/89/progress/
Should return JSON with progress data.

HTML CHECK:
Inspect element on progress bar, should have:
- class="progress-bar" inside div with id="training-progress"
- Elements with IDs: current-epoch, train-loss, val-dice, etc.
""")

import subprocess
import time

print("\n🔍 Quick Status Check:")

# Check if model 89 exists and its status
try:
    result = subprocess.run([
        "sqlite3", "data/db.sqlite3", 
        "SELECT id, name, status, current_epoch, total_epochs FROM ml_manager_mlmodel WHERE id = 89;"
    ], capture_output=True, text=True, timeout=10)
    
    if result.stdout.strip():
        print(f"✅ Model 89 data: {result.stdout.strip()}")
    else:
        print("❌ Model 89 not found in database")
        
except Exception as e:
    print(f"❌ Database check failed: {e}")

# Check if Django is running
try:
    import requests
    response = requests.get("http://localhost:8000/ml/model/89/progress/", timeout=5)
    print(f"✅ API endpoint status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ API returns: status={data.get('status')}, model_status={data.get('model_status')}")
    else:
        print(f"❌ API error: {response.status_code}")
except Exception as e:
    print(f"❌ API check failed: {e}")

print("\n📋 Next Steps:")
print("1. Open the model detail page in browser")
print("2. Check browser console for debug messages")
print("3. Try manual refresh button")
print("4. If model status is not 'training', create a new training model")
