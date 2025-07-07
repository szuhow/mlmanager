#!/usr/bin/env python3
"""
Test if progress auto-refresh works by checking browser console logs
"""

import requests
import time
import sys

def test_auto_refresh():
    """Test if auto-refresh is working"""
    
    BASE_URL = "http://localhost:8000"
    LOGIN_URL = f"{BASE_URL}/accounts/login/"
    
    print("🔧 Testing Auto-Refresh with Enhanced Logging")
    print("=" * 50)
    
    # Create session for maintaining login
    session = requests.Session()
    
    # Step 1: Login
    print("🔐 Logging in...")
    login_page = session.get(LOGIN_URL)
    if login_page.status_code != 200:
        print(f"❌ Failed to get login page: {login_page.status_code}")
        return False
    
    csrf_token = None
    for line in login_page.text.split('\n'):
        if 'csrfmiddlewaretoken' in line and 'value=' in line:
            csrf_token = line.split('value="')[1].split('"')[0]
            break
    
    if not csrf_token:
        print("❌ Could not find CSRF token")
        return False
    
    login_data = {
        'username': 'admin',
        'password': 'admin123',
        'csrfmiddlewaretoken': csrf_token
    }
    
    login_response = session.post(LOGIN_URL, data=login_data)
    if login_response.status_code not in [200, 302]:
        print(f"❌ Login failed: {login_response.status_code}")
        return False
    
    print("✅ Login successful")
    
    # Find a training model (status = training)
    print("🔍 Looking for training models...")
    
    # Check recent models for training status
    test_model_ids = [89, 88, 87, 86, 85]  # Check recent models
    training_model = None
    
    for model_id in test_model_ids:
        progress_url = f"{BASE_URL}/ml/model/{model_id}/progress/"
        try:
            progress_response = session.get(progress_url)
            if progress_response.status_code == 200:
                data = progress_response.json()
                model_status = data.get('model_status', 'unknown')
                print(f"   Model {model_id}: status = {model_status}")
                
                if model_status in ['training', 'pending']:
                    training_model = model_id
                    print(f"✅ Found training model: {model_id}")
                    break
        except:
            continue
    
    if not training_model:
        print("⚠️  No training models found. Testing with completed model...")
        training_model = 89  # Use any model for testing
    
    # Test auto-refresh by making multiple API calls
    print(f"\n🧪 Testing auto-refresh simulation for model {training_model}")
    print("Making 3 consecutive API calls to simulate auto-refresh...")
    
    for i in range(3):
        print(f"\n📡 API Call {i+1}/3")
        progress_url = f"{BASE_URL}/ml/model/{training_model}/progress/"
        
        start_time = time.time()
        try:
            response = session.get(progress_url)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                response_time = (end_time - start_time) * 1000
                
                print(f"   ✅ Response time: {response_time:.1f}ms")
                print(f"   📊 Status: {data.get('model_status', 'unknown')}")
                
                if 'progress' in data:
                    progress = data['progress']
                    print(f"   🎯 Epoch: {progress.get('current_epoch', 0)}/{progress.get('total_epochs', 0)}")
                    print(f"   📈 Progress: {progress.get('progress_percentage', 0):.1f}%")
                    
                if 'metrics' in data:
                    metrics = data['metrics']
                    print(f"   🔢 Train Loss: {metrics.get('train_loss', 0):.4f}")
                    print(f"   🔢 Val Dice: {metrics.get('val_dice', 0):.4f}")
                
            else:
                print(f"   ❌ API call failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
        
        if i < 2:  # Don't sleep after last call
            print("   ⏱️  Waiting 2 seconds (simulating auto-refresh interval)...")
            time.sleep(2)
    
    print("\n📋 Instructions for Manual Testing:")
    print(f"1. Open browser to: {BASE_URL}/ml/model/{training_model}/")
    print("2. Open Developer Tools (F12) -> Console")
    print("3. Look for these log messages:")
    print("   - 'ModelDetailManager: Initializing for model X'")
    print("   - 'ModelDetailManager: Starting updates for model status: training'")
    print("   - 'ModelDetailManager: Fetching progress from /ml/model/X/progress/'")
    print("   - 'ModelDetailManager: Progress data received: {...}'")
    print("   - 'ModelDetailManager: Updating progress bars with: {...}'")
    print("   - 'Updated main progress bar: ...'")
    print("4. Watch for updates every 2 seconds")
    
    return True

if __name__ == "__main__":
    success = test_auto_refresh()
    sys.exit(0 if success else 1)
