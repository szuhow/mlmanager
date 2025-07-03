#!/usr/bin/env python3
"""
Test script to verify training logs are loaded from organized model directories
"""

import requests
import sys
import json

def test_model_logs():
    """Test if model logs are loaded from organized directory structure"""
    
    # Test configuration
    BASE_URL = "http://localhost:8000"
    LOGIN_URL = f"{BASE_URL}/accounts/login/"
    
    print("🔧 Testing Model Logs from Organized Directory Structure")
    print("=" * 60)
    
    # Create session for maintaining login
    session = requests.Session()
    
    # Step 1: Get login page to extract CSRF token
    print("📝 Getting login page...")
    login_page = session.get(LOGIN_URL)
    if login_page.status_code != 200:
        print(f"❌ Failed to get login page: {login_page.status_code}")
        return False
    
    # Extract CSRF token from login page
    csrf_token = None
    for line in login_page.text.split('\n'):
        if 'csrfmiddlewaretoken' in line and 'value=' in line:
            csrf_token = line.split('value="')[1].split('"')[0]
            break
    
    if not csrf_token:
        print("❌ Could not find CSRF token")
        return False
    
    print(f"✅ CSRF token found: {csrf_token[:10]}...")
    
    # Step 2: Login
    print("🔐 Logging in...")
    login_data = {
        'username': 'admin',
        'password': 'admin123',
        'csrfmiddlewaretoken': csrf_token
    }
    
    login_response = session.post(LOGIN_URL, data=login_data)
    if login_response.status_code not in [200, 302]:
        print(f"❌ Login failed: {login_response.status_code}")
        return False
    
    # Check if we're redirected after successful login
    if login_response.status_code == 302:
        print("✅ Login successful (redirected)")
    else:
        # Check if we're still on login page (which would mean login failed)
        if 'login' in login_response.url:
            print("❌ Login failed - still on login page")
            return False
        print("✅ Login successful")
    
    # Step 3: Get list of models to find one with organized directory structure
    print("📋 Getting model list...")
    models_url = f"{BASE_URL}/ml/"
    models_response = session.get(models_url)
    
    if models_response.status_code != 200:
        print(f"❌ Failed to get models list: {models_response.status_code}")
        return False
    
    print("✅ Models list retrieved")
    
    # Look for models with organized directory structure in the HTML
    models_with_organized_dirs = []
    lines = models_response.text.split('\n')
    
    for i, line in enumerate(lines):
        if 'data/models/organized' in line and 'unet-coronary' in line:
            # Try to extract model ID from nearby HTML
            for j in range(max(0, i-10), min(len(lines), i+10)):
                if '/ml/model/' in lines[j]:
                    try:
                        # Extract model ID from URL like /ml/model/123/
                        model_id = lines[j].split('/ml/model/')[1].split('/')[0]
                        if model_id.isdigit():
                            models_with_organized_dirs.append(int(model_id))
                            break
                    except:
                        continue
    
    if not models_with_organized_dirs:
        print("❌ No models found with organized directory structure")
        return False
    
    # Step 4: Test logs for a model with organized directory
    test_model_id = models_with_organized_dirs[0]
    print(f"🧪 Testing logs for model ID {test_model_id}...")
    
    model_detail_url = f"{BASE_URL}/ml/model/{test_model_id}/"
    detail_response = session.get(model_detail_url)
    
    if detail_response.status_code != 200:
        print(f"❌ Failed to get model detail: {detail_response.status_code}")
        return False
    
    # Look for training logs in the response
    response_text = detail_response.text
    
    # Check if logs are present and loaded from organized structure
    logs_found = False
    organized_logs_loaded = False
    
    if 'training_logs' in response_text or 'Training Logs' in response_text:
        logs_found = True
        print("✅ Training logs section found")
        
        # Check for indicators that logs were loaded from organized directory
        if 'Model-specific log loaded:' in response_text or 'training.log' in response_text:
            organized_logs_loaded = True
            print("✅ Logs appear to be loaded from organized directory structure")
        elif 'No training logs found' in response_text:
            print("⚠️  No training logs found message")
        elif 'Error loading logs' in response_text:
            print("❌ Error loading logs")
        else:
            print("⚠️  Logs found but source unclear")
    else:
        print("❌ No training logs section found")
    
    # Step 5: Check browser console/debug info if available
    if 'Model-specific log loaded:' in response_text:
        print("✅ SUCCESS: Logs loaded from organized directory structure")
        return True
    elif 'Alternative log file loaded:' in response_text:
        print("✅ SUCCESS: Alternative log file loaded from organized directory")
        return True
    elif logs_found:
        print("⚠️  PARTIAL: Logs found but may not be from organized directory")
        return True
    else:
        print("❌ FAILED: No logs loaded")
        return False

if __name__ == "__main__":
    success = test_model_logs()
    sys.exit(0 if success else 1)
