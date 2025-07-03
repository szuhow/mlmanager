#!/usr/bin/env python3
"""
Test script to verify training logs are loaded from organized directories with auto-discovery
"""

import requests
import sys
import json

def test_log_auto_discovery():
    """Test if model logs are auto-discovered from organized directory structure"""
    
    # Test configuration
    BASE_URL = "http://localhost:8000"
    LOGIN_URL = f"{BASE_URL}/accounts/login/"
    
    print("🔧 Testing Log Auto-Discovery from Organized Directory Structure")
    print("=" * 70)
    
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
    
    print("✅ Login successful")
    
    # Step 3: Test multiple models to see if auto-discovery works
    test_model_ids = [5, 15, 25]  # Test models that might not have model_directory set
    
    success_count = 0
    for model_id in test_model_ids:
        print(f"\n🧪 Testing model ID {model_id}...")
        
        model_detail_url = f"{BASE_URL}/ml/model/{model_id}/"
        detail_response = session.get(model_detail_url)
        
        if detail_response.status_code == 404:
            print(f"⚠️  Model {model_id} not found, skipping...")
            continue
        elif detail_response.status_code != 200:
            print(f"❌ Failed to get model {model_id} detail: {detail_response.status_code}")
            continue
        
        response_text = detail_response.text
        
        # Check for various log loading indicators
        if "Found model directory by pattern matching" in response_text:
            print(f"✅ Model {model_id}: Auto-discovered directory by pattern matching")
            success_count += 1
        elif "Found model directory by ID matching" in response_text:
            print(f"✅ Model {model_id}: Auto-discovered directory by ID matching")
            success_count += 1
        elif "Model-specific log loaded:" in response_text:
            print(f"✅ Model {model_id}: Logs loaded from existing model_directory")
            success_count += 1
        elif "Alternative log file loaded:" in response_text:
            print(f"✅ Model {model_id}: Alternative log file loaded")
            success_count += 1
        elif "No model directory found" in response_text:
            print(f"⚠️  Model {model_id}: No model directory found (expected for some models)")
        elif "Global log (fallback)" in response_text:
            print(f"📄 Model {model_id}: Using global log fallback")
        else:
            print(f"❓ Model {model_id}: Unclear log status")
        
        # Check if training logs section exists
        if 'training_logs' in response_text or 'Training Logs' in response_text:
            print(f"   📋 Training logs section present")
        else:
            print(f"   ❌ No training logs section found")
    
    print(f"\n📊 Summary: {success_count}/{len(test_model_ids)} models had successful log discovery")
    
    # Step 4: Test that the system doesn't crash and provides reasonable fallbacks
    return success_count > 0  # Success if at least one model worked

if __name__ == "__main__":
    success = test_log_auto_discovery()
    sys.exit(0 if success else 1)
