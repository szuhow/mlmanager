#!/usr/bin/env python3
"""
Test script to verify progress API returns correct data structure for progress bars
"""

import requests
import sys
import json

def test_progress_api():
    """Test if progress API returns correct data structure"""
    
    # Test configuration
    BASE_URL = "http://localhost:8000"
    LOGIN_URL = f"{BASE_URL}/accounts/login/"
    
    print("🔧 Testing Progress API Data Structure")
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
    
    # Step 2: Test progress API for multiple models
    test_model_ids = [5, 15, 25, 38, 89]  # Test various models
    
    for model_id in test_model_ids:
        print(f"\n🧪 Testing progress API for model ID {model_id}...")
        
        progress_url = f"{BASE_URL}/ml/model/{model_id}/progress/"
        try:
            progress_response = session.get(progress_url)
            
            if progress_response.status_code == 404:
                print(f"⚠️  Model {model_id} not found, skipping...")
                continue
            elif progress_response.status_code != 200:
                print(f"❌ Failed to get progress for model {model_id}: {progress_response.status_code}")
                continue
            
            # Parse JSON response
            try:
                data = progress_response.json()
                print(f"✅ Model {model_id}: API responded successfully")
                
                # Check data structure
                if 'progress' in data:
                    progress = data['progress']
                    print(f"   📊 Progress data available")
                    
                    # Check required fields
                    required_fields = ['current_epoch', 'total_epochs', 'progress_percentage']
                    missing_fields = [f for f in required_fields if f not in progress]
                    
                    if missing_fields:
                        print(f"   ❌ Missing required fields: {missing_fields}")
                    else:
                        print(f"   ✅ All required progress fields present")
                        print(f"      Epoch: {progress.get('current_epoch', 0)}/{progress.get('total_epochs', 0)}")
                        print(f"      Progress: {progress.get('progress_percentage', 0):.1f}%")
                    
                    # Check batch progress fields
                    batch_fields = ['current_batch', 'total_batches_per_epoch', 'batch_progress_percentage']
                    batch_present = [f for f in batch_fields if f in progress]
                    
                    if batch_present:
                        print(f"   📈 Batch progress fields: {batch_present}")
                        if 'batch_progress_percentage' in progress:
                            print(f"      Batch progress: {progress.get('batch_progress_percentage', 0):.1f}%")
                    else:
                        print(f"   ⚠️  No batch progress fields found")
                    
                    # Show raw progress data for debugging
                    print(f"   🔍 Raw progress data: {json.dumps(progress, indent=2)}")
                    
                else:
                    print(f"   ❌ No 'progress' key in response")
                
                # Check metrics
                if 'metrics' in data:
                    metrics = data['metrics']
                    print(f"   📈 Metrics available: {list(metrics.keys())}")
                else:
                    print(f"   ⚠️  No metrics in response")
                
            except json.JSONDecodeError as e:
                print(f"   ❌ Failed to parse JSON response: {e}")
                print(f"   Raw response: {progress_response.text[:200]}...")
                
        except Exception as e:
            print(f"   ❌ Exception testing model {model_id}: {e}")
    
    return True

if __name__ == "__main__":
    success = test_progress_api()
    sys.exit(0 if success else 1)
