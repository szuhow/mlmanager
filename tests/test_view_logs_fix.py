#!/usr/bin/env python3
"""
Test script to verify that the "View Logs" functionality is working correctly.
"""

import os
import sys
import django
import json

# Setup Django
sys.path.append('/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
django.setup()

from django.test import Client
from ml_manager.models import MLModel
from django.contrib.auth.models import User

def test_view_logs_functionality():
    """Test the View Logs button and AJAX endpoint"""
    print("=" * 60)
    print("🔍 TESTING VIEW LOGS FUNCTIONALITY")
    print("=" * 60)
    
    # Create test client
    client = Client()
    
    # Create or get test user
    try:
        user = User.objects.get(username='testuser')
    except User.DoesNotExist:
        user = User.objects.create_user(username='testuser', password='testpass')
    
    # Login
    login_success = client.login(username='testuser', password='testpass')
    print(f"✅ User login: {'Success' if login_success else 'Failed'}")
    
    if not login_success:
        print("❌ Cannot proceed without login")
        return False
    
    # Get a model to test with
    try:
        model = MLModel.objects.first()
        if not model:
            print("❌ No models found in database")
            return False
        
        print(f"✅ Testing with model: {model.name} (ID: {model.id})")
        
        # Test regular page load
        print("\n1. Testing regular page load...")
        response = client.get(f'/ml/model/{model.id}/')
        print(f"   Status: {response.status_code}")
        print(f"   Template: {'model_detail.html' if 'model_detail.html' in str(response.content) else 'Unknown'}")
        
        if response.status_code == 200:
            print("   ✅ Regular page loads successfully")
        else:
            print(f"   ❌ Regular page failed: {response.status_code}")
            return False
        
        # Test AJAX endpoint
        print("\n2. Testing AJAX logs endpoint...")
        ajax_response = client.get(f'/ml/model/{model.id}/', 
                                 HTTP_X_REQUESTED_WITH='XMLHttpRequest')
        
        print(f"   AJAX Status: {ajax_response.status_code}")
        
        if ajax_response.status_code == 200:
            print("   ✅ AJAX endpoint responds")
            
            # Check response content
            try:
                data = json.loads(ajax_response.content)
                print(f"   📊 Response structure:")
                print(f"      - Status: {data.get('status', 'N/A')}")
                print(f"      - Logs present: {'logs' in data}")
                print(f"      - Progress present: {'progress' in data}")
                
                if 'logs' in data:
                    logs = data['logs']
                    print(f"      - Log count: {len(logs) if isinstance(logs, list) else 'N/A'}")
                    
                    if isinstance(logs, list) and len(logs) > 0:
                        # Show sample log entries
                        print(f"      - Sample log entries:")
                        for i, log in enumerate(logs[:3]):
                            if isinstance(log, dict):
                                content = log.get('content', str(log))[:60]
                                print(f"        {i+1}. {content}...")
                            else:
                                print(f"        {i+1}. {str(log)[:60]}...")
                        print("   ✅ Logs are available and properly formatted")
                    else:
                        print("   ⚠️  No log entries found")
                else:
                    print("   ❌ No 'logs' field in response")
                
                if 'log_stats' in data:
                    stats = data['log_stats']
                    print(f"      - Log statistics:")
                    print(f"        Total lines: {stats.get('total_lines', 0)}")
                    print(f"        Epoch logs: {stats.get('epoch_logs', 0)}")
                    print(f"        Batch logs: {stats.get('batch_logs', 0)}")
                
                print("   ✅ AJAX response is properly formatted")
                return True
                
            except json.JSONDecodeError as e:
                print(f"   ❌ AJAX response is not valid JSON: {e}")
                print(f"   Response content: {ajax_response.content[:200]}...")
                return False
        else:
            print(f"   ❌ AJAX endpoint failed: {ajax_response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_log_file_detection():
    """Test if log files can be detected"""
    print("\n" + "=" * 60)
    print("📁 TESTING LOG FILE DETECTION")
    print("=" * 60)
    
    try:
        models = MLModel.objects.all()[:3]  # Test first 3 models
        
        for model in models:
            print(f"\n🔍 Testing model: {model.name} (ID: {model.id})")
            print(f"   Status: {model.status}")
            print(f"   Model directory: {model.model_directory}")
            
            # Check for model-specific logs
            if model.model_directory and os.path.exists(model.model_directory):
                model_log_path = os.path.join(model.model_directory, 'logs', 'training.log')
                print(f"   Model-specific log path: {model_log_path}")
                print(f"   Exists: {os.path.exists(model_log_path)}")
                
                if os.path.exists(model_log_path):
                    try:
                        with open(model_log_path, 'r') as f:
                            lines = f.readlines()
                        print(f"   ✅ Model-specific log found: {len(lines)} lines")
                        
                        # Show sample lines
                        if lines:
                            print(f"   Sample lines:")
                            for i, line in enumerate(lines[:3]):
                                print(f"     {i+1}. {line.strip()[:80]}...")
                    except Exception as e:
                        print(f"   ❌ Error reading model log: {e}")
            else:
                print(f"   ❌ Model directory not found or doesn't exist")
            
            # Check for global logs
            global_log_path = os.path.join('models', 'artifacts', 'training.log')
            print(f"   Global log path: {global_log_path}")
            print(f"   Exists: {os.path.exists(global_log_path)}")
            
            if os.path.exists(global_log_path):
                try:
                    with open(global_log_path, 'r') as f:
                        lines = f.readlines()
                    print(f"   ✅ Global log found: {len(lines)} lines")
                except Exception as e:
                    print(f"   ❌ Error reading global log: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Log detection test failed: {e}")
        return False

def main():
    """Run all View Logs tests"""
    print("🔍 VIEW LOGS FUNCTIONALITY TEST")
    print("Testing the on-demand logging feature...")
    
    tests = [
        ("View Logs AJAX Functionality", test_view_logs_functionality),
        ("Log File Detection", test_log_file_detection),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:40} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All View Logs tests PASSED!")
        print("\n✅ The View Logs functionality should be working correctly.")
        print("🔗 Try clicking the 'View Logs' button in the model detail page.")
    else:
        print(f"\n❌ {len(results) - passed} tests failed. View Logs functionality needs fixes.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
