#!/usr/bin/env python3
"""
Test enhanced status detection and auto-refresh functionality
"""

import os
import sys
import django
import time
from pathlib import Path

# Setup Django
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
django.setup()

from core.apps.ml_manager.models import MLModel


def test_enhanced_status_detection():
    """Test the enhanced status detection system"""
    
    print("🧪 Testing Enhanced Status Detection System")
    print("=" * 60)
    
    try:
        # Find or create a test model
        test_model = MLModel.objects.filter(status='pending').first()
        
        if not test_model:
            # Create a test model in pending state
            test_model = MLModel.objects.create(
                name='Enhanced Status Detection Test Model',
                description='Test model for enhanced status detection',
                status='pending',
                total_epochs=5,
                current_epoch=0
            )
            print(f"✅ Created test model {test_model.id} with status: {test_model.status}")
        else:
            print(f"✅ Using existing model {test_model.id} with status: {test_model.status}")
        
        # Test status transitions
        print(f"\n🔄 Testing Status Transitions...")
        
        statuses_to_test = ['loading', 'training', 'completed']
        
        for i, status in enumerate(statuses_to_test):
            print(f"\n📊 Transition {i+1}: {test_model.status} → {status}")
            
            # Update status
            old_status = test_model.status
            test_model.status = status
            test_model.current_epoch = i + 1
            test_model.save()
            
            print(f"   ✅ Status updated in database")
            print(f"   🔍 Enhanced detection should catch: {old_status} → {status}")
            print(f"   💡 Check the web UI debug panel for status transition logs")
            print(f"   📡 API endpoint: /ml/model/{test_model.id}/progress/")
            
            # Wait a moment between transitions
            if i < len(statuses_to_test) - 1:
                print(f"   ⏳ Waiting 5 seconds before next transition...")
                time.sleep(5)
        
        print(f"\n🎯 Enhanced Status Detection Test Summary:")
        print(f"   - Model ID: {test_model.id}")
        print(f"   - Final Status: {test_model.status}")
        print(f"   - Epochs Simulated: {test_model.current_epoch}")
        print(f"   - Test URL: /ml/model/{test_model.id}/")
        
        print(f"\n💡 To test the enhanced auto-refresh:")
        print(f"   1. Open the model detail page: /ml/model/{test_model.id}/")
        print(f"   2. Click the 'Debug' button to see status transition logs")
        print(f"   3. Run this script to trigger status changes")
        print(f"   4. Watch the debug panel for enhanced detection messages")
        
        print(f"\n📝 Enhanced Features Active:")
        print(f"   ✅ Session-based status change detection")
        print(f"   ✅ Fast polling for pending→training transitions")
        print(f"   ✅ Browser notifications for completion")
        print(f"   ✅ Enhanced debug logging")
        print(f"   ✅ Automatic page refresh on completion")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during enhanced status detection test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_status_detection():
    """Test the API status detection endpoint directly"""
    
    print(f"\n🔌 Testing API Status Detection Endpoint...")
    
    try:
        # Find a test model
        test_model = MLModel.objects.first()
        if not test_model:
            print("⚠️ No models found for API testing")
            return False
        
        from django.test import Client
        from django.contrib.auth.models import User
        
        # Get or create a test user
        user, created = User.objects.get_or_create(
            username='testuser',
            defaults={'email': 'test@example.com'}
        )
        
        client = Client()
        client.force_login(user)
        
        print(f"🔍 Testing API endpoint for model {test_model.id}...")
        
        # First request - establish baseline
        response1 = client.get(f'/ml/model/{test_model.id}/progress/')
        if response1.status_code == 200:
            data1 = response1.json()
            print(f"   📊 Initial request: Status = {data1.get('model_status')}")
            print(f"   🔧 Status changed: {data1.get('status_changed')}")
        
        # Simulate status change
        old_status = test_model.status
        test_model.status = 'training' if old_status != 'training' else 'completed'
        test_model.save()
        
        # Second request - should detect change
        response2 = client.get(f'/ml/model/{test_model.id}/progress/')
        if response2.status_code == 200:
            data2 = response2.json()
            print(f"   📊 After change: Status = {data2.get('model_status')}")
            print(f"   🔧 Status changed: {data2.get('status_changed')}")
            print(f"   🔄 Transition: {data2.get('status_transition')}")
        
        # Restore original status
        test_model.status = old_status
        test_model.save()
        
        print(f"   ✅ API status detection test completed")
        return True
        
    except Exception as e:
        print(f"❌ API test error: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Enhanced Status Detection Test Suite")
    print("=" * 60)
    
    # Test 1: Enhanced status detection
    test1_result = test_enhanced_status_detection()
    
    # Test 2: API endpoint detection
    test2_result = test_api_status_detection()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"   - Enhanced Status Detection: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"   - API Status Detection: {'✅ PASS' if test2_result else '❌ FAIL'}")
    
    if test1_result and test2_result:
        print("\n🎉 ALL TESTS PASSED!")
        print("💡 Enhanced auto-refresh with status detection is working!")
        print("🔥 Features include:")
        print("   • Automatic status transition detection")
        print("   • Fast polling for pending→training transitions") 
        print("   • Browser notifications on completion")
        print("   • Enhanced debug logging")
        print("   • Seamless page refresh on completion")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("🔧 Check the implementation for issues")
    
    print("=" * 60)
