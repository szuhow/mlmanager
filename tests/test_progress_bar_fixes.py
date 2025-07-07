#!/usr/bin/env python3
"""
Test skrypt do sprawdzenia napraw progress barów dla batchy
"""

import os
import sys
import time
import django
import requests
from datetime import datetime

# Setup Django
sys.path.append('core')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
django.setup()

from apps.ml_manager.models import MLModel

def test_progress_bar_fixes():
    """Test napraw progress barów"""
    
    print(f"🧪 Testing Progress Bar Fixes at {datetime.now()}")
    
    # 1. Create test model with batch data
    print("\n1. Creating test model with batch progress data...")
    model = MLModel.objects.create(
        name="Test Progress Bar Model",
        status="training",
        current_epoch=3,
        total_epochs=10,
        current_batch=15,  # Important: batch data
        total_batches_per_epoch=32,  # Important: batch data
        train_loss=0.3,
        val_loss=0.25,
        train_dice=0.85,
        val_dice=0.82,
        best_val_dice=0.82,
        training_data_info={
            'model_type': 'unet',
            'loss_function': 'combined',
            'segmentation_metric': 'dice',
        }
    )
    
    print(f"✅ Model created with ID: {model.id}")
    print(f"📊 Model data:")
    print(f"   Epoch: {model.current_epoch}/{model.total_epochs}")
    print(f"   Batch: {model.current_batch}/{model.total_batches_per_epoch}")
    print(f"   Progress percentage: {model.progress_percentage:.1f}%")
    print(f"   Batch progress percentage: {model.batch_progress_percentage:.1f}%")
    
    # 2. Test API endpoint
    print("\n2. Testing training progress API...")
    try:
        url = f"http://localhost:8000/ml/api/training-progress/{model.id}/"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Response Status: {data.get('status')}")
            
            if data.get('status') == 'success':
                progress = data.get('progress', {})
                
                print(f"📊 API Progress Data:")
                print(f"   Current Epoch: {progress.get('current_epoch')}")
                print(f"   Total Epochs: {progress.get('total_epochs')}")
                print(f"   Current Batch: {progress.get('current_batch')}")
                print(f"   Total Batches: {progress.get('total_batches_per_epoch')}")
                print(f"   Progress Percentage: {progress.get('progress_percentage'):.1f}%")
                print(f"   Batch Progress Percentage: {progress.get('batch_progress_percentage'):.1f}%")
                
                # Verify batch data is present
                if progress.get('current_batch') and progress.get('total_batches_per_epoch'):
                    print("✅ Batch data is correctly returned by API")
                else:
                    print("❌ Batch data is missing from API response")
                    
                if progress.get('batch_progress_percentage') > 0:
                    print("✅ Batch progress percentage is calculated")
                else:
                    print("❌ Batch progress percentage is 0 or missing")
                    
            else:
                print(f"❌ API returned error: {data.get('message')}")
        else:
            print(f"❌ API returned status {response.status_code}")
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
    
    # 3. Test start-training page with model_id
    print("\n3. Testing start-training page...")
    try:
        url = f"http://localhost:8000/ml/start-training/?model_id={model.id}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ Start training page loads successfully")
            
            # Check if the page contains the new batch progress elements
            content = response.text
            
            checks = [
                ('batch-progress-section', 'Batch progress section element'),
                ('batch-progress-bar', 'Batch progress bar element'),
                ('batch-info-text', 'Batch info text element'),
                ('updateProgressDisplay', 'Updated progress display function'),
                ('batch_progress_percentage', 'Batch progress percentage handling')
            ]
            
            for element, description in checks:
                if element in content:
                    print(f"✅ {description} found in page")
                else:
                    print(f"❌ {description} missing from page")
                    
        else:
            print(f"❌ Start training page returned status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Page test failed: {e}")
    
    # 4. Simulate training progress with batch updates
    print("\n4. Simulating training progress with batch updates...")
    
    for batch in range(16, 25):  # Simulate batches 16-24
        model.current_batch = batch
        model.save()
        
        # Test API response
        try:
            response = requests.get(f"http://localhost:8000/ml/api/training-progress/{model.id}/")
            if response.status_code == 200:
                data = response.json()
                progress = data.get('progress', {})
                print(f"   Batch {batch}: {progress.get('batch_progress_percentage'):.1f}% batch progress")
            else:
                print(f"   Batch {batch}: API error {response.status_code}")
        except:
            print(f"   Batch {batch}: API request failed")
        
        time.sleep(0.5)  # Brief pause
    
    # 5. Test epoch completion
    print("\n5. Testing epoch completion...")
    model.current_epoch = 4
    model.current_batch = 1  # Reset to start of new epoch
    model.save()
    
    try:
        response = requests.get(f"http://localhost:8000/ml/api/training-progress/{model.id}/")
        if response.status_code == 200:
            data = response.json()
            progress = data.get('progress', {})
            print(f"✅ New epoch: {progress.get('current_epoch')}/{progress.get('total_epochs')}")
            print(f"✅ Batch reset: {progress.get('current_batch')}/{progress.get('total_batches_per_epoch')}")
            print(f"✅ Epoch progress: {progress.get('progress_percentage'):.1f}%")
            print(f"✅ Batch progress: {progress.get('batch_progress_percentage'):.1f}%")
    except:
        print("❌ Epoch completion test failed")
    
    # Cleanup
    print(f"\n6. Cleaning up...")
    try:
        model.delete()
        print(f"✅ Test model deleted")
    except Exception as e:
        print(f"⚠️  Could not delete test model: {e}")
    
    print(f"\n🎉 Progress Bar Fixes Test completed at {datetime.now()}")
    print("\n📋 Summary of fixes:")
    print("✅ Added batch progress bar to start_training.html")
    print("✅ Fixed API to use model.progress_percentage property")
    print("✅ Updated updateProgressDisplay() to handle batch data")
    print("✅ Added CSS styling for batch progress bar")
    print("✅ Added debugging logs for troubleshooting")
    
    print("\n🔧 Next steps:")
    print("1. Start a real training to see progress bars in action")
    print("2. Check browser console for debugging information")
    print("3. Verify batch progress updates every 2 seconds")

if __name__ == '__main__':
    test_progress_bar_fixes()
