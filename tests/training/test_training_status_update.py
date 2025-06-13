#!/usr/bin/env python3
"""
Test script to verify that training status updates are working properly
"""

import os
import sys
import django
import time
import subprocess
from pathlib import Path

# Setup Django
sys.path.insert(0, '/Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
django.setup()

from core.apps.ml_manager.models import MLModel

def test_training_status_updates():
    """Test if training status updates are working in the UI"""
    
    print("🔧 Testing Training Status Updates...")
    
    try:
        # Find a training model or create a test one
        training_models = MLModel.objects.filter(status='training')
        
        if training_models.exists():
            model = training_models.first()
            print(f"✅ Found training model: {model.name} (ID: {model.id})")
            
            # Monitor the model for 30 seconds and check for updates
            initial_epoch = model.current_epoch
            initial_batch = model.current_batch
            initial_train_loss = model.train_loss
            
            print(f"📊 Initial state:")
            print(f"   - Epoch: {initial_epoch}/{model.total_epochs}")
            print(f"   - Batch: {initial_batch}/{model.total_batches_per_epoch}")
            print(f"   - Train Loss: {initial_train_loss}")
            print(f"   - Status: {model.status}")
            
            print("\n⏳ Monitoring for changes (30 seconds)...")
            
            changes_detected = []
            for i in range(30):
                time.sleep(1)
                model.refresh_from_db()
                
                # Check for changes
                if (model.current_epoch != initial_epoch or 
                    model.current_batch != initial_batch or 
                    abs(model.train_loss - initial_train_loss) > 0.0001):
                    
                    change = {
                        'time': i + 1,
                        'epoch': model.current_epoch,
                        'batch': model.current_batch,
                        'train_loss': model.train_loss,
                        'status': model.status
                    }
                    changes_detected.append(change)
                    print(f"📈 Change at {i+1}s: Epoch {model.current_epoch}/{model.total_epochs}, "
                          f"Batch {model.current_batch}/{model.total_batches_per_epoch}, "
                          f"Loss {model.train_loss:.4f}")
                    
                    # Update baseline
                    initial_epoch = model.current_epoch
                    initial_batch = model.current_batch
                    initial_train_loss = model.train_loss
            
            print(f"\n📊 Monitoring complete!")
            print(f"   - Changes detected: {len(changes_detected)}")
            
            if changes_detected:
                print("✅ Training status updates are working!")
                print("\n📈 Summary of changes:")
                for change in changes_detected[-3:]:  # Show last 3 changes
                    print(f"   - At {change['time']}s: Epoch {change['epoch']}, "
                          f"Batch {change['batch']}, Loss {change['train_loss']:.4f}")
                
                return True
            else:
                print("⚠️ No changes detected. Possible issues:")
                print("   1. Training might be paused or stuck")
                print("   2. Training callback might not be working")
                print("   3. Model updates might not be reaching the database")
                
                # Additional debugging
                print(f"\n🔍 Current model state:")
                print(f"   - Status: {model.status}")
                print(f"   - MLflow Run ID: {model.mlflow_run_id}")
                print(f"   - Stop Requested: {model.stop_requested}")
                
                return False
        else:
            print("⚠️ No training models found.")
            print("💡 To test this:")
            print("   1. Start a training job in the web interface")
            print("   2. Run this script while training is active")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def test_manual_status_update():
    """Test if we can manually update a model's status"""
    
    print("\n🔧 Testing Manual Status Updates...")
    
    try:
        # Find any model
        model = MLModel.objects.first()
        if not model:
            print("⚠️ No models found in database")
            return False
            
        print(f"✅ Found model: {model.name}")
        
        # Save original values
        original_epoch = model.current_epoch
        original_batch = model.current_batch
        original_loss = model.train_loss
        
        # Update values
        model.current_epoch = (original_epoch or 0) + 1
        model.current_batch = (original_batch or 0) + 5
        model.train_loss = 0.1234
        model.save()
        
        print(f"📝 Updated model values")
        
        # Verify update
        model.refresh_from_db()
        if (model.current_epoch == (original_epoch or 0) + 1 and
            model.current_batch == (original_batch or 0) + 5 and
            abs(model.train_loss - 0.1234) < 0.0001):
            
            print("✅ Manual status updates work correctly!")
            
            # Restore original values
            model.current_epoch = original_epoch or 0
            model.current_batch = original_batch or 0
            model.train_loss = original_loss
            model.save()
            
            return True
        else:
            print("❌ Manual status updates failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during manual update test: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Training Status Update Test")
    print("=" * 50)
    
    # Test manual updates first
    manual_test_result = test_manual_status_update()
    
    # Test live training updates
    live_test_result = test_training_status_updates()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"   - Manual Updates: {'✅ PASS' if manual_test_result else '❌ FAIL'}")
    print(f"   - Live Updates: {'✅ PASS' if live_test_result else '❌ FAIL'}")
    
    if manual_test_result and live_test_result:
        print("\n🎉 All tests passed! Training status updates are working.")
    elif manual_test_result and not live_test_result:
        print("\n⚠️ Manual updates work, but live training updates may need attention.")
        print("💡 This could mean:")
        print("   - Training process is not using the callback properly")
        print("   - There are no active training jobs")
        print("   - Training is in a waiting/paused state")
    else:
        print("\n❌ There are issues with the training status update system.")
        print("🔧 This needs investigation.")
    
    print("=" * 50)
