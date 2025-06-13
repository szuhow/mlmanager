#!/usr/bin/env python3
"""
Quick training test to verify status updates during actual training
"""

import os
import sys
import django
import subprocess
import time
import signal
from pathlib import Path

# Setup Django
sys.path.insert(0, '/Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
django.setup()

from core.apps.ml_manager.models import MLModel

def quick_training_test():
    """Start a very short training to test status updates"""
    
    print("🚀 Quick Training Test (1 epoch, minimal dataset)")
    print("=" * 60)
    
    try:
        # Create a test model
        model = MLModel.objects.create(
            name='Quick Status Test',
            description='Testing real-time status updates',
            status='pending',
            current_epoch=0,
            total_epochs=1,  # Just 1 epoch
            model_family='UNet-Coronary',
            model_type='unet'
        )
        
        print(f"✅ Created test model: {model.name} (ID: {model.id})")
        
        # Start training in background
        training_args = [
            'python', '/app/ml/training/train.py',
            '--mode', 'train',
            '--model-type', 'unet',
            '--batch-size', '1',  # Minimal batch size
            '--epochs', '1',      # Just 1 epoch
            '--learning-rate', '0.001',
            '--data-path', '/app/data/datasets',
            '--validation-split', '0.2',
            '--crop-size', '64',   # Small crop size for speed
            '--num-workers', '0',
            '--model-id', str(model.id),
        ]
        
        print(f"🏃 Starting quick training...")
        print(f"Command: {' '.join(training_args)}")
        
        # Start the process
        process = subprocess.Popen(
            training_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print(f"✅ Training started (PID: {process.pid})")
        print("📊 Monitoring status changes...")
        
        # Monitor for 60 seconds
        start_time = time.time()
        last_status = model.status
        status_changes = []
        
        while time.time() - start_time < 60:
            # Check if process is still running
            if process.poll() is not None:
                print(f"🏁 Process completed with return code: {process.poll()}")
                break
            
            # Check model status
            model.refresh_from_db()
            if model.status != last_status:
                elapsed = time.time() - start_time
                status_changes.append((elapsed, last_status, model.status))
                print(f"📈 Status change at {elapsed:.1f}s: '{last_status}' → '{model.status}'")
                print(f"   Current: Epoch {model.current_epoch}/{model.total_epochs}, "
                      f"Batch {model.current_batch}/{model.total_batches_per_epoch}")
                last_status = model.status
                
                # If training completed, stop monitoring
                if model.status in ['completed', 'failed']:
                    print(f"🎯 Training finished with status: {model.status}")
                    break
            
            # Show periodic updates
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0 and elapsed > 0:
                print(f"⏰ {int(elapsed)}s - Status: {model.status}, "
                      f"Epoch: {model.current_epoch}/{model.total_epochs}")
            
            time.sleep(1)
        
        # Cleanup process if still running
        if process.poll() is None:
            print("🛑 Stopping training process...")
            try:
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        
        # Final status check
        model.refresh_from_db()
        print(f"\n📊 Final Results:")
        print(f"   - Final Status: {model.status}")
        print(f"   - Final Epoch: {model.current_epoch}/{model.total_epochs}")
        print(f"   - Status Changes: {len(status_changes)}")
        
        if status_changes:
            print(f"   - Status Transitions:")
            for elapsed, old, new in status_changes:
                print(f"     {elapsed:.1f}s: {old} → {new}")
        
        # Cleanup
        model.delete()
        print(f"🧹 Cleaned up test model")
        
        # Determine success
        success = len(status_changes) >= 2  # At least pending→loading→training or similar
        if success:
            print("\n✅ SUCCESS: Status updates are working during training!")
        else:
            print("\n⚠️ LIMITED SUCCESS: Some status updates detected, but may need improvement")
            
        return success
        
    except Exception as e:
        print(f"❌ Error during quick training test: {e}")
        return False

if __name__ == "__main__":
    success = quick_training_test()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 TRAINING STATUS UPDATES ARE WORKING!")
        print("💡 The automatic refresh in the UI should now work properly.")
    else:
        print("⚠️ TRAINING STATUS UPDATES NEED ATTENTION")
        print("💡 Manual refresh may be needed until this is fixed.")
    print("=" * 60)
