#!/usr/bin/env python3
"""
Test the fixed system metrics logging to MLflow
"""

import os
import sys
import django
import time
import mlflow
import logging

# Setup Django
sys.path.append('/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
django.setup()

from ml_manager.mlflow_utils import setup_mlflow, create_new_run
from shared.utils.system_monitor import SystemMonitor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_fixed_system_metrics():
    """Test that the fixed system metrics work properly"""
    print("=" * 60)
    print("🔧 TESTING FIXED SYSTEM METRICS LOGGING")
    print("=" * 60)
    
    try:
        # 1. Setup MLflow
        print("\n1. Setting up MLflow...")
        setup_mlflow()
        
        # 2. Create and start a test run
        print("\n2. Creating MLflow run...")
        run_id = create_new_run(params={'test_type': 'fixed_system_metrics'})
        
        with mlflow.start_run(run_id=run_id):
            print(f"✅ Started MLflow run: {mlflow.active_run().info.run_id}")
            
            # 3. Create system monitor
            print("\n3. Creating SystemMonitor...")
            system_monitor = SystemMonitor(log_interval=3, enable_gpu=True)
            
            # 4. Start monitoring (this should capture the run_id)
            print("\n4. Starting background monitoring...")
            system_monitor.start_monitoring()
            print("✅ Background monitoring started")
            
            # 5. Wait for the background thread to log some metrics
            print("\n5. Waiting for background metrics (10 seconds)...")
            for i in range(10):
                time.sleep(1)
                print(f"   Waiting... {i+1}/10 seconds", end='\r')
            print("\n✅ Wait complete")
            
            # 6. Stop monitoring
            print("\n6. Stopping monitoring...")
            system_monitor.stop_monitoring()
            print("✅ Monitoring stopped")
            
        # 7. Check what was logged
        print("\n7. Checking logged metrics...")
        final_run = mlflow.get_run(run_id)
        logged_metrics = final_run.data.metrics
        
        system_metrics = {k: v for k, v in logged_metrics.items() if k.startswith('system_')}
        print(f"✅ Total metrics: {len(logged_metrics)}")
        print(f"✅ System metrics: {len(system_metrics)}")
        
        if len(system_metrics) > 0:
            print("   Sample system metrics:")
            for key, value in sorted(system_metrics.items())[:5]:
                print(f"   {key}: {value}")
            print("🎉 SUCCESS: System metrics are being logged!")
            return True
        else:
            print("❌ FAILED: No system metrics found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_system_metrics()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 SYSTEM METRICS FIX IS WORKING!")
    else:
        print("🔧 SYSTEM METRICS STILL NEED MORE WORK")
    print("=" * 60)
