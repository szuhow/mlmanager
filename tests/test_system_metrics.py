#!/usr/bin/env python3
"""
Test script to debug system metrics logging to MLflow
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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_system_metrics():
    """Test that system metrics are properly logged to MLflow"""
    print("=" * 60)
    print("🔍 DEBUGGING SYSTEM METRICS LOGGING TO MLFLOW")
    print("=" * 60)
    
    try:
        # 1. Setup MLflow
        print("\n1. Setting up MLflow...")
        setup_mlflow()
        print("✅ MLflow setup complete")
        
        # 2. Create a test run
        print("\n2. Creating test MLflow run...")
        test_params = {
            'test_type': 'system_metrics_debug',
            'monitoring_interval': 5
        }
        run_id = create_new_run(params=test_params)
        print(f"✅ Created MLflow run: {run_id}")
        
        # 3. Start the run and create system monitor
        print("\n3. Starting MLflow run and system monitor...")
        mlflow.start_run(run_id=run_id)
        print(f"✅ MLflow run started: {mlflow.active_run().info.run_id}")
        
        # Create system monitor with short interval for testing
        system_monitor = SystemMonitor(log_interval=5, enable_gpu=True)
        print("✅ SystemMonitor created")
        
        # 4. Test immediate metrics logging
        print("\n4. Testing immediate metrics logging...")
        metrics = system_monitor.get_system_metrics()
        print(f"✅ Collected {len(metrics)} system metrics:")
        for key, value in sorted(metrics.items())[:5]:  # Show first 5
            print(f"   {key}: {value}")
        print("   ...")
        
        # Try logging metrics manually
        try:
            mlflow.log_metrics(metrics, step=0)
            print("✅ Manual metrics logging successful")
        except Exception as e:
            print(f"❌ Manual metrics logging failed: {e}")
            return False
        
        # 5. Test SystemMonitor automatic logging
        print("\n5. Testing SystemMonitor automatic logging...")
        try:
            system_monitor.log_metrics_to_mlflow(step=1)
            print("✅ SystemMonitor.log_metrics_to_mlflow() successful")
        except Exception as e:
            print(f"❌ SystemMonitor.log_metrics_to_mlflow() failed: {e}")
            return False
        
        # 6. Test background monitoring
        print("\n6. Testing background monitoring for 15 seconds...")
        system_monitor.start_monitoring()
        print("✅ Background monitoring started")
        
        # Wait and monitor for a bit
        for i in range(3):
            time.sleep(5)
            print(f"   Monitoring... {(i+1)*5}s elapsed")
        
        system_monitor.stop_monitoring()
        print("✅ Background monitoring stopped")
        
        # 7. Check what was logged to MLflow
        print("\n7. Checking MLflow run data...")
        current_run = mlflow.get_run(run_id)
        logged_metrics = current_run.data.metrics
        print(f"✅ Total metrics logged to MLflow: {len(logged_metrics)}")
        
        # Show some system metrics
        system_metrics = {k: v for k, v in logged_metrics.items() if k.startswith('system_')}
        print(f"✅ System metrics found: {len(system_metrics)}")
        
        if system_metrics:
            print("   Sample system metrics:")
            for key, value in sorted(system_metrics.items())[:10]:  # Show first 10
                print(f"   {key}: {value}")
            return True
        else:
            print("❌ No system metrics found in MLflow run!")
            print("   All logged metrics:")
            for key, value in sorted(logged_metrics.items()):
                print(f"   {key}: {value}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            if 'system_monitor' in locals() and system_monitor:
                system_monitor.stop_monitoring()
            mlflow.end_run()
            print("\n✅ Cleanup complete")
        except:
            pass

if __name__ == "__main__":
    success = test_system_metrics()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 RESULT: SYSTEM METRICS LOGGING IS WORKING!")
    else:
        print("🔧 RESULT: SYSTEM METRICS LOGGING NEEDS DEBUGGING")
    print("=" * 60)
