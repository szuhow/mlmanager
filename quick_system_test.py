#!/usr/bin/env python3
"""
Quick test to verify system metrics work in the training script
"""

import sys
import os
sys.path.append('/app')
sys.path.append('/app/shared')

import mlflow
import time
from shared.utils.system_monitor import SystemMonitor

def test_system_metrics_quick():
    """Quick test of system metrics"""
    print("🔍 Testing SystemMonitor...")
    
    # Test 1: Basic metrics collection
    monitor = SystemMonitor(log_interval=2, enable_gpu=True)
    metrics = monitor.get_system_metrics()
    print(f"✅ Collected {len(metrics)} metrics")
    
    # Test 2: MLflow integration
    print("🧪 Testing MLflow integration...")
    
    # Set up MLflow tracking URI
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("system_metrics_test")
    
    with mlflow.start_run(run_name="quick_system_test"):
        print(f"✅ Started MLflow run: {mlflow.active_run().info.run_id}")
        
        # Test direct logging
        monitor.log_metrics_to_mlflow(step=0)
        print("✅ Direct metrics logging successful")
        
        # Test background monitoring
        print("🚀 Starting background monitoring...")
        monitor.start_monitoring()
        
        # Wait for background thread to log
        time.sleep(6)  # Wait for 3 logging cycles (2s interval)
        
        monitor.stop_monitoring()
        print("✅ Background monitoring complete")
        
        # Check what was logged
        run_info = mlflow.get_run(mlflow.active_run().info.run_id)
        metrics_logged = len(run_info.data.metrics)
        system_metrics = len([k for k in run_info.data.metrics.keys() if k.startswith('system_')])
        
        print(f"📊 Total metrics logged: {metrics_logged}")
        print(f"📊 System metrics logged: {system_metrics}")
        
        if system_metrics > 0:
            print("🎉 SUCCESS: System metrics are working!")
            return True
        else:
            print("❌ FAILED: No system metrics found")
            return False

if __name__ == "__main__":
    try:
        success = test_system_metrics_quick()
        if success:
            print("\n🎊 SYSTEM METRICS ARE WORKING IN THE CONTAINER!")
        else:
            print("\n🔧 SYSTEM METRICS NEED MORE DEBUGGING")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
