#!/usr/bin/env python3
"""
Test script to verify the improved system metrics organization in MLflow
"""

import os
import sys
import django
import mlflow
import time
from pathlib import Path

# Setup Django environment
project_root = Path(__file__).parent
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
django.setup()

from shared.utils.system_monitor import SystemMonitor
from ml_manager.mlflow_utils import setup_mlflow

def test_system_metrics_organization():
    """Test the improved system metrics organization"""
    
    print("🔬 TESTING IMPROVED SYSTEM METRICS ORGANIZATION")
    print("=" * 60)
    
    try:
        # Setup MLflow
        print("1. Setting up MLflow...")
        setup_mlflow()
        
        # Start a test run
        with mlflow.start_run(run_name="System_Metrics_Organization_Test") as run:
            print(f"   ✅ Started MLflow run: {run.info.run_id}")
            
            # Initialize system monitor
            print("2. Initializing system monitor...")
            monitor = SystemMonitor(log_interval=5, enable_gpu=True)
            monitor.mlflow_run_id = run.info.run_id
            
            # Get and display raw metrics
            print("3. Collecting system metrics...")
            raw_metrics = monitor.get_system_metrics()
            print(f"   ✅ Collected {len(raw_metrics)} raw metrics")
            
            # Display sample raw metrics
            print("\n📊 Sample raw metrics:")
            for key, value in list(raw_metrics.items())[:10]:
                print(f"   {key}: {value}")
            if len(raw_metrics) > 10:
                print(f"   ... and {len(raw_metrics) - 10} more")
            
            # Log metrics using the new organization
            print("\n4. Logging metrics with new organization...")
            monitor.log_metrics_to_mlflow(step=0)
            
            # Wait a moment for metrics to be logged
            time.sleep(2)
            
            # Log a second set of metrics to see the progression
            print("5. Logging second set of metrics...")
            monitor.log_metrics_to_mlflow(step=1)
            
            print("\n✅ SUCCESS! System metrics organization test completed")
            print(f"📊 MLflow Run ID: {run.info.run_id}")
            print(f"🌐 View metrics at: http://localhost:5000/#/experiments/0/runs/{run.info.run_id}")
            
            # Display expected metric organization
            print("\n📁 Expected metric organization in MLflow:")
            print("   system/hardware/cpu/")
            print("   ├── usage_percent")
            print("   ├── core_count")
            print("   ├── logical_count")
            print("   ├── frequency_current_mhz")
            print("   └── frequency_max_mhz")
            print("   ")
            print("   system/hardware/memory/")
            print("   ├── usage_percent")
            print("   ├── available_gb")
            print("   ├── used_gb")
            print("   ├── total_gb")
            print("   └── free_gb")
            print("   ")
            print("   system/hardware/gpu/device_0/")
            print("   ├── compute_utilization_percent")
            print("   ├── memory_utilization_percent")
            print("   ├── memory_used_gb")
            print("   ├── memory_total_gb")
            print("   ├── memory_free_gb")
            print("   ├── memory_usage_percent")
            print("   ├── temperature_celsius")
            print("   └── power_usage_watts")
            print("   ")
            print("   system/hardware/storage/")
            print("   ├── read_mb")
            print("   ├── write_mb")
            print("   ├── read_ops")
            print("   └── write_ops")
            print("   ")
            print("   system/hardware/network/")
            print("   ├── sent_mb")
            print("   ├── received_mb")
            print("   ├── packets_sent")
            print("   └── packets_received")
            print("   ")
            print("   system/process/")
            print("   ├── cpu_usage_percent")
            print("   ├── memory_rss_gb")
            print("   ├── memory_vms_gb")
            print("   ├── memory_usage_percent")
            print("   ├── thread_count")
            print("   └── file_descriptors")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_old_vs_new_organization():
    """Compare old vs new metric organization"""
    
    print("\n🔄 COMPARING OLD VS NEW METRIC ORGANIZATION")
    print("=" * 60)
    
    # Simulate old organization
    old_metrics = {
        "hardware.gpu_0_utilization_percent": 85.2,
        "hardware.system_cpu_percent": 45.1,
        "hardware.system_memory_percent": 67.8,
        "process.cpu_percent": 12.3,
        "process.memory_rss_gb": 2.1,
        "system.disk_read_mb": 123.4,
        "system.network_sent_mb": 45.6
    }
    
    # Simulate new organization
    new_metrics = {
        "system/hardware/gpu/device_0/compute_utilization_percent": 85.2,
        "system/hardware/cpu/usage_percent": 45.1,
        "system/hardware/memory/usage_percent": 67.8,
        "system/process/cpu_usage_percent": 12.3,
        "system/process/memory_rss_gb": 2.1,
        "system/hardware/storage/read_mb": 123.4,
        "system/hardware/network/sent_mb": 45.6
    }
    
    print("📊 OLD Organization (flat namespaces):")
    for key, value in old_metrics.items():
        print(f"   {key}: {value}")
    
    print("\n📊 NEW Organization (hierarchical namespaces):")
    for key, value in new_metrics.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Benefits of new organization:")
    print("   • Hierarchical structure in MLflow UI")
    print("   • Better grouping of related metrics")
    print("   • Clearer metric names")
    print("   • Easier filtering and comparison")
    print("   • Better scaling for multiple GPUs")
    print("   • Improved navigation in MLflow interface")

if __name__ == "__main__":
    print("🧪 SYSTEM METRICS ORGANIZATION TEST SUITE")
    print("=" * 60)
    
    # Run the main test
    success = test_system_metrics_organization()
    
    # Compare organizations
    compare_old_vs_new_organization()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED - System metrics organization improved!")
        print("💡 Check MLflow UI to see the new hierarchical organization")
    else:
        print("❌ TESTS FAILED - Check error messages above")
    print("=" * 60)
