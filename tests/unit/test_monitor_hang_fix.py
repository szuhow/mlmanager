#!/usr/bin/env python3
"""
Test script to verify the SystemMonitor hang fix
"""
import os
import sys
import time
import threading
import logging
import mlflow

# Setup Django environment
sys.path.append('/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.testing')

import django
django.setup()

from ml_manager.mlflow_utils import setup_mlflow, create_new_run
from ml.utils.utils.system_monitor import SystemMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_monitor_fix.log')
    ]
)
logger = logging.getLogger(__name__)

def test_system_monitor_hang_fix():
    """Test that SystemMonitor properly stops when MLflow run ends"""
    logger.info("=== Testing SystemMonitor Hang Fix ===")
    
    try:
        # 1. Setup MLflow and create a test run
        logger.info("Setting up MLflow and creating test run...")
        setup_mlflow()
        
        test_params = {
            'test_type': 'system_monitor_hang_fix',
            'monitoring_interval': 2  # Short interval for testing
        }
        run_id = create_new_run(params=test_params)
        logger.info(f"Created MLflow run: {run_id}")
        
        # 2. Start MLflow run and system monitor
        logger.info("Starting MLflow run and system monitor...")
        mlflow.start_run(run_id=run_id)
        
        # Create system monitor with short interval
        system_monitor = SystemMonitor(log_interval=2, enable_gpu=False)
        system_monitor.start_monitoring()
        logger.info("System monitoring started")
        
        # 3. Let it run for a few seconds
        logger.info("Letting monitor run for 6 seconds...")
        time.sleep(6)
        
        # 4. Check thread status before stopping
        initial_threads = threading.enumerate()
        logger.info(f"Active threads before stopping: {len(initial_threads)}")
        for t in initial_threads:
            if hasattr(t, '_target') and 'monitor' in str(t._target):
                logger.info(f"  Monitor thread: {t.name} (alive: {t.is_alive()})")
        
        # 5. Stop system monitor first (like in real training)
        logger.info("Stopping system monitor...")
        system_monitor.stop_monitoring()
        logger.info("System monitor stopped")
        
        # 6. End MLflow run
        logger.info("Ending MLflow run...")
        mlflow.end_run()
        logger.info("MLflow run ended")
        
        # 7. Wait a bit and check if threads are cleaned up
        time.sleep(3)
        final_threads = threading.enumerate()
        logger.info(f"Active threads after stopping: {len(final_threads)}")
        
        # Check for lingering monitor threads
        monitor_threads = []
        for t in final_threads:
            if hasattr(t, '_target') and 'monitor' in str(t._target):
                monitor_threads.append(t)
                logger.warning(f"  Lingering monitor thread: {t.name} (alive: {t.is_alive()})")
        
        if not monitor_threads:
            logger.info("✅ SUCCESS: No lingering monitor threads detected")
            return True
        else:
            logger.error(f"❌ FAILURE: {len(monitor_threads)} monitor threads still active")
            return False
            
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        return False

def test_mlflow_run_status_check():
    """Test that SystemMonitor properly detects finished runs"""
    logger.info("=== Testing MLflow Run Status Check ===")
    
    try:
        # Setup MLflow
        setup_mlflow()
        
        # Create and immediately finish a run
        test_params = {'test_type': 'status_check_test'}
        run_id = create_new_run(params=test_params)
        
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric('test_metric', 1.0)
        # Run is now finished
        
        logger.info(f"Created and finished MLflow run: {run_id}")
        
        # Try to create a SystemMonitor and log to the finished run
        system_monitor = SystemMonitor(log_interval=1, enable_gpu=False)
        system_monitor.mlflow_run_id = run_id
        
        # This should detect the run is finished and stop monitoring
        logger.info("Attempting to log to finished run...")
        system_monitor.log_metrics_to_mlflow(step=0)
        
        # Check if monitoring was disabled
        if not system_monitor.monitoring:
            logger.info("✅ SUCCESS: SystemMonitor correctly detected finished run")
            return True
        else:
            logger.error("❌ FAILURE: SystemMonitor did not detect finished run")
            return False
            
    except Exception as e:
        logger.error(f"Status check test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("Starting SystemMonitor hang fix tests...")
    
    # Test 1: Normal stop behavior
    test1_success = test_system_monitor_hang_fix()
    
    # Test 2: Finished run detection
    test2_success = test_mlflow_run_status_check()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"Test 1 (Normal Stop): {'✅ PASS' if test1_success else '❌ FAIL'}")
    logger.info(f"Test 2 (Status Check): {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    overall_success = test1_success and test2_success
    if overall_success:
        logger.info("🎉 ALL TESTS PASSED - SystemMonitor hang fix is working!")
    else:
        logger.info("🔧 SOME TESTS FAILED - Need further investigation")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
