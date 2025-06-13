#!/usr/bin/env python3
"""
Debug script to identify why training processes hang after MLflow run completion
"""
import os
import sys
import time
import signal
import threading
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug_process_hang.log')
    ]
)
logger = logging.getLogger(__name__)

def check_active_threads():
    """Check all active threads in the current process"""
    threads = threading.enumerate()
    logger.info(f"Found {len(threads)} active threads:")
    
    for i, thread in enumerate(threads):
        logger.info(f"  Thread {i+1}: {thread.name} (daemon={thread.daemon}, alive={thread.is_alive()})")
        
        # Check if thread has a specific target function
        if hasattr(thread, '_target') and thread._target:
            target_name = getattr(thread._target, '__name__', str(thread._target))
            logger.info(f"    Target function: {target_name}")
    
    return threads

def check_docker_processes():
    """Check processes running in Docker container"""
    try:
        logger.info("Checking processes in Docker container...")
        os.system("docker exec web sh -c 'ls /proc/*/cmdline 2>/dev/null | head -20 | xargs cat 2>/dev/null | tr \"\\0\" \" \" | grep -E \"(python|train)\"'")
    except Exception as e:
        logger.error(f"Failed to check Docker processes: {e}")

def simulate_training_completion():
    """Simulate what happens at training completion"""
    logger.info("=== Simulating Training Completion ===")
    
    # Import MLflow and training modules
    try:
        import mlflow
        from ml.utils.utils.system_monitor import SystemMonitor
        
        # Check current MLflow run status
        if mlflow.active_run():
            run_info = mlflow.active_run().info
            logger.info(f"Active MLflow run: {run_info.run_id} (status: {run_info.status})")
        else:
            logger.info("No active MLflow run")
        
        # Create a SystemMonitor instance (like in training)
        monitor = SystemMonitor(log_interval=30, enable_gpu=False)
        
        # Start monitoring (like in training)
        logger.info("Starting system monitoring...")
        monitor.start_monitoring()
        
        # Wait a bit
        time.sleep(5)
        
        # Check threads
        check_active_threads()
        
        # Stop monitoring (like in training)
        logger.info("Stopping system monitoring...")
        monitor.stop_monitoring()
        
        # Check threads again
        check_active_threads()
        
    except Exception as e:
        logger.error(f"Error in simulation: {e}")

def main():
    """Main diagnostic function"""
    logger.info("=== Process Hang Diagnostic ===")
    
    # Check initial thread state
    logger.info("=== Initial Thread State ===")
    check_active_threads()
    
    # Check Docker processes
    check_docker_processes()
    
    # Simulate training completion
    simulate_training_completion()
    
    # Final check
    logger.info("=== Final Thread State ===")
    final_threads = check_active_threads()
    
    # Check for non-daemon threads that might prevent exit
    non_daemon_threads = [t for t in final_threads if not t.daemon and t.is_alive() and t != threading.current_thread()]
    
    if non_daemon_threads:
        logger.warning(f"Found {len(non_daemon_threads)} non-daemon threads that could prevent process exit:")
        for thread in non_daemon_threads:
            logger.warning(f"  - {thread.name} (target: {getattr(thread, '_target', 'unknown')})")
    else:
        logger.info("All threads are daemon threads - process should exit normally")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Diagnostic failed: {e}")
        sys.exit(1)
