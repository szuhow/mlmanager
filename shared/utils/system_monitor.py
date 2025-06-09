"""
System monitoring utilities for tracking resource usage during training.
Logs CPU, memory, GPU usage to MLflow for comprehensive training monitoring.
"""

import time
import threading
import mlflow
import logging
from typing import Dict, Optional
import os

# Try to import psutil - if not available, disable system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("psutil not available - system monitoring disabled. Install with: pip install psutil")

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    if PSUTIL_AVAILABLE:  # Only warn if psutil is available
        logger = logging.getLogger(__name__)
        logger.warning("GPUtil not available - GPU monitoring via GPUtil disabled")

try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except (ImportError, Exception):
    NVML_AVAILABLE = False
    if PSUTIL_AVAILABLE:  # Only warn if psutil is available
        logger = logging.getLogger(__name__)
        logger.warning("pynvml not available - advanced GPU monitoring disabled")

logger = logging.getLogger(__name__)

class SystemMonitor:
    """System resource monitor for MLflow logging during training"""
    
    def __init__(self, log_interval: int = 30, enable_gpu: bool = True):
        """
        Initialize system monitor
        
        Args:
            log_interval: Interval in seconds between metric logging
            enable_gpu: Whether to monitor GPU metrics
        """
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available - system monitoring will be disabled")
            self.enabled = False
            return
        
        self.enabled = True
        self.log_interval = log_interval
        self.enable_gpu = enable_gpu and (GPU_AVAILABLE or NVML_AVAILABLE)
        self.monitoring = False
        self.monitor_thread = None
        self.mlflow_run_id = None  # Store the MLflow run ID for thread safety
        self.mlflow_run_id = None  # Store the MLflow run ID for thread safety
        
        # Initialize GPU monitoring if available
        if self.enable_gpu and NVML_AVAILABLE:
            try:
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"GPU monitoring enabled for {self.gpu_count} devices")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")
                self.enable_gpu = False
        elif self.enable_gpu and GPU_AVAILABLE:
            try:
                self.gpus = GPUtil.getGPUs()
                logger.info(f"GPU monitoring enabled for {len(self.gpus)} devices")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")
                self.enable_gpu = False
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        if not PSUTIL_AVAILABLE or not self.enabled:
            logger.warning("System monitoring disabled - psutil not available")
            return {}
            
        metrics = {}
        
        # CPU metrics
        try:
            metrics['system_cpu_percent'] = psutil.cpu_percent(interval=1)
            metrics['system_cpu_count'] = psutil.cpu_count()
            
            # CPU frequency if available
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    metrics['system_cpu_freq_current'] = cpu_freq.current
                    metrics['system_cpu_freq_max'] = cpu_freq.max
            except (AttributeError, OSError):
                pass
        except Exception as e:
            logger.warning(f"Failed to get CPU metrics: {e}")
        
        # Memory metrics
        try:
            memory = psutil.virtual_memory()
            metrics['system_memory_percent'] = memory.percent
            metrics['system_memory_available_gb'] = memory.available / (1024**3)
            metrics['system_memory_used_gb'] = memory.used / (1024**3)
            metrics['system_memory_total_gb'] = memory.total / (1024**3)
        except Exception as e:
            logger.warning(f"Failed to get memory metrics: {e}")
        
        # Disk I/O metrics
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics['system_disk_read_mb'] = disk_io.read_bytes / (1024**2)
                metrics['system_disk_write_mb'] = disk_io.write_bytes / (1024**2)
        except Exception as e:
            logger.warning(f"Failed to get disk I/O metrics: {e}")
        
        # Network I/O metrics
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                metrics['system_network_sent_mb'] = net_io.bytes_sent / (1024**2)
                metrics['system_network_recv_mb'] = net_io.bytes_recv / (1024**2)
        except Exception as e:
            logger.warning(f"Failed to get network I/O metrics: {e}")
        
        # Process-specific metrics
        try:
            process = psutil.Process()
            metrics['process_cpu_percent'] = process.cpu_percent()
            
            memory_info = process.memory_info()
            metrics['process_memory_rss_gb'] = memory_info.rss / (1024**3)
            metrics['process_memory_vms_gb'] = memory_info.vms / (1024**3)
            
            # Memory percentage of total system memory
            total_memory = psutil.virtual_memory().total
            metrics['process_memory_percent'] = (memory_info.rss / total_memory) * 100
            
        except Exception as e:
            logger.warning(f"Failed to get process metrics: {e}")
        
        # GPU metrics
        if self.enable_gpu:
            gpu_metrics = self.get_gpu_metrics()
            metrics.update(gpu_metrics)
        
        return metrics
    
    def get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU metrics"""
        metrics = {}
        
        if NVML_AVAILABLE:
            try:
                for i in range(self.gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics[f'gpu_{i}_utilization_percent'] = util.gpu
                    metrics[f'gpu_{i}_memory_utilization_percent'] = util.memory
                    
                    # GPU memory
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    metrics[f'gpu_{i}_memory_used_gb'] = memory_info.used / (1024**3)
                    metrics[f'gpu_{i}_memory_total_gb'] = memory_info.total / (1024**3)
                    metrics[f'gpu_{i}_memory_free_gb'] = memory_info.free / (1024**3)
                    metrics[f'gpu_{i}_memory_percent'] = (memory_info.used / memory_info.total) * 100
                    
                    # GPU temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        metrics[f'gpu_{i}_temperature_celsius'] = temp
                    except:
                        pass
                    
                    # GPU power usage
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                        metrics[f'gpu_{i}_power_watts'] = power
                    except:
                        pass
                        
            except Exception as e:
                logger.warning(f"Failed to get GPU metrics via NVML: {e}")
                
        elif GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    metrics[f'gpu_{i}_utilization_percent'] = gpu.load * 100
                    metrics[f'gpu_{i}_memory_used_gb'] = gpu.memoryUsed / 1024
                    metrics[f'gpu_{i}_memory_total_gb'] = gpu.memoryTotal / 1024
                    metrics[f'gpu_{i}_memory_free_gb'] = gpu.memoryFree / 1024
                    metrics[f'gpu_{i}_memory_percent'] = gpu.memoryUtil * 100
                    metrics[f'gpu_{i}_temperature_celsius'] = gpu.temperature
                    
            except Exception as e:
                logger.warning(f"Failed to get GPU metrics via GPUtil: {e}")
        
        return metrics
    
    def log_metrics_to_mlflow(self, step: Optional[int] = None):
        """Log current system metrics to MLflow"""
        # Check if we have an active run or stored run_id
        current_run = mlflow.active_run()
        
        if not current_run and not self.mlflow_run_id:
            logger.warning("No active MLflow run and no stored run_id - cannot log system metrics")
            return
        
        # Use stored run_id if no active run (for background thread)
        if not current_run and self.mlflow_run_id:
            try:
                # Start a run context temporarily for logging
                with mlflow.start_run(run_id=self.mlflow_run_id):
                    self._log_metrics_helper(step)
            except Exception as e:
                logger.error(f"Failed to log metrics with stored run_id: {e}")
        else:
            # We have an active run, log directly
            self._log_metrics_helper(step)
    
    def _log_metrics_helper(self, step: Optional[int] = None):
        """Helper method to actually log the metrics"""
        metrics = self.get_system_metrics()
        
        try:
            # Separate system metrics into different categories for better organization
            system_metrics = {}
            hardware_metrics = {}
            process_metrics = {}
            
            for key, value in metrics.items():
                if key.startswith('gpu_') or key.startswith('system_cpu') or key.startswith('system_memory'):
                    # Hardware metrics go to hardware namespace
                    hardware_key = f"hardware.{key}"
                    hardware_metrics[hardware_key] = value
                elif key.startswith('process_'):
                    # Process metrics go to process namespace
                    process_key = f"process.{key.replace('process_', '')}"
                    process_metrics[process_key] = value
                elif key.startswith('system_'):
                    # Other system metrics go to system namespace
                    system_key = f"system.{key.replace('system_', '')}"
                    system_metrics[system_key] = value
                else:
                    # Fallback to system namespace
                    system_metrics[f"system.{key}"] = value
            
            # Log metrics in separate namespaces
            all_metrics = {}
            all_metrics.update(system_metrics)
            all_metrics.update(hardware_metrics)
            all_metrics.update(process_metrics)
            
            if step is not None:
                mlflow.log_metrics(all_metrics, step=step)
            else:
                mlflow.log_metrics(all_metrics)
                
            logger.debug(f"Logged {len(all_metrics)} system metrics to MLflow in organized namespaces")
            
        except Exception as e:
            logger.error(f"Failed to log system metrics to MLflow: {e}")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        step_counter = 0
        
        while self.monitoring:
            try:
                self.log_metrics_to_mlflow(step=step_counter)
                step_counter += 1
                
                # Sleep for the specified interval
                time.sleep(self.log_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.log_interval)  # Continue monitoring even if there's an error
    
    def start_monitoring(self):
        """Start background system monitoring"""
        if not PSUTIL_AVAILABLE or not self.enabled:
            logger.warning("Cannot start monitoring - psutil not available or monitor disabled")
            return
            
        if self.monitoring:
            logger.warning("System monitoring is already running")
            return
        
        # Store the current MLflow run ID for background thread use
        current_run = mlflow.active_run()
        if current_run:
            self.mlflow_run_id = current_run.info.run_id
            logger.info(f"Stored MLflow run ID for monitoring: {self.mlflow_run_id}")
        else:
            logger.warning("No active MLflow run - background monitoring may not work properly")
            
        logger.info(f"Starting system monitoring with {self.log_interval}s interval")
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # Log initial system configuration
        self.log_system_info()
    
    def stop_monitoring(self):
        """Stop background system monitoring"""
        if not self.monitoring:
            return
            
        logger.info("Stopping system monitoring")
        self.monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
    
    def log_system_info(self):
        """Log static system information to MLflow"""
        # Check if we have an active run or stored run_id
        current_run = mlflow.active_run()
        
        if not current_run and not self.mlflow_run_id:
            logger.warning("No active MLflow run and no stored run_id - cannot log system info")
            return
        
        # Use stored run_id if no active run (for background thread)
        if not current_run and self.mlflow_run_id:
            try:
                with mlflow.start_run(run_id=self.mlflow_run_id):
                    self._log_system_info_helper()
            except Exception as e:
                logger.error(f"Failed to log system info with stored run_id: {e}")
        else:
            # We have an active run, log directly
            self._log_system_info_helper()
    
    def _log_system_info_helper(self):
        """Helper method to actually log the system info"""
        if not PSUTIL_AVAILABLE:
            logger.warning("Cannot log system info - psutil not available")
            return
            
        try:
            # System information as parameters
            system_params = {
                'system_platform': os.name,
                'system_cpu_count_logical': psutil.cpu_count(logical=True),
                'system_cpu_count_physical': psutil.cpu_count(logical=False),
                'system_memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'system_python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            }
            
            # GPU information
            if self.enable_gpu and NVML_AVAILABLE:
                try:
                    for i in range(self.gpu_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        
                        system_params[f'gpu_{i}_name'] = name
                        system_params[f'gpu_{i}_memory_total_gb'] = round(memory_info.total / (1024**3), 2)
                        
                except Exception as e:
                    logger.warning(f"Failed to get GPU info: {e}")
            
            # Log as parameters (static info)
            mlflow.log_params(system_params)
            
            # Add tags to categorize this run as having system monitoring
            mlflow.set_tags({
                "monitoring.system_metrics": "enabled",
                "monitoring.type": "comprehensive",
                "monitoring.interval_seconds": str(self.log_interval),
                "monitoring.gpu_enabled": str(self.enable_gpu),
                "system.platform": os.name,
                "system.cpu_count": str(psutil.cpu_count()),
            })
            
            logger.info("Logged system configuration and tags to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log system info to MLflow: {e}")

# Convenience function for easy usage
def create_system_monitor(log_interval: int = 30, enable_gpu: bool = True) -> SystemMonitor:
    """Create and return a SystemMonitor instance"""
    return SystemMonitor(log_interval=log_interval, enable_gpu=enable_gpu)
