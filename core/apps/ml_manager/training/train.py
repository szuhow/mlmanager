import os
import torch.multiprocessing as mp
import logging
import signal
import sys
import threading
from pathlib import Path

# Check for early interruption signals during dataset loading
def check_training_interruption(model_id, phase="training"):
    """Check if training should be interrupted"""
    try:
        from .tasks.tasks import training_stop_flags
        if training_stop_flags.get(model_id, None) and training_stop_flags[model_id].is_set():
            logger.info(f"Training interruption requested during {phase} for model {model_id}")
            raise InterruptedError(f"Training stopped by user during {phase}")
    except ImportError:
        pass  # If tasks module not available, continue
    except Exception as e:
        logger.warning(f"Error checking training interruption: {e}")

def fix_mlflow_container_paths():
    """
    Fix MLflow paths to work correctly in Docker container.
    Prevents MLflow from trying to access host paths like /home/rafal
    """
    # Force MLflow to use container paths
    container_artifacts_root = '/app/core/data/mlflow'
    
    # Set MLflow artifact root to container path
    os.environ['MLFLOW_ARTIFACT_ROOT'] = container_artifacts_root
    os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = container_artifacts_root
    os.environ['MLFLOW_ARTIFACTS_DESTINATION'] = container_artifacts_root
    logging.info(f"[MLFLOW_FIX] Set MLFLOW artifact roots to {container_artifacts_root}")
    
    # Override any host paths that might leak through
    original_home = os.environ.get('HOME', '')
    if original_home and '/home/' in original_home:
        os.environ['HOME'] = '/app'
        logging.info(f"[MLFLOW_FIX] Changed HOME from {original_home} to /app")
    
    # Force HOME to container path regardless
    os.environ['HOME'] = '/app'
    os.environ['USERPROFILE'] = '/app'  # Windows equivalent
    os.environ['USER'] = 'appuser'
    
    # Force temp directories to container paths
    os.environ['TMPDIR'] = '/tmp'
    os.environ['TMP'] = '/tmp'
    os.environ['TEMP'] = '/tmp'
    
    # Monkey patch os.makedirs to prevent host directory creation
    original_makedirs = os.makedirs
    def safe_makedirs(name, mode=0o777, exist_ok=False):
        if isinstance(name, str) and '/home/rafal' in name:
            safe_name = name.replace('/home/rafal', '/app/core/data/mlflow')
            logging.warning(f"[MLFLOW_FIX] Redirected makedirs from {name} to {safe_name}")
            return original_makedirs(safe_name, mode, exist_ok)
        return original_makedirs(name, mode, exist_ok)
    os.makedirs = safe_makedirs
    
    # Import pathlib and patch it too
    try:
        import pathlib
        original_home = pathlib.Path.home
        def safe_home():
            return pathlib.Path('/app')
        pathlib.Path.home = staticmethod(safe_home)
        logging.info("[MLFLOW_FIX] Patched pathlib.Path.home")
    except Exception as e:
        logging.warning(f"[MLFLOW_FIX] Could not patch pathlib: {e}")
    
    logging.info("[MLFLOW_FIX] Applied comprehensive container path fixes for MLflow")

# Apply MLflow container path fixes immediately
fix_mlflow_container_paths()

# Set resource limits to prevent system slowdown
def set_resource_limits():
    """Set CPU and memory limits to prevent system slowdown during training"""
    try:
        # Limit number of threads for various libraries
        os.environ.setdefault('OMP_NUM_THREADS', '2')
        os.environ.setdefault('MKL_NUM_THREADS', '2') 
        os.environ.setdefault('NUMBA_NUM_THREADS', '2')
        os.environ.setdefault('TORCH_NUM_THREADS', '2')
        
        # Set PyTorch to use limited threads
        import torch
        torch.set_num_threads(2)
        
        logging.info("[RESOURCE_LIMITS] Applied thread limits to prevent system slowdown")
        
        # Set process priority to be less aggressive (higher nice value = lower priority)
        try:
            import psutil
            current_process = psutil.Process()
            current_process.nice(10)  # Lower priority
            logging.info("[RESOURCE_LIMITS] Set process priority to nice=10 (lower priority)")
        except Exception as e:
            logging.warning(f"[RESOURCE_LIMITS] Could not set process priority: {e}")
            
    except Exception as e:
        logging.warning(f"[RESOURCE_LIMITS] Error setting resource limits: {e}")

# Apply resource limits
set_resource_limits()

# Force spawn method to prevent zombie processes from DataLoader workers
try:
    mp.set_start_method('spawn', force=True)
    logging.info("[MULTIPROCESSING] Set start method to 'spawn' to prevent zombie processes")
except RuntimeError as e:
    logging.warning(f"[MULTIPROCESSING] Could not set start method: {e}")

# Also ensure torch uses spawn method
import torch
try:
    torch.multiprocessing.set_start_method('spawn', force=True)
    logging.info("[TORCH] Set torch multiprocessing to 'spawn'")
except RuntimeError as e:
    logging.warning(f"[TORCH] Could not set torch start method: {e}")
# Celery status update function
def update_celery_task_status(task_id, state='PROGRESS', meta=None):
    """Update Celery task status from training subprocess"""
    if not task_id:
        return
        
    try:
        # Try to connect to Redis and update task status
        import redis
        from celery import current_app
        from celery.result import AsyncResult
        
        # Get Celery app connection
        redis_client = redis.Redis.from_url(current_app.conf.broker_url)
        
        # Create AsyncResult and update state
        result = AsyncResult(task_id, app=current_app)
        result.update_state(state=state, meta=meta or {})
        
        startup_logger.info(f"[CELERY] Updated task {task_id} status to {state}")
        
    except Exception as e:
        startup_logger.warning(f"[CELERY] Failed to update task status: {e}")

def setup_python_path():
    """Setup comprehensive Python path for container and local environments"""
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Possible root directories 
    possible_roots = [
        # Container paths
        '/app',
        '/app/core',
        # Local development paths
        current_script_dir,
        os.path.dirname(current_script_dir),  # parent dir
        os.path.dirname(os.path.dirname(current_script_dir)),  # grandparent
        os.path.dirname(os.path.dirname(os.path.dirname(current_script_dir))),  # great-grandparent
    ]
    
    # Add paths that should be in sys.path
    paths_to_add = []
    for root in possible_roots:
        if os.path.exists(root):
            # Add root itself
            paths_to_add.append(root)
            
            # Add core directory if it exists
            core_dir = os.path.join(root, 'core')
            if os.path.exists(core_dir):
                paths_to_add.append(core_dir)
            
            # Add apps directory if it exists
            apps_dir = os.path.join(root, 'core', 'apps')
            if os.path.exists(apps_dir):
                paths_to_add.append(apps_dir)
                
            # Add core/apps directory directly
            core_apps_dir = os.path.join(root, 'apps')
            if os.path.exists(core_apps_dir):
                paths_to_add.append(core_apps_dir)
    
    # Add unique paths to sys.path
    added_paths = []
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
            added_paths.append(path)
    
    return added_paths

# Setup Python path immediately
added_paths = setup_python_path()

# Django imports
try:
    from django.conf import settings
    BASE_DIR = settings.BASE_DIR
except ImportError:
    # Fallback for when Django isn't available  
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Use core/data directory for all data storage (keep data contained within core)
CORE_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"

# Global flag for graceful shutdown
STOP_TRAINING = threading.Event()

def signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    global STOP_TRAINING
    try:
        logging.info(f"[SIGNAL] Received signal {signum}, initiating graceful shutdown...")
        
        # Try to end MLflow run gracefully if active
        try:
            import mlflow
            if mlflow.active_run():
                run_id = mlflow.active_run().info.run_id
                logging.info(f"[SIGNAL] Ending MLflow run {run_id} due to signal {signum}")
                
                # Set termination tags
                mlflow.set_tag('training_status', 'terminated_by_signal')
                mlflow.set_tag('termination_signal', str(signum))
                mlflow.set_tag('terminated_at', datetime.now().isoformat())
                
                # End run with appropriate status
                status = 'KILLED' if signum == signal.SIGKILL else 'FAILED'
                mlflow.end_run(status=status)
                logging.info(f"[SIGNAL] MLflow run ended with status: {status}")
        except Exception as mlflow_error:
            try:
                logging.warning(f"[SIGNAL] Failed to end MLflow run gracefully: {mlflow_error}")
            except:
                print(f"[SIGNAL] Failed to end MLflow run gracefully: {mlflow_error}")
        
    except Exception as e:
        try:
            logging.error(f"[SIGNAL] Error in signal handler: {e}")
        except:
            print(f"[SIGNAL] Error in signal handler: {e}")
    
    STOP_TRAINING.set()

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown - only call from main thread"""
    try:
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        logging.info("[SIGNAL] Signal handlers setup successfully")
    except ValueError as e:
        logging.warning(f"[SIGNAL] Could not setup signal handlers: {e}")
    except Exception as e:
        logging.error(f"[SIGNAL] Error setting up signal handlers: {e}")

# --- Prevent duplicate logging ---
_TRAIN_MODULE_INITIALIZED = False

# --- Minimal initial logging setup for startup messages only ---
if not _TRAIN_MODULE_INITIALIZED:
    # Create data directory structure within core
    logging.info("Setting up core data directory structure using CORE_DATA_DIR - {}".format(CORE_DATA_DIR))
    os.makedirs(os.path.join(CORE_DATA_DIR, 'logs'), exist_ok=True)
    _TRAIN_MODULE_INITIALIZED = True

# Create a startup logger that will be replaced with model-specific logging later
startup_logger = logging.getLogger('startup')
startup_logger.setLevel(logging.INFO)
if not startup_logger.handlers:
    startup_handler = logging.FileHandler(os.path.join(CORE_DATA_DIR, 'logs', 'training.log'), mode='a')
    startup_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [STARTUP] %(message)s'))
    startup_logger.addHandler(startup_handler)
    startup_logger.propagate = False

startup_logger.info('--- Training script started ---')

# Log the path setup results
if added_paths:
    startup_logger.info(f"[PATH_SETUP] Added {len(added_paths)} paths to sys.path: {added_paths}")
else:
    startup_logger.info("[PATH_SETUP] No new paths added to sys.path")
startup_logger.info(f"[PATH_SETUP] Current sys.path length: {len(sys.path)}")

import sys
import time
import torch
import torch.optim as optim
import mlflow
import argparse
import json
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import uuid
from datetime import datetime
from torchvision import transforms as tv_transforms  # added for ARCADE transforms
from torch.utils.data import DataLoader as TorchDataLoader  # ARCADE DataLoader
import shutil

def log_artifact_to_model_directory(artifact_path: str, model_directory: str = None, artifact_type: str = None):
    """
    Log artifact primarily to MLflow with optional local copy to model directory.
    
    MLflow is the primary artifact storage system. The model directory copy is for 
    immediate local access and debugging purposes only.
    
    Args:
        artifact_path: Path to the artifact file to log
        model_directory: Optional model directory path for local copy
        artifact_type: Optional artifact type/subfolder name for MLflow organization
    """
    # Add debug logging
    logging.info(f"[ARTIFACT_DEBUG] Called with artifact_path={artifact_path}, model_directory={model_directory}, artifact_type={artifact_type}")
    
    try:
        # Verify MLflow configuration before logging
        tracking_uri = mlflow.get_tracking_uri()
        logging.info(f"[MLFLOW_DEBUG] Tracking URI: {tracking_uri}")
        
        # Check current MLflow run
        current_run = mlflow.active_run()
        if current_run:
            logging.info(f"[MLFLOW_DEBUG] Active run ID: {current_run.info.run_id}")
            logging.info(f"[MLFLOW_DEBUG] Artifact URI: {current_run.info.artifact_uri}")
        else:
            logging.warning("[MLFLOW_DEBUG] No active MLflow run found")
        
        # PRIMARY: Log to MLflow - this is the main storage
        if not os.path.exists(artifact_path):
            logging.warning(f"[MLFLOW] Artifact file not found for MLflow logging: {artifact_path}")
            return
            
        # Additional path validation to prevent host path access
        if '/home/' in artifact_path and not artifact_path.startswith('/app/'):
            logging.error(f"[MLFLOW_ERROR] Dangerous host path detected: {artifact_path}")
            logging.error("[MLFLOW_ERROR] Artifact path should be within container (/app/)")
            return
            
        if artifact_type:
            mlflow.log_artifact(artifact_path, artifact_path=artifact_type)
            logging.info(f"[MLFLOW] ✅ Logged artifact to MLflow: {artifact_type}/{os.path.basename(artifact_path)}")
        else:
            mlflow.log_artifact(artifact_path)
            logging.info(f"[MLFLOW] ✅ Logged artifact to MLflow: {os.path.basename(artifact_path)}")
        
        # SECONDARY: Optional local copy to model directory for immediate access
        if model_directory:
            logging.info(f"[LOCAL_DEBUG] Attempting local copy to model_directory: {model_directory}")
            try:
                # Create artifacts directory in model folder
                model_artifacts_dir = os.path.join(model_directory, 'artifacts')
                os.makedirs(model_artifacts_dir, exist_ok=True)
                logging.info(f"[LOCAL_DEBUG] Created artifacts directory: {model_artifacts_dir}")
                
                # Construct destination path
                if artifact_type:
                    # Create subfolder structure in model artifacts directory
                    dest_dir = os.path.join(model_artifacts_dir, artifact_type)
                    os.makedirs(dest_dir, exist_ok=True)
                    dest_path = os.path.join(dest_dir, os.path.basename(artifact_path))
                else:
                    dest_path = os.path.join(model_artifacts_dir, os.path.basename(artifact_path))
                
                logging.info(f"[LOCAL_DEBUG] Destination path: {dest_path}")
                
                # Copy the artifact locally
                shutil.copy2(artifact_path, dest_path)
                logging.info(f"[LOCAL] ✅ Copied artifact to model directory: {dest_path}")
                
            except Exception as local_error:
                logging.warning(f"[LOCAL] Failed to copy artifact to model directory: {local_error}")
                # Continue - local copy failure doesn't affect MLflow logging
        else:
            logging.info(f"[LOCAL_DEBUG] Skipping local copy - model_directory is None")
            
    except Exception as e:
        logging.error(f"[MLFLOW] Failed to log artifact to MLflow: {artifact_path} - {e}")
        raise  # Re-raise since MLflow logging is critical

# === Early Django Setup ===
# Setup Django early to avoid import issues with training_callback
DJANGO_AVAILABLE = False
try:
    import django
    
    # Ensure the project root is in Python path
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    core_path = str(project_root / 'core')
    
    if core_path not in sys.path:
        sys.path.insert(0, core_path)
    
    # Set the Django settings module - use container settings in Docker environment
    if os.environ.get('DJANGO_SETTINGS_MODULE'):
        # Use existing environment setting (likely from container)
        startup_logger.info(f"[DJANGO] Using existing settings module: {os.environ.get('DJANGO_SETTINGS_MODULE')}")
    else:
        # Default to development settings
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
        startup_logger.info("[DJANGO] Using default development settings")
    
    # Setup Django
    try:
        django.setup()
        DJANGO_AVAILABLE = True
        startup_logger.info("[DJANGO] Django setup completed successfully")
    except RuntimeError as e:
        if "populated" in str(e):
            DJANGO_AVAILABLE = True
            startup_logger.info("[DJANGO] Django already configured")
        else:
            startup_logger.error(f"[DJANGO] Django setup failed: {e}")
            # Don't raise, just continue without Django
        
except ImportError:
    startup_logger.warning("[DJANGO] Django not available in current environment")
    startup_logger.warning("[DJANGO] Training callback will not be available")
except Exception as e:
    startup_logger.warning(f"[DJANGO] Django setup failed: {e}")
    startup_logger.warning("[DJANGO] Training callback will not be available")

from monai.networks.nets import UNet as MonaiUNet
from monai.data import CacheDataset, DataLoader as MonaiDataLoader
from monai.data import Dataset, DataLoader
from monai.transforms import (
    LoadImage, ScaleIntensity, ToTensor, Compose,
    Resize, EnsureChannelFirst, ConvertToMultiChannelBasedOnBratsClassesd,
    Lambda
)

# Fix import issues by trying multiple import approaches
import os
import sys
import importlib.util
import traceback

# Add multiple possible paths to Python path
paths_to_try = [
    # Current directory structure
    os.path.abspath(os.path.dirname(__file__)),
    # Direct parent
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
    # Project root assuming standard structure
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')),
    # /app path for Docker container
    '/app',
    # Container app path
    '/app/core'
]

for path in paths_to_try:
    if path not in sys.path:
        sys.path.insert(0, path)
        startup_logger.info(f"[IMPORT] Added path to sys.path: {path}")

# Universal import helper
def flexible_import(module_names, from_list=None, attr_name=None):
    """
    Flexible import helper that tries multiple import paths
    
    Args:
        module_names: List of module paths to try importing
        from_list: List of attributes to import from the module
        attr_name: Optional attribute to extract from module
        
    Returns:
        The imported module or attribute, or None if import failed
    """
    last_error = None
    
    # Try each module name
    for module_name in module_names:
        try:
            startup_logger.info(f"[IMPORT] Trying import: {module_name}")
            
            # Handle direct path import with importlib
            if module_name.endswith('.py'):
                module_path = module_name
                module_name = os.path.basename(module_path).replace('.py', '')
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    startup_logger.info(f"[IMPORT] Successfully imported {module_name} using file path")
                    
                    # Return requested attribute if specified
                    if attr_name and hasattr(module, attr_name):
                        return getattr(module, attr_name)
                    return module
            else:
                # Standard import
                if from_list:
                    module = __import__(module_name, globals(), locals(), from_list)
                    # Return requested attribute if specified
                    if attr_name and hasattr(module, attr_name):
                        return getattr(module, attr_name)
                    
                    # Return specific attribute from from_list if requested
                    if len(from_list) == 1 and not attr_name:
                        attr = from_list[0]
                        if hasattr(module, attr):
                            return getattr(module, attr)
                            
                    startup_logger.info(f"[IMPORT] Successfully imported {module_name}")
                    return module
                else:
                    module = __import__(module_name)
                    # Handle dot notation imports
                    components = module_name.split('.')
                    for comp in components[1:]:
                        module = getattr(module, comp)
                        
                    # Return requested attribute if specified
                    if attr_name and hasattr(module, attr_name):
                        return getattr(module, attr_name)
                        
                    startup_logger.info(f"[IMPORT] Successfully imported {module_name}")
                    return module
                    
        except (ImportError, AttributeError, ModuleNotFoundError) as e:
            startup_logger.warning(f"[IMPORT] Failed to import {module_name}: {str(e)}")
            last_error = e
        except Exception as e:
            startup_logger.warning(f"[IMPORT] Unexpected error importing {module_name}: {str(e)}")
            last_error = e
            
    # If we get here, all imports failed
    startup_logger.error(f"[IMPORT] All import attempts failed. Last error: {last_error}")
    startup_logger.error(f"[IMPORT] sys.path = {sys.path}")
    return None

# First try relative import
architecture_registry = None
startup_logger.info("[IMPORT] Attempting imports for architecture_registry")

# Try with our flexible import helper
module_paths = [
    # Relative imports
    '..utils.architecture_registry',
    # Absolute import assuming core.apps structure
    'core.apps.ml_manager.utils.architecture_registry',
    # Absolute import from current package
    'ml_manager.utils.architecture_registry',
    # App-absolute imports
    'apps.ml_manager.utils.architecture_registry',
    # Direct path-based import (full path)
    os.path.join(os.path.dirname(__file__), "../utils/architecture_registry.py"),
    # Container paths
    '/app/core/apps/ml_manager/utils/architecture_registry.py'
]

# Import the registry attribute from architecture_registry module
architecture_registry = flexible_import(module_paths, from_list=['registry'], attr_name='registry')

if architecture_registry is None:
    # Last resort - try to find the file directly and load it
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "../utils/architecture_registry.py"),
        os.path.join(os.path.dirname(__file__), "../../utils/architecture_registry.py"),
        "/app/core/apps/ml_manager/utils/architecture_registry.py"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            startup_logger.info(f"[IMPORT] Found architecture_registry at: {path}")
            try:
                spec = importlib.util.spec_from_file_location("architecture_registry", path)
                if spec is not None:
                    arch_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(arch_module)
                    if hasattr(arch_module, 'registry'):
                        architecture_registry = arch_module.registry
                        startup_logger.info(f"[IMPORT] Successfully imported architecture_registry from {path}")
                        break
            except Exception as e:
                startup_logger.warning(f"[IMPORT] Failed to load {path}: {e}")
                
if architecture_registry is None:
    startup_logger.error("[IMPORT] All import methods failed for architecture_registry")
    startup_logger.error(f"[IMPORT] sys.path = {sys.path}")
    raise ImportError("Failed to import architecture_registry module")

# Utility function for flexible module importing
def import_module_flexibly(module_path, module_name, error_tag):
    """Import a module flexibly using different approaches"""
    # Try different import paths - expanded for better compatibility
    import_paths = [
        # Original absolute path
        f"core.apps.ml_manager.{module_path}",
        # Relative from current file
        f"..{module_path}",
        # Direct path without core.apps prefix
        f"ml_manager.{module_path}",
        # Just module name
        module_path,
        # Just the last component
        module_path.split(".")[-1],
        # App-relative path
        f"apps.ml_manager.{module_path}"
    ]
    
    # Add direct file paths to try
    file_paths = [
        os.path.join(os.path.dirname(__file__), f"../{module_path.replace('.', '/')}.py"),
        os.path.join(os.path.dirname(__file__), f"../../{module_path.replace('.', '/')}.py"),
        f"/app/core/apps/ml_manager/{module_path.replace('.', '/')}.py",
        f"/app/{module_path.replace('.', '/')}.py"
    ]
    
    # First try module import approaches
    for path in import_paths:
        try:
            if "." in path:
                package_parts = path.split(".")
                fromlist = [package_parts[-1]]
                if len(package_parts) > 1:
                    package_path = ".".join(package_parts[:-1])
                else:
                    package_path = package_parts[0]
                    fromlist = []
                
                module = __import__(package_path, fromlist=fromlist)
                
                # Traverse through module parts
                for part in package_parts[1:]:
                    module = getattr(module, part)
                    
                startup_logger.info(f"[IMPORT] Successfully imported {module_name} from {path}")
                return module, True
        except (ImportError, AttributeError) as e:
            startup_logger.warning(f"[IMPORT] Failed to import {module_name} from {path}: {e}")
    
    # Then try file-based import approaches
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    startup_logger.info(f"[IMPORT] Successfully imported {module_name} from file {file_path}")
                    return module, True
        except Exception as e:
            startup_logger.warning(f"[IMPORT] Failed to import {module_name} from file {file_path}: {e}")
    
    startup_logger.error(f"[{error_tag}] Failed to import {module_name}, tried module paths: {import_paths}")
    startup_logger.error(f"[{error_tag}] Tried file paths: {file_paths}")
    return None, False

# Import dynamic learning rate scheduler
dynamic_lr_mod, DYNAMIC_LR_AVAILABLE = import_module_flexibly("utils.dynamic_lr_scheduler", "DynamicLearningRateScheduler", "DYNAMIC_LR")
if DYNAMIC_LR_AVAILABLE:
    DynamicLearningRateScheduler = dynamic_lr_mod.DynamicLearningRateScheduler
    startup_logger.info("[DYNAMIC_LR] Dynamic learning rate scheduler available")
else:
    startup_logger.warning("[DYNAMIC_LR] Dynamic learning rate scheduler not available")

# Import early stopping
early_stopping_mod, EARLY_STOPPING_AVAILABLE = import_module_flexibly("utils.early_stopping", "EarlyStopping", "EARLY_STOPPING") 
if EARLY_STOPPING_AVAILABLE:
    EarlyStopping = early_stopping_mod.EarlyStopping
    startup_logger.info("[EARLY_STOPPING] Early stopping available")
else:
    startup_logger.warning("[EARLY_STOPPING] Early stopping not available")

# Try to import ARCADE dataset integration
arcade_mod, ARCADE_AVAILABLE = import_module_flexibly("datasets.torch_arcade_loader", "arcade_loader", "ARCADE")
if ARCADE_AVAILABLE:
    create_arcade_dataloader = arcade_mod.create_arcade_dataloader
    get_arcade_dataset_info = arcade_mod.get_arcade_dataset_info
    startup_logger.info("[ARCADE] ARCADE dataset integration available")
else:
    startup_logger.warning("[ARCADE] ARCADE dataset not available")

try:
    from monai.transforms import AddChanneld, EnsureChannelFirstd
    AddChannelTransform = AddChanneld
    EnsureChannelTransform = EnsureChannelFirstd
except ImportError:
    from monai.transforms import EnsureChannelFirstd
    AddChannelTransform = EnsureChannelFirstd
    EnsureChannelTransform = EnsureChannelFirstd

from monai.transforms import Compose, LoadImaged, ScaleIntensityd, ToTensord, RandCropByPosNegLabeld, RandFlipd, RandRotate90d, RandScaleIntensityd, Lambdad, Resized, RandSpatialCropd
from monai.losses import DiceLoss as MonaiDiceLoss
from monai.metrics import DiceMetric, MeanIoU

# Import training callback
callback_mod, TRAINING_CALLBACK_AVAILABLE = import_module_flexibly("utils.training_callback", "TrainingCallback", "TRAINING_CALLBACK")
if TRAINING_CALLBACK_AVAILABLE:
    TrainingCallback = callback_mod.TrainingCallback
    startup_logger.info("[TRAINING_CALLBACK] Training callback available")
else:
    startup_logger.warning("[TRAINING_CALLBACK] Training callback not available")
    
    # Create a dummy TrainingCallback class for compatibility
    class DummyTrainingCallback:
        def __init__(self, *args, **kwargs):
            startup_logger.warning("[DUMMY_CALLBACK] Using dummy training callback")
            # Add model attribute to prevent AttributeError
            self.model = type('DummyModel', (), {'stop_requested': False})()
        
        def update_progress(self, *args, **kwargs):
            pass
        
        def update_status(self, *args, **kwargs):
            pass
            
        def update_metrics(self, *args, **kwargs):
            pass
            
        def finalize(self, *args, **kwargs):
            pass
            
        def set_model_directory(self, *args, **kwargs):
            pass
            
        def on_training_start(self, *args, **kwargs):
            pass
            
        def on_epoch_end(self, *args, **kwargs):
            pass
            
        def on_training_end(self, *args, **kwargs):
            pass
            
        def on_training_error(self, *args, **kwargs):
            pass
            
        # Dodane brakujące metody na podstawie logów błędów
        def on_dataset_loaded(self, *args, **kwargs):
            pass
            
        def on_training_failed(self, *args, **kwargs):
            pass
            
        def on_training_stopped(self, *args, **kwargs):
            pass
            
        def on_epoch_start(self, *args, **kwargs):
            return True
            
        def on_batch_start(self, *args, **kwargs):
            return True
            
        def on_batch_end(self, *args, **kwargs):
            pass
            
        def update_model_metadata(self, *args, **kwargs):
            pass
            
        def update_training_config(self, *args, **kwargs):
            pass
            
        def update_architecture_info(self, *args, **kwargs):
            pass
            
        def set_epoch_batches(self, *args, **kwargs):
            pass
            
    TrainingCallback = DummyTrainingCallback
    startup_logger.info("[TRAINING_CALLBACK] Created dummy training callback for compatibility")

# Import advanced losses
advanced_losses_mod, ADVANCED_LOSSES_AVAILABLE = import_module_flexibly("utils.advanced_losses", "advanced_losses", "ADVANCED_LOSSES")
if ADVANCED_LOSSES_AVAILABLE:
    TverskyLoss = advanced_losses_mod.TverskyLoss
    FocalLoss = advanced_losses_mod.FocalLoss
    ComboDiceBCELoss = advanced_losses_mod.ComboDiceBCELoss
    SoftDiceLoss = advanced_losses_mod.SoftDiceLoss
    WeightedBCELoss = advanced_losses_mod.WeightedBCELoss
    BoundaryLoss = advanced_losses_mod.BoundaryLoss
    StableBCELoss = advanced_losses_mod.StableBCELoss
    create_advanced_loss = advanced_losses_mod.create_advanced_loss
    get_recommended_loss = advanced_losses_mod.get_recommended_loss
    startup_logger.info("[ADVANCED_LOSSES] Advanced loss functions available")
else:
    startup_logger.warning("[ADVANCED_LOSSES] Advanced loss functions not available")

# Import medical preprocessing
med_preprocessing_mod, MEDICAL_PREPROCESSING_AVAILABLE = import_module_flexibly("utils.medical_preprocessing", "medical_preprocessing", "MEDICAL_PREPROCESSING")
if MEDICAL_PREPROCESSING_AVAILABLE:
    MedicalImagePreprocessor = med_preprocessing_mod.MedicalImagePreprocessor
    preprocess_angiography = med_preprocessing_mod.preprocess_angiography
    preprocess_ct_coronary = med_preprocessing_mod.preprocess_ct_coronary
    preprocess_oct_coronary = med_preprocessing_mod.preprocess_oct_coronary
    startup_logger.info("[MEDICAL_PREPROCESSING] Medical preprocessing available")
else:
    startup_logger.warning("[MEDICAL_PREPROCESSING] Medical preprocessing not available")

# Import loss manager
loss_manager_mod, LOSS_MANAGER_AVAILABLE = import_module_flexibly("utils.loss_manager", "LossManager", "LOSS_MANAGER")
if LOSS_MANAGER_AVAILABLE:
    LossManager = loss_manager_mod.LossManager
    startup_logger.info("[LOSS_MANAGER] Loss manager available")
else:
    startup_logger.warning("[LOSS_MANAGER] Loss manager not available")

# Global logger for architecture functions
logger = logging.getLogger(__name__)

# Utility functions moved to top to avoid NameError
def create_organized_model_directory(model_id=None, model_family="UNET", version="1.0.0", mlflow_run_id=None):
    """Create an organized directory structure for model storage"""
    
    # Generate unique identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Include MLflow run ID in the unique identifier for better traceability
    if mlflow_run_id:
        mlflow_short_id = str(mlflow_run_id)[:8]  # Use first 8 characters of MLflow run ID
        unique_id = f"{model_family.replace(' ', '_').lower()}_{timestamp}_{mlflow_short_id}_{str(uuid.uuid4())[:8]}"
    else:
        unique_id = f"{model_family.replace(' ', '_').lower()}_{timestamp}_{str(uuid.uuid4())[:8]}"
    
    # Create date-based organization
    date_str = datetime.now().strftime("%Y/%m")
    family_str = model_family.replace(" ", "_").lower()
    
    # Use CORE_DATA_DIR to ensure consistent path resolution within core
    model_dir = os.path.join(
        CORE_DATA_DIR,
        "models",
        "organized", 
        date_str,
        family_str,
        f"{unique_id}_v{version}"
    )
    
    # Get absolute path for consistent access
    model_dir = os.path.abspath(model_dir)
    
    # Create the directory structure immediately to ensure it exists
    # This ensures proper folder generation for each training
    try:
        os.makedirs(model_dir, exist_ok=True)
        
        # Create essential subdirectories
        subdirs = ['logs', 'checkpoints', 'artifacts', 'predictions', 'configs', 'weights', 'metrics']
        for subdir in subdirs:
            subdir_path = os.path.join(model_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            
        startup_logger.info(f"✅ Created organized model directory: {model_dir}")
        startup_logger.info(f"📁 Created subdirectories: {subdirs}")
        
        # Create a README file to document the model
        readme_path = os.path.join(model_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(f"""# Model: {unique_id}

## Model Information
- **Family**: {model_family}
- **Version**: {version}
- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Unique ID**: {unique_id}

## Directory Structure
- `logs/`: Training and validation logs
- `checkpoints/`: Model checkpoints and weights
- `artifacts/`: Training artifacts and visualizations
- `predictions/`: Sample predictions and comparisons
- `configs/`: Training configuration files
- `weights/`: Model weight files
- `metrics/`: Training metrics and performance data

## Model Training
Model ID: {model_id if model_id else 'TBD'}
Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
        
        startup_logger.info(f"📄 Created model README: {readme_path}")
        
    except Exception as e:
        startup_logger.warning(f"❌ Failed to create organized model directory {model_dir}: {e}")
        # Fallback to basic directory creation
        os.makedirs(model_dir, exist_ok=True)
    
    return model_dir, unique_id

def ensure_model_directories(model_dir):
    """Create model subdirectories when actually needed"""
    subdirs = ["weights", "config", "artifacts", "predictions", "metrics", "logs"]
    for subdir in subdirs:
        os.makedirs(os.path.join(model_dir, subdir), exist_ok=True)

def save_enhanced_training_curves(epoch_history, model_dir, epoch):
    """Save comprehensive training curves with multiple metrics"""
    try:
        if len(epoch_history) < 2:
            return None
        
        # Ensure model directories exist when we actually need them
        ensure_model_directories(model_dir)
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Progress - Epoch {epoch+1}', fontsize=16)
        
        epochs = list(range(1, len(epoch_history) + 1))
        
        # Extract metrics from history
        train_losses = [h['train_loss'] for h in epoch_history]
        val_losses = [h['val_loss'] for h in epoch_history]
        train_dices = [h['train_dice'] for h in epoch_history]
        val_dices = [h['val_dice'] for h in epoch_history]
        
        # Loss curves
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        # axes[0, 0].grid(True, alpha=0.3)  # Usuń siatkę z wykresów
        
        # Dice score curves
        axes[0, 1].plot(epochs, train_dices, 'b-', label='Training Dice', linewidth=2)
        axes[0, 1].plot(epochs, val_dices, 'r-', label='Validation Dice', linewidth=2)
        axes[0, 1].set_title('Dice Score Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        # axes[0, 1].grid(True, alpha=0.3)  # Usuń siatkę z wykresów
        
        # Loss difference (overfitting indicator)
        loss_diff = [v - t for v, t in zip(val_losses, train_losses)]
        axes[0, 2].plot(epochs, loss_diff, 'g-', linewidth=2)
        axes[0, 2].set_title('Validation - Training Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss Difference')
        # axes[0, 2].grid(True, alpha=0.3)  # Usuń siatkę z wykresów
        axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Dice difference
        dice_diff = [v - t for v, t in zip(val_dices, train_dices)]
        axes[1, 0].plot(epochs, dice_diff, 'purple', linewidth=2)
        axes[1, 0].set_title('Validation - Training Dice')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice Difference')
        # axes[1, 0].grid(True, alpha=0.3)  # Usuń siatkę z wykresów
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Best metrics summary
        best_val_dice = max(val_dices)
        best_epoch = val_dices.index(best_val_dice) + 1
        
        summary_text = f"""
        Current Epoch: {epoch + 1}
        Best Val Dice: {best_val_dice:.4f} (Epoch {best_epoch})
        Current Val Dice: {val_dices[-1]:.4f}
        Current Train Dice: {train_dices[-1]:.4f}
        
        Current Val Loss: {val_losses[-1]:.4f}
        Current Train Loss: {train_losses[-1]:.4f}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
        
        # Learning progress (smoothed)
        if len(val_dices) > 5:
            # Simple moving average for trend
            window = min(5, len(val_dices))
            val_dice_smooth = []
            for i in range(len(val_dices)):
                start = max(0, i - window + 1)
                val_dice_smooth.append(np.mean(val_dices[start:i+1]))
            
            axes[1, 2].plot(epochs, val_dices, 'r--', alpha=0.5, label='Raw Validation Dice')
            axes[1, 2].plot(epochs, val_dice_smooth, 'r-', linewidth=2, label='Smoothed Validation Dice')
            axes[1, 2].set_title('Learning Progress (Smoothed)')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Dice Score')
            axes[1, 2].legend()
            # axes[1, 2].grid(True, alpha=0.3)  # Usuń siatkę z wykresów
        else:
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save in artifacts subdirectory
        artifacts_dir = os.path.join(model_dir, "artifacts")
        filename = os.path.join(artifacts_dir, f'enhanced_training_curves_epoch_{epoch+1}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    except Exception as e:
        print(f"Could not save enhanced training curves: {e}")
        return None

def get_default_model_config(model_type, args=None):
    """Get default configuration for a model type with custom architecture options"""
    logger.info(f"[CONFIG] get_default_model_config called with model_type={model_type}")
    logger.info(f"[CONFIG] args provided: {args is not None}")
    
    # Default configurations for common architectures
    default_configs = {
        'unet': {
            'spatial_dims': 2,
            'in_channels': 1,
            'out_channels': 1,
            'channels': (16, 32, 64, 128, 256),
            'strides': (2, 2, 2, 2),
            'num_res_units': 2,
        },
        'configurable_monai_unet': {
            'spatial_dims': 2,
            'in_channels': 1,
            'out_channels': 1,
            'channels': (16, 32, 64, 128, 256),
            'strides': (2, 2, 2, 2),
            'num_res_units': 2,
        },
        'monai_unet': {
            'spatial_dims': 2,
            'in_channels': 1,
            'out_channels': 1,
            'channels': (16, 32, 64, 128, 256),
            'strides': (2, 2, 2, 2),
            'num_res_units': 2,
        }
    }
    
    # Get base config
    base_config = default_configs.get(model_type, {}).copy()
    logger.info(f"[CONFIG] Base config for {model_type}: {base_config}")
    
    # Apply model architecture customizations if args are provided
    if args:
        # Check different ways args might contain model_size/custom_channels
        model_size = None
        custom_channels = None
        
        # Method 1: Direct attributes on args
        if hasattr(args, 'model_size'):
            model_size = args.model_size
            logger.info(f"[CONFIG] Found model_size in args: {model_size}")
        if hasattr(args, 'custom_channels'):
            custom_channels = args.custom_channels
            logger.info(f"[CONFIG] Found custom_channels in args: {custom_channels}")
            
        # Method 2: model_architecture dict
        if hasattr(args, 'model_architecture') and args.model_architecture:
            model_arch = args.model_architecture
            model_size = model_arch.get('model_size', model_size)
            custom_channels = model_arch.get('custom_channels', custom_channels)
            logger.info(f"[CONFIG] From model_architecture - model_size: {model_size}, custom_channels: {custom_channels}")
        
        # Method 3: Check if args is a dict
        if isinstance(args, dict):
            model_size = args.get('model_size', model_size)
            custom_channels = args.get('custom_channels', custom_channels)
            logger.info(f"[CONFIG] From args dict - model_size: {model_size}, custom_channels: {custom_channels}")
        
        # Apply size-based channel mapping
        if model_size and model_size != 'standard':
            # Size mapping that matches JavaScript in UI
            size_mapping = {
                'micro': (8, 16, 32, 64),           # ~100K parameters
                'tiny': (16, 32, 64, 128, 256),     # ~1.6M parameters
                'small': (32, 64, 128, 256, 512),   # ~6.5M parameters
                'standard': (32, 64, 128, 256, 512), # Standard size
                'large': (32, 64, 128, 256, 512),   # Large with attention
                'xl': (64, 128, 256, 512, 1024)     # XL size
            }
            
            if model_size in size_mapping:
                base_config['channels'] = size_mapping[model_size]
                logger.info(f"[CONFIG] Applied size mapping {model_size} -> channels: {base_config['channels']}")
            
        # Apply custom channels if provided (overrides size mapping)
        if custom_channels:
            try:
                if isinstance(custom_channels, str):
                    # Parse custom channels string like "8,16,32,64"
                    channels = tuple(int(x.strip()) for x in custom_channels.split(','))
                    base_config['channels'] = channels
                    logger.info(f"[CONFIG] Applied custom channels from string: {channels}")
                elif isinstance(custom_channels, (list, tuple)):
                    base_config['channels'] = tuple(custom_channels)
                    logger.info(f"[CONFIG] Applied custom channels from list/tuple: {base_config['channels']}")
            except (ValueError, AttributeError) as e:
                logger.warning(f"[CONFIG] Invalid custom channels '{custom_channels}': {e}")
    
    
    # Check if architecture has default config from registry
    try:
        # Import architecture registry with fallback paths
        arch_registry_mod, ARCH_REGISTRY_AVAILABLE = import_module_flexibly("utils.architecture_registry", "registry", "ARCH_REGISTRY")
        if ARCH_REGISTRY_AVAILABLE:
            architecture_registry = arch_registry_mod.registry
            arch_info = architecture_registry.get_architecture(model_type)
            if arch_info and arch_info.default_config:
                # Merge registry config with our customizations
                registry_config = arch_info.default_config.copy()
                registry_config.update(base_config)
                logger.info(f"[CONFIG] Using registry config merged with customizations: {registry_config}")
                return registry_config
    except:
        pass
    
    logger.info(f"[CONFIG] Final config for {model_type}: {base_config}")
    return base_config

def validate_architecture(model_type):
    """Validate that the specified architecture is available"""
    logger.info(f"Validating architecture: {model_type}")
    
    # Check if architecture is registered
    arch_info = architecture_registry.get_architecture(model_type)
    if not arch_info:
        logger.error(f"Architecture '{model_type}' not found in registry")
        available = [key for key in architecture_registry.get_all_architectures().keys()]
        logger.error(f"Available architectures: {available}")
        raise ValueError(f"Unknown architecture: {model_type}. Available: {available}")
    
    # Validate the architecture
    is_valid, message = architecture_registry.validate_architecture(model_type)
    if not is_valid:
        logger.error(f"Architecture validation failed: {message}")
        raise ValueError(f"Invalid architecture '{model_type}': {message}")
    
    logger.info(f"Architecture '{model_type}' validated successfully")
    return arch_info

# --- Patch: Add detailed logging around model architecture loading ---
def create_model_from_registry(model_type, device, task_type=None, **model_kwargs):
    """Create a model instance using the architecture registry"""
    # Ensure architecture_registry is available
    if architecture_registry is None:
        error_msg = "Architecture registry not available - import failed"
        logger.error(f"[ARCH] {error_msg}")
        raise ImportError(error_msg)
        
    logger.info(f"[ARCH] Attempting to load model_type: {model_type} with kwargs: {model_kwargs}")
    logger.info(f"[ARCH] Task type: {task_type}")
    try:
        logger.info(f"[ARCH] Available architectures in registry: {list(architecture_registry._architectures.keys())}")
        logger.info(f"[ARCH] Model creation requested: model_type={model_type}")
        logger.info(f"[ARCH] Model kwargs before mapping: {model_kwargs}")
        # If channels or custom_channels present, log them explicitly
        if 'channels' in model_kwargs:
            logger.info(f"[ARCH] channels: {model_kwargs['channels']}")
        if 'custom_channels' in model_kwargs:
            logger.info(f"[ARCH] custom_channels: {model_kwargs['custom_channels']}")
        if 'n_channels' in model_kwargs:
            logger.info(f"[ARCH] n_channels: {model_kwargs['n_channels']}")
        if 'in_channels' in model_kwargs:
            logger.info(f"[ARCH] in_channels: {model_kwargs['in_channels']}")
        if 'out_channels' in model_kwargs:
            logger.info(f"[ARCH] out_channels: {model_kwargs['out_channels']}")
        if 'n_classes' in model_kwargs:
            logger.info(f"[ARCH] n_classes: {model_kwargs['n_classes']}")
    except Exception as e:
        logger.warning(f"[ARCH] Could not list available architectures: {e}")
    
    try:
        # Check if this is a classification task and adapt model selection
        if task_type == 'artery_classification':
            logger.info("[ARCH] Classification task detected - selecting appropriate classifier")
            
            # Map segmentation models to their classification counterparts
            classification_model_map = {
                'unet': 'unet_classifier',
                'monai_unet': 'unet_classifier',
                'resunet': 'resunet_classifier',
                'deep_resunet': 'deep_resunet_classifier',
                'resunet_attention': 'resunet_attention_classifier',
                'deep_resunet_attention': 'resunet_attention_classifier'
            }
            
            # Get the classification model type
            classification_model = classification_model_map.get(model_type, 'unet_classifier')
            logger.info(f"[ARCH] Mapping {model_type} -> {classification_model} for classification")
            
            # Validate and get classification architecture
            arch_info = validate_architecture(classification_model)
            model_class = arch_info.model_class
            
            # Apply default config for classification
            if arch_info.default_config:
                logger.info(f"[ARCH] Applying default config for {classification_model}: {arch_info.default_config}")
                current_model_kwargs = arch_info.default_config.copy()
                current_model_kwargs.update(model_kwargs) # User-provided kwargs override defaults
            else:
                current_model_kwargs = model_kwargs
            
            # Ensure classification parameters
            current_model_kwargs['n_classes'] = model_kwargs.get('out_channels', 2)  # Map out_channels to n_classes
            if 'out_channels' in current_model_kwargs:
                del current_model_kwargs['out_channels']  # Remove segmentation parameter
            
            # Fix channel configuration for artery classification
            if task_type == 'artery_classification':
                # Artery classification uses binary masks (1 channel input)
                current_model_kwargs['n_channels'] = model_kwargs.get('in_channels', 1)
                logger.info(f"[ARCH] Override n_channels for artery classification: {current_model_kwargs['n_channels']}")
            
            # Ensure we have the correct input channels configuration
            if 'in_channels' in model_kwargs:
                current_model_kwargs['n_channels'] = model_kwargs['in_channels']
                logger.info(f"[ARCH] Using explicit in_channels: {current_model_kwargs['n_channels']}")
            
            # Create classification model instance
            logger.info(f"[ARCH] Instantiating {arch_info.display_name} with effective kwargs: {current_model_kwargs}")
            model = model_class(**current_model_kwargs)
            logger.info(f"[ARCH] Successfully loaded classification model: {arch_info.display_name}")
            
        # Handle special case for MONAI UNet (legacy compatibility) for segmentation
        elif model_type in ['unet', 'monai_unet', 'configurable_monai_unet']:
            logger.info("[ARCH] Using MONAI UNet with default configuration (special case)")
            
            # Get proper configuration including size-based channels
            default_config = get_default_model_config(model_type, args=None)  # We'll pass kwargs instead
            logger.info(f"[ARCH] Default config from get_default_model_config: {default_config}")
            
            # Override with any user-provided kwargs
            final_config = default_config.copy()
            final_config.update(model_kwargs)
            logger.info(f"[ARCH] Final config after merging with model_kwargs: {final_config}")
            
            model = MonaiUNet(
                spatial_dims=final_config.get('spatial_dims', 2),
                in_channels=final_config.get('in_channels', 1),
                out_channels=final_config.get('out_channels', 1),
                channels=final_config.get('channels', (16, 32, 64, 128, 256)),
                strides=final_config.get('strides', (2, 2, 2, 2)),
                num_res_units=final_config.get('num_res_units', 2),
            )
            logger.info(f"[ARCH] Created MONAI UNet with channels: {final_config.get('channels')}")
            # Create a fake arch_info for unet
            from types import SimpleNamespace
            arch_info = SimpleNamespace(
                display_name="MONAI UNet",
                framework="PyTorch",
                key="monai_unet",
                version="1.0.0",
                description="MONAI U-Net for medical image segmentation"
            )
            logger.info(f"[ARCH] Successfully loaded model: {arch_info.display_name}")
        else:
            # Validate architecture and use from registry
            logger.info(f"[ARCH] Validating architecture: {model_type}")
            arch_info = validate_architecture(model_type)
            model_class = arch_info.model_class
            
            # Apply default config if available
            if arch_info.default_config:
                logger.info(f"[ARCH] Applying default config for {model_type}: {arch_info.default_config}")
                # Make a copy to avoid modifying the original default_config in the registry
                current_model_kwargs = arch_info.default_config.copy()
                current_model_kwargs.update(model_kwargs) # User-provided kwargs override defaults
            else:
                current_model_kwargs = model_kwargs
            
            # Handle parameter mapping for different model types AFTER merging configs
            # Map MONAI-style parameters to local model parameters
            if 'in_channels' in current_model_kwargs and 'n_channels' not in current_model_kwargs:
                current_model_kwargs['n_channels'] = current_model_kwargs.pop('in_channels')
                logger.info(f"[ARCH] Mapped in_channels -> n_channels: {current_model_kwargs['n_channels']}")
            elif 'in_channels' in current_model_kwargs and 'n_channels' in current_model_kwargs:
                # If both exist, use in_channels value for n_channels and remove in_channels
                current_model_kwargs['n_channels'] = current_model_kwargs.pop('in_channels')
                logger.info(f"[ARCH] Overrode n_channels with in_channels: {current_model_kwargs['n_channels']}")
            
            if 'out_channels' in current_model_kwargs and 'n_classes' not in current_model_kwargs:
                current_model_kwargs['n_classes'] = current_model_kwargs.pop('out_channels')
                logger.info(f"[ARCH] Mapped out_channels -> n_classes: {current_model_kwargs['n_classes']}")
            elif 'out_channels' in current_model_kwargs and 'n_classes' in current_model_kwargs:
                # If both exist, use out_channels value for n_classes and remove out_channels
                current_model_kwargs['n_classes'] = current_model_kwargs.pop('out_channels')
                logger.info(f"[ARCH] Overrode n_classes with out_channels: {current_model_kwargs['n_classes']}")
            
            # Create model instance
            logger.info(f"[ARCH] Instantiating {arch_info.display_name} with effective kwargs: {current_model_kwargs}")
            model = model_class(**current_model_kwargs)
            logger.info(f"[ARCH] Successfully loaded model: {arch_info.display_name}")
        
        # Move to device
        model = model.to(device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created successfully:")
        logger.info(f"  Architecture: {arch_info.display_name}")
        logger.info(f"  Framework: {arch_info.framework}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Device: {device}")
        
        # Log final model parameters after creation
        logger.info(f"[ACTUAL_MODEL] Final model parameters after creation:")
        logger.info(f"[ACTUAL_MODEL]   Total parameters: {total_params:,}")
        logger.info(f"[ACTUAL_MODEL]   Trainable parameters: {trainable_params:,}")
        logger.info(f"[ACTUAL_MODEL]   Model type: {model_type}")
        logger.info(f"[ACTUAL_MODEL]   Model class: {type(model).__name__}")
        logger.info(f"[ACTUAL_MODEL]   Device: {device}")
        
        # Check for parameter mismatch
        if total_params > 1_000_000:
            if 'micro' in str(model_kwargs) or 'tiny' in str(model_kwargs):
                logger.warning(f"[PARAMETER_MISMATCH] Expected small model but got {total_params:,} parameters!")
                logger.warning(f"[PARAMETER_MISMATCH] Model kwargs: {model_kwargs}")
        
        return model, arch_info
        
    except Exception as e:
        logger.error(f"Failed to create model '{model_type}': {e}")
        raise ValueError(f"Model creation failed for '{model_type}': {e}")

def save_enhanced_model_metadata(model_dir, model_id, unique_id, args, model_info, training_metrics, model_family="UNET", arch_info=None):
    """Save comprehensive model metadata"""
    # Ensure model directories exist when we actually need them
    ensure_model_directories(model_dir)
    
    # Use architecture info from registry if provided
    if arch_info is not None:
        architecture_name = arch_info.display_name
        framework = arch_info.framework
        architecture_key = arch_info.key
        architecture_version = arch_info.version
        architecture_description = arch_info.description
    else:
        architecture_name = "MonaiUNet"
        framework = "PyTorch"
        architecture_key = "monai_unet"
        architecture_version = "1.0.0"
        architecture_description = "MONAI U-Net"

    metadata = {
        "model_info": {
            "model_id": model_id,
            "unique_identifier": unique_id,
            "model_family": model_family,
            "version": architecture_version,
            "architecture": architecture_name,
            "architecture_key": architecture_key,
            "created_at": datetime.now().isoformat(),
            "framework": framework,
            "architecture_description": architecture_description
        },
        "architecture_details": model_info,
        "training_config": {
            "parameters": vars(args),
            "hyperparameters": {
                "optimizer": "Adam",
                "loss_function": "DiceLoss",
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "validation_split": args.validation_split
            }
        },
        "performance_metrics": training_metrics,
        "data_info": {
            "input_shape": "[1, 128, 128]",
            "output_shape": "[1, 128, 128]",
            "data_format": "DICOM/PNG",
            "normalization": "Scale Intensity [0, 1]"
        },
        "artifacts": {
            "weights_file": "weights/model.pth",
            "config_file": "config/model_config.json", 
            "training_curves": "artifacts/training_curves.png",
            "sample_predictions": "predictions/",
            "training_log": "logs/training.log"
        }
    }
    metadata_path = os.path.join(model_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4, default=str)
    return metadata_path

def save_model_comparison_artifacts(model_dir, model_metrics, epoch):
    """Save artifacts for model comparison and analysis"""
    try:
        # Ensure model directories exist when we actually need them
        ensure_model_directories(model_dir)
        
        artifacts_dir = os.path.join(model_dir, "artifacts")
        
        # Save detailed metrics as JSON
        metrics_file = os.path.join(artifacts_dir, f"detailed_metrics_epoch_{epoch+1}.json")
        with open(metrics_file, 'w') as f:
            json.dump(model_metrics, f, indent=4, default=str)
        
        # Create performance radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Define metrics for radar chart
        metrics_names = ['Dice Score', 'Loss (inverted)', 'Stability', 'Convergence']
        
        # Calculate normalized metrics (0-1 scale)
        dice_score = model_metrics.get('val_dice', 0)
        loss_inverted = 1 - min(model_metrics.get('val_loss', 1), 1) # Invert and cap loss
        
        # Calculate stability (lower variance in recent epochs is better)
        epoch_history = model_metrics.get('epoch_history', [])
        if len(epoch_history) > 5:
            recent_dices = [h.get('val_dice', 0) for h in epoch_history[-5:]]
            stability = 1 - (np.std(recent_dices) / np.mean(recent_dices)) if np.mean(recent_dices) > 0 else 0
        else:
            stability = 0.5
        
        # Calculate convergence (improvement trend)
        if len(epoch_history) > 3:
            early_dice = np.mean([h.get('val_dice', 0) for h in epoch_history[:3]])
            recent_dice = np.mean([h.get('val_dice', 0) for h in epoch_history[-3:]])
            convergence = min(recent_dice / early_dice if early_dice > 0 else 1, 1)
        else:
            convergence = 0.5
        
        values = [dice_score, loss_inverted, stability, convergence]
        
        # Complete the circle
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1)
        ax.set_title(f'Model Performance Radar - Epoch {epoch+1}', size=14, weight='bold')
        
        plt.tight_layout()
        radar_file = os.path.join(artifacts_dir, f"performance_radar_epoch_{epoch+1}.png")
        plt.savefig(radar_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return [metrics_file, radar_file]
    except Exception as e:
        print(f"Could not save model comparison artifacts: {e}")
        return []

def ensure_single_channel(x):
    """Ensure tensor has single channel - convert RGB to grayscale if needed"""
    if len(x.shape) == 3 and x.shape[0] == 3:  # RGB image
        # For masks/labels, check if all channels are the same (common for binary masks)
        if torch.allclose(x[0], x[1]) and torch.allclose(x[1], x[2]):
            # All channels are the same, just take the first one
            return x[0].unsqueeze(0)
        else:
            # Convert RGB to grayscale using standard weights
            return (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).unsqueeze(0)
    elif len(x.shape) == 3 and x.shape[0] == 1:  # Already single channel
        return x
    elif len(x.shape) == 2:  # No channel dimension
        return x.unsqueeze(0)
    else:
        return x

# --- GLOBAL WRAPPER FOR PICKLING ---
def medical_preprocessing_wrapper(image_array, preprocessing_type, preprocessing_params):
    return apply_medical_preprocessing(image_array, preprocessing_type, **preprocessing_params)

def apply_medical_preprocessing(image_array, preprocessing_type='angiography', **preprocessing_params):
    """Apply medical preprocessing to image array with configurable parameters"""
    if not MEDICAL_PREPROCESSING_AVAILABLE:
        logger.warning(f"[MEDICAL_PREPROCESSING] Medical preprocessing not available, returning original image")
        return image_array
    
    try:
        # Convert torch tensor to numpy if needed
        if isinstance(image_array, torch.Tensor):
            img_np = image_array.squeeze().cpu().numpy()
            device = image_array.device
        else:
            img_np = image_array
            device = None
            
        # Extract preprocessing parameters with defaults
        clahe_clip_limit = preprocessing_params.get('clahe_clip_limit', 3.0)
        clahe_tile_size = preprocessing_params.get('clahe_tile_size', 8)
        use_unsharp = preprocessing_params.get('use_unsharp_masking', False)
        unsharp_radius = preprocessing_params.get('unsharp_radius', 1.0)
        unsharp_amount = preprocessing_params.get('unsharp_amount', 1.0)
        use_frangi = preprocessing_params.get('use_frangi', False)
        frangi_scale_range = preprocessing_params.get('frangi_scale_range', '1,10')
        frangi_scale_step = preprocessing_params.get('frangi_scale_step', 2.0)
        use_histogram_eq = preprocessing_params.get('use_histogram_equalization', False)
        use_denoising = preprocessing_params.get('use_denoising', False)
        noise_variance = preprocessing_params.get('noise_variance', 0.1)
        intensity_range = preprocessing_params.get('intensity_range', 'auto')
        gamma_correction = preprocessing_params.get('gamma_correction', 1.0)
        vessel_enhancement_sigma = preprocessing_params.get('vessel_enhancement_sigma', 1.0)
        custom_pipeline = preprocessing_params.get('custom_pipeline', '')
        
        logger.info(f"[MEDICAL_PREPROCESSING] Applying {preprocessing_type} preprocessing with custom parameters")
        
        # Use custom pipeline if specified
        if custom_pipeline:
            logger.info(f"[MEDICAL_PREPROCESSING] Using custom pipeline: {custom_pipeline}")
            processed_img = _apply_custom_preprocessing_pipeline(
                img_np, custom_pipeline, preprocessing_params
            )
        # Apply medical preprocessing based on type with custom parameters
        elif preprocessing_type == 'angiography':
            target_size = preprocessing_params.get('target_size', (512, 512))
            processed_img = preprocess_angiography(
                img_np, 
                enhance_vessels=use_frangi,
                target_size=target_size
            )
        elif preprocessing_type == 'ct_coronary':
            target_size = preprocessing_params.get('target_size', (512, 512))
            processed_img = preprocess_ct_coronary(
                img_np,
                target_size=target_size
            )
        elif preprocessing_type == 'oct_coronary':
            target_size = preprocessing_params.get('target_size', (512, 512))
            processed_img = preprocess_oct_coronary(
                img_np,
                speckle_reduction=use_denoising,
                target_size=target_size
            )
        else:
            # Use general preprocessing with custom parameters
            logger.info(f"[MEDICAL_PREPROCESSING] Using general preprocessing with custom parameters")
            processed_img = _apply_general_preprocessing(img_np, preprocessing_params)
        
        # Convert back to torch tensor
        if isinstance(image_array, torch.Tensor):
            return torch.from_numpy(processed_img).unsqueeze(0).to(device)
        else:
            return processed_img
            
    except Exception as e:
        logger.warning(f"[MEDICAL_PREPROCESSING] Failed to apply preprocessing: {e}")
        return image_array

def _apply_custom_preprocessing_pipeline(image, pipeline_str, params):
    """Apply custom preprocessing pipeline based on string specification"""
    steps = [step.strip() for step in pipeline_str.split(',') if step.strip()]
    processed = image.copy()
    
    logger.info(f"[CUSTOM_PIPELINE] Applying steps: {steps}")
    
    # Create a preprocessor instance for method access
    preprocessor = MedicalImagePreprocessor()
    
    for step in steps:
        if step == 'clahe':
            processed = preprocessor.enhance_contrast_clahe(
                processed, 
                clip_limit=params.get('clahe_clip_limit', 3.0),
                tile_grid_size=(params.get('clahe_tile_size', 8), params.get('clahe_tile_size', 8))
            )
        elif step == 'unsharp':
            processed = preprocessor.enhance_contrast_unsharp_mask(
                processed,
                radius=params.get('unsharp_radius', 1.0),
                amount=params.get('unsharp_amount', 1.0)
            )
        elif step == 'frangi':
            scale_range = params.get('frangi_scale_range', '1,10')
            scale_min, scale_max = map(float, scale_range.split(','))
            processed = preprocessor.enhance_vessels_frangi(
                processed,
                scale_range=(scale_min, scale_max),
                scale_step=params.get('frangi_scale_step', 2.0)
            )
        elif step == 'denoise':
            processed = preprocessor.denoise(
                processed,
                method='bilateral'  # Use bilateral filtering
            )
        elif step == 'histeq':
            # Use OpenCV histogram equalization
            import cv2
            if len(processed.shape) == 2:
                processed = cv2.equalizeHist((processed * 255).astype(np.uint8)).astype(np.float64) / 255.0
            else:
                logger.warning(f"[CUSTOM_PIPELINE] Histogram equalization skipped - unsupported shape: {processed.shape}")
        elif step == 'gamma':
            gamma = params.get('gamma_correction', 1.0)
            if gamma != 1.0:
                processed = np.power(processed, gamma)
        elif step == 'normalize':
            processed = preprocessor.normalize(processed)
        else:
            logger.warning(f"[CUSTOM_PIPELINE] Unknown preprocessing step: {step}")
    
    return processed

def _apply_general_preprocessing(image, params):
    """Apply general preprocessing with all available options"""
    preprocessor = MedicalImagePreprocessor(
        target_size=None,  # Keep original size, will be resized later by MONAI
        normalize_method='percentile',
        enhance_contrast=params.get('clahe_clip_limit', 3.0) > 0,
        enhance_vessels=params.get('use_frangi', False),
        clahe_clip_limit=params.get('clahe_clip_limit', 3.0),
        clahe_tile_size=params.get('clahe_tile_size', 8),
        use_unsharp_masking=params.get('use_unsharp_masking', False),
        unsharp_radius=params.get('unsharp_radius', 1.0),
        unsharp_amount=params.get('unsharp_amount', 1.0),
        use_denoising=params.get('use_denoising', False),
        noise_variance=params.get('noise_variance', 0.1),
        gamma_correction=params.get('gamma_correction', 1.0)
    )
    result = preprocessor.preprocess(image)
    return result['image']

def get_monai_transforms(params, for_training=True, dataset_type=None):
    """Get MONAI transforms with configurable augmentations and optional medical preprocessing"""
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelTransform(keys=["image", "label"]),  # Ensure channel dimension
        # Convert images to single channel (grayscale) if they are RGB
        Lambdad(keys=["image"], func=ensure_single_channel),
        # Convert labels/masks to single channel (grayscale) if they are RGB
        Lambdad(keys=["label"], func=ensure_single_channel),
    ]
    
    # Add medical preprocessing if enabled and available
    use_medical_preprocessing = params.get('use_medical_preprocessing', False)
    if use_medical_preprocessing and MEDICAL_PREPROCESSING_AVAILABLE:
        preprocessing_type = params.get('medical_preprocessing_type', 'angiography')
        logger.info(f"[TRANSFORMS] Adding medical preprocessing ({preprocessing_type}) to pipeline")
        
        # Extract all preprocessing parameters
        preprocessing_params = {
            'clahe_clip_limit': params.get('preprocessing_clahe_clip_limit', 3.0),
            'clahe_tile_size': params.get('preprocessing_clahe_tile_size', 8),
            'use_unsharp_masking': params.get('preprocessing_use_unsharp_masking', False),
            'unsharp_radius': params.get('preprocessing_unsharp_radius', 1.0),
            'unsharp_amount': params.get('preprocessing_unsharp_amount', 1.0),
            'use_frangi': params.get('preprocessing_use_frangi', False),
            'frangi_scale_range': params.get('preprocessing_frangi_scale_range', '1,10'),
            'frangi_scale_step': params.get('preprocessing_frangi_scale_step', 2.0),
            'use_histogram_equalization': params.get('preprocessing_use_histogram_equalization', False),
            'use_denoising': params.get('preprocessing_use_denoising', False),
            'noise_variance': params.get('preprocessing_noise_variance', 0.1),
            'intensity_range': params.get('preprocessing_intensity_range', 'auto'),
            'gamma_correction': params.get('preprocessing_gamma_correction', 1.0),
            'vessel_enhancement_sigma': params.get('preprocessing_vessel_enhancement_sigma', 1.0),
            'custom_pipeline': params.get('preprocessing_custom_pipeline', '')
        }
        
        # Use global wrapper function with partial to make it pickle-able
        from functools import partial
        transforms.append(Lambdad(keys=["image"], func=partial(
            medical_preprocessing_wrapper, 
            preprocessing_type=preprocessing_type, 
            preprocessing_params=preprocessing_params
        )))
    
    # Standard intensity scaling
    transforms.extend([
        ScaleIntensityd(keys=["image"]),
        # Normalize labels to 0-1 range for binary masks
        ScaleIntensityd(keys=["label"], minv=0.0, maxv=1.0),
    ])
    
    if for_training:
        # Add training-specific augmentations
        if params.get('use_random_scale', True):
            # Use RandScaleIntensityd for random intensity scaling
            transforms.append(RandScaleIntensityd(keys=["image"], factors=(0.8, 1.2), prob=0.5))
        
        crop_size = params.get('crop_size', 128)
        
        # Use RandCropByPosNegLabeld only when explicitly enabled via checkbox
        use_pos_neg_crop = params.get('use_pos_neg_cropping', False)
        
        if use_pos_neg_crop:
            # Use RandCropByPosNegLabeld for advanced segmentation cropping
            # Set num_samples=1 to maintain batch size consistency
            transforms.append(RandCropByPosNegLabeld(
                keys=["image", "label"], 
                label_key="label",
                spatial_size=[crop_size, crop_size],
                pos=1,
                neg=1,
                num_samples=1,  # Changed from 4 to 1 to maintain batch size
                image_key="image",
                image_threshold=0
            ))
        else:
            # For regular cropping, use standard random crop
            transforms.append(RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[crop_size, crop_size],
                random_center=True,
                random_size=False
            ))
        
        if params.get('use_random_flip', True):
            transforms.append(RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0))
        
        if params.get('use_random_rotate', True):
            transforms.append(RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3))
        
        if params.get('use_random_intensity', True):
            # Use RandScaleIntensityd for random intensity variations
            transforms.append(RandScaleIntensityd(
                keys=["image"],
                factors=(0.9, 1.1),
                prob=0.7
            ))
    else:
        # For validation/visualization - resize to standard size but keep full images
        transforms.append(Resized(keys=["image", "label"], spatial_size=[256, 256], mode=["bilinear", "nearest"]))
    
    transforms.append(ToTensord(keys=["image", "label"]))
    return Compose(transforms)




def save_model_summary(model, model_dir=None):
    if model_dir:
        # Ensure model directories exist when we actually need them
        ensure_model_directories(model_dir)
        
    summary = []
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        summary.append(f"{name}: {list(param.shape)}, params: {param_count}")
    summary_text = "\n".join([
        "Model Summary:",
        "=" * 50,
        "\n".join(summary),
        "=" * 50,
        f"Total parameters: {total_params:,}",
        f"Trainable parameters: {trainable_params:,}",
    ])
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        out_path = os.path.join(model_dir, "model_summary.txt")
    else:
        out_path = "model_summary.txt"
    with open(out_path, "w") as f:
        f.write(summary_text)
    return out_path

def save_training_curves(epoch, metrics, logger, model_dir=None):
    """Save training curves plot"""
    try:
        if model_dir:
            # Ensure model directories exist when we actually need them
            ensure_model_directories(model_dir)
            
        # This is a simple implementation - in practice you'd want to track metrics over time
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # For now, just show current epoch metrics
        ax1.bar(['Train', 'Val'], [metrics['train_loss'], metrics['val_loss']])
        ax1.set_title('Loss')
        ax1.set_ylabel('Loss')
        
        ax2.bar(['Train', 'Val'], [metrics['train_dice'], metrics['val_dice']])
        ax2.set_title('Dice Score')
        ax2.set_ylabel('Dice')
        
        ax3.text(0.5, 0.5, f"Epoch: {epoch + 1}\nTrain Loss: {metrics['train_loss']:.4f}\nVal Loss: {metrics['val_loss']:.4f}", 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Current Metrics')
        ax3.set_xlabel('Epoch')
        ax3.axis('off')
        
        ax4.text(0.5, 0.5, f"Train Dice: {metrics['train_dice']:.4f}\nVal Dice: {metrics['val_dice']:.4f}", 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Performance')
        ax4.axis('off')
        
        plt.tight_layout()
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            filename = os.path.join(model_dir, f'training_curves_epoch_{epoch+1}.png')
        else:
            filename = f'training_curves_epoch_{epoch+1}.png'
        plt.savefig(filename)
        plt.close()
        return filename
    except Exception as e:
        logger.warning(f"Could not save training curves: {e}")
        return None

def save_sample_predictions(model, val_loader, device, epoch, model_dir=None, class_info=None, threshold=0.5):
    """Save sample predictions from validation set with enhanced error handling
    
    Enhanced error handling includes:
    - Safe batch format detection and validation
    - Multiple fallback paths for different batch types
    - Comprehensive error logging with traceback
    - Fallback error image creation when prediction generation fails
    - File size validation to ensure valid PNG output
    - Classification visualization support
    
    Args:
        threshold: Binary segmentation threshold for hard predictions
    """
    logger.info(f"[PREDICTIONS] Starting to save sample predictions for epoch {epoch+1}")
    
    # Check if this is a classification task
    is_classification_task = class_info and class_info.get('class_type') == 'classification'
    
    if model_dir:
        # Ensure model directories exist
        ensure_model_directories(model_dir)
        
    model.eval()
    
    # Ensure we have a clean matplotlib state
    plt.close('all')
    
    with torch.no_grad():
        try:
            # Safe batch retrieval
            val_batch = next(iter(val_loader))
            images, labels = None, None
            
            if isinstance(val_batch, dict):
                if "image" in val_batch and "label" in val_batch:
                    images = val_batch["image"].to(device)
                    labels = val_batch["label"].to(device)
                else:
                    logger.error(f"[PREDICTIONS] Dict batch missing required keys. Available keys: {list(val_batch.keys())}")
                    raise ValueError("Batch dict missing 'image' or 'label' keys")
            elif isinstance(val_batch, (list, tuple)) and len(val_batch) == 2:
                images, labels = val_batch[0].to(device), val_batch[1].to(device)
            else:
                logger.error(f"[PREDICTIONS] Unexpected batch format: {type(val_batch)}")
                raise ValueError(f"Unsupported batch format: {type(val_batch)}")
            
            if images is None or labels is None:
                raise ValueError("Failed to extract images and labels from batch")
                
            logger.info(f"[PREDICTIONS] Processing batch with {images.shape[0]} samples, image shape: {images.shape}")
            
            # Model inference
            outputs = model(images)
            
            # Handle classification vs segmentation tasks differently
            if is_classification_task:
                # Classification task - create classification visualization
                logger.info(f"[PREDICTIONS] Creating classification visualization for epoch {epoch+1}")
                
                # Get predicted classes and probabilities
                probs = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(outputs, dim=1)
                
                # Create figure for classification visualization
                fig, axes = plt.subplots(2, 4, figsize=(15, 8))
                plt.suptitle(f'Artery Classification Predictions - Epoch {epoch+1}', fontsize=16)
                
                num_samples = min(4, images.shape[0])
                class_names = ['Right Artery', 'Left Artery']  # Standard for artery classification
                colors = ['lightcoral', 'lightblue']  # Red for right, blue for left
                
                for i in range(num_samples):
                    # Input mask (for artery classification, input is typically binary mask)
                    axes[0, i].imshow(images[i, 0].cpu().numpy(), cmap='gray')
                    axes[0, i].set_title(f'Input Mask #{i+1}')
                    axes[0, i].axis('off')
                    
                    # Classification result with confidence
                    pred_class = predicted_classes[i].item()
                    true_class = labels[i].item()
                    confidence = probs[i, pred_class].item()
                    
                    # Create classification result visualization
                    axes[1, i].clear()
                    axes[1, i].set_xlim(0, 1)
                    axes[1, i].set_ylim(0, 1)
                    
                    # Color background based on prediction
                    axes[1, i].set_facecolor(colors[pred_class])
                    
                    # Add text with prediction and confidence
                    pred_text = f"Pred: {class_names[pred_class]}\nConf: {confidence:.3f}"
                    true_text = f"True: {class_names[true_class]}"
                    
                    # Show prediction
                    axes[1, i].text(0.5, 0.7, pred_text, ha='center', va='center', 
                                   fontsize=12, fontweight='bold', 
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                    
                    # Show ground truth
                    axes[1, i].text(0.5, 0.3, true_text, ha='center', va='center', 
                                   fontsize=10, 
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
                    
                    # Mark correctness
                    if pred_class == true_class:
                        marker_color = 'green'
                        marker_text = '✓'
                    else:
                        marker_color = 'red'
                        marker_text = '✗'
                    
                    axes[1, i].text(0.9, 0.9, marker_text, ha='center', va='center', 
                                   fontsize=20, color=marker_color, fontweight='bold')
                    
                    axes[1, i].set_title(f'Classification #{i+1}')
                    axes[1, i].axis('off')
                
                # Hide unused subplots
                for i in range(num_samples, 4):
                    axes[0, i].axis('off')
                    axes[1, i].axis('off')
                
                # Save classification visualization and return
                plt.tight_layout()
                
                # Determine output path for classification
                if model_dir and os.path.exists(model_dir):
                    pred_dir = os.path.join(model_dir, f'predictions/epoch_{epoch+1:03d}')
                    os.makedirs(pred_dir, exist_ok=True)
                    filename = os.path.join(pred_dir, f'predictions_epoch_{epoch+1:03d}.png')
                else:
                    fallback_dir = os.path.join(CORE_DATA_DIR, 'models', 'artifacts', 'predictions')
                    os.makedirs(fallback_dir, exist_ok=True)
                    filename = os.path.join(fallback_dir, f'predictions_epoch_{epoch+1:03d}.png')
                
                plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
                plt.close()
                
                # Verify file was created
                if os.path.exists(filename) and os.path.getsize(filename) > 1000:
                    logger.info(f"[PREDICTIONS] Successfully saved classification prediction samples to: {filename}")
                    return filename
                else:
                    logger.error(f"[PREDICTIONS] Failed to create valid classification prediction file: {filename}")
                    return None
                
            else:
                # Segmentation task - use existing logic
                # Apply appropriate post-processing based on number of output channels
                num_output_channels = outputs.shape[1]
                logger.info(f"[PREDICTIONS] Output shape: {outputs.shape}, channels: {num_output_channels}")
                logger.info(f"[PREDICTIONS] Raw output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                
                if num_output_channels == 1:
                    # Binary segmentation
                    outputs_raw = outputs.clone()  # Keep raw logits for debugging
                    outputs_soft = torch.sigmoid(outputs)  # Soft predictions (probabilities)
                    outputs_hard = (outputs_soft > threshold).float()  # Hard predictions (binary)
                    
                    logger.info(f"[PREDICTIONS] Raw logits range: [{outputs_raw.min().item():.4f}, {outputs_raw.max().item():.4f}]")
                    logger.info(f"[PREDICTIONS] Soft predictions range: [{outputs_soft.min().item():.4f}, {outputs_soft.max().item():.4f}]")
                    logger.info(f"[PREDICTIONS] Hard predictions range: [{outputs_hard.min().item():.4f}, {outputs_hard.max().item():.4f}]")
                    logger.info(f"[PREDICTIONS] Unique values after threshold: {torch.unique(outputs_hard).cpu().numpy()}")
                    logger.info(f"[PREDICTIONS] Applied binary segmentation post-processing (sigmoid + threshold)")
                    
                    use_colormap = False
                    # Use hard predictions for visualization, keep soft for analysis
                    outputs_for_viz = outputs_hard
                else:
                    # Multi-class semantic segmentation
                    outputs = torch.softmax(outputs, dim=1)
                    outputs_for_viz = torch.argmax(outputs, dim=1, keepdim=True).float()
                    logger.info(f"[PREDICTIONS] Applied multi-class segmentation post-processing (softmax + argmax) for {num_output_channels} classes")
                    use_colormap = True
                    outputs_soft = None  # No soft predictions for multi-class
            
            # Create figure with proper settings
            fig, axes = plt.subplots(4, 4, figsize=(15, 12))  # Add 4th row for soft predictions
            plt.suptitle(f'Sample Predictions - Epoch {epoch+1} (Full Images)', fontsize=16)
            
            # Create custom colormap for semantic segmentation
            if use_colormap:
                import matplotlib.colors as mcolors
                # Create a custom colormap with distinct colors for each class
                colors = [
                    '#000000',  # 0 - black (background)
                    '#FF0000',  # 1 - red
                    '#00FF00',  # 2 - green
                    '#0000FF',  # 3 - blue
                    '#FFFF00',  # 4 - yellow
                    '#FF00FF',  # 5 - magenta
                    '#00FFFF',  # 6 - cyan
                    '#FFA500',  # 7 - orange
                    '#800080',  # 8 - purple
                    '#FFC0CB',  # 9 - pink
                    '#ADFF2F',  # 10 - green yellow
                    '#1E90FF',  # 11 - dodger blue
                    '#FF1493',  # 12 - deep pink
                    '#00FA9A',  # 13 - medium spring green
                    '#FF4500',  # 14 - red orange
                    '#483D8B',  # 15 - dark slate blue
                    '#FFD700',  # 16 - gold
                    '#DC143C',  # 17 - crimson
                    '#7CFC00',  # 18 - lawn green
                    '#BA55D3',  # 19 - medium orchid
                    '#8A2BE2',  # 20 - blue violet
                    '#FF69B4',  # 21 - hot pink
                    '#FF8C00',  # 22 - dark orange
                    '#B8860B',  # 23 - dark goldenrod
                    '#4682B4',  # 24 - steel blue
                    '#00CED1',  # 25 - dark turquoise
                    '#FF6347'   # 26 - tomato red
                ]
                # Pad with additional colors if needed
                while len(colors) < num_output_channels:
                    colors.append('#FFFFFF')  # white for overflow
                
                cmap = mcolors.ListedColormap(colors[:num_output_channels])
            
            num_samples = min(4, images.shape[0])
            for i in range(num_samples):
                # Input image
                axes[0, i].imshow(images[i, 0].cpu().numpy(), cmap='gray')
                axes[0, i].set_title(f'Input #{i+1}')
                axes[0, i].axis('off')
                
                # Ground truth - use appropriate visualization
                if use_colormap:
                    # For multi-class segmentation, use custom color mapping
                    gt_data = labels[i, 0].cpu().numpy() if labels.shape[1] == 1 else labels[i].cpu().numpy()
                    # If gt_data is one-hot encoded, convert to class indices
                    if len(gt_data.shape) == 3:
                        gt_data = np.argmax(gt_data, axis=0)
                    elif len(gt_data.shape) == 2 and gt_data.shape[0] > 1:
                        gt_data = np.argmax(gt_data, axis=0)
                    
                    unique_classes = np.unique(gt_data)
                    logger.info(f"[PREDICTIONS] Ground truth classes for sample {i+1}: {unique_classes}")
                    axes[1, i].imshow(gt_data, cmap=cmap, vmin=0, vmax=num_output_channels-1)
                    axes[1, i].set_title(f'Ground Truth #{i+1} ({len(unique_classes)} classes)')
                else:
                    # For binary segmentation, use grayscale
                    axes[1, i].imshow(labels[i, 0].cpu().numpy(), cmap='gray')
                    axes[1, i].set_title(f'Ground Truth #{i+1}')
                axes[1, i].axis('off')
                
                # Hard Prediction - use appropriate visualization
                if use_colormap:
                    # For multi-class segmentation, use custom color mapping
                    pred_data = outputs_for_viz[i, 0].cpu().numpy()
                    unique_pred_classes = np.unique(pred_data)
                    logger.info(f"[PREDICTIONS] Predicted classes for sample {i+1}: {unique_pred_classes}")
                    axes[2, i].imshow(pred_data, cmap=cmap, vmin=0, vmax=num_output_channels-1)
                    axes[2, i].set_title(f'Hard Pred #{i+1} ({len(unique_pred_classes)} classes)')
                else:
                    # For binary segmentation, use grayscale
                    pred_data = outputs_for_viz[i, 0].cpu().numpy()
                    logger.info(f"[PREDICTIONS] Binary hard pred sample {i+1} range: [{pred_data.min():.4f}, {pred_data.max():.4f}]")
                    logger.info(f"[PREDICTIONS] Binary hard pred sample {i+1} unique values: {np.unique(pred_data)}")
                    axes[2, i].imshow(pred_data, cmap='gray', vmin=0, vmax=1)  # Force range 0-1
                    axes[2, i].set_title(f'Hard Pred #{i+1}')
                axes[2, i].axis('off')
                
                # Soft Prediction (only for binary segmentation)
                if not use_colormap and outputs_soft is not None:
                    soft_data = outputs_soft[i, 0].cpu().numpy()
                    logger.info(f"[PREDICTIONS] Binary soft pred sample {i+1} range: [{soft_data.min():.4f}, {soft_data.max():.4f}]")
                    axes[3, i].imshow(soft_data, cmap='gray', vmin=0, vmax=1)  # Force range 0-1
                    axes[3, i].set_title(f'Soft Pred #{i+1}')
                else:
                    axes[3, i].axis('off')  # Hide for multi-class
                axes[3, i].axis('off')
            
            # Hide unused subplots if we have fewer than 4 samples
            for i in range(num_samples, 4):
                axes[0, i].axis('off')
                axes[1, i].axis('off')
                axes[2, i].axis('off')
                axes[3, i].axis('off')
            
            plt.tight_layout()
            
            # Determine output path - ensure predictions directory exists
            if model_dir and os.path.exists(model_dir):
                pred_dir = os.path.join(model_dir, f'predictions/epoch_{epoch+1:03d}')
                os.makedirs(pred_dir, exist_ok=True)
                filename = os.path.join(pred_dir, f'predictions_epoch_{epoch+1:03d}.png')
            else:
                # Fallback directory
                fallback_dir = os.path.join(CORE_DATA_DIR, 'models', 'artifacts', 'predictions')
                os.makedirs(fallback_dir, exist_ok=True)
                filename = os.path.join(fallback_dir, f'predictions_epoch_{epoch+1:03d}.png')
            
            # Save with high quality and ensure file is written
            try:
                # Use PNG format explicitly and handle potential issues
                plt.savefig(filename, format='png', dpi=150, bbox_inches='tight', 
                           facecolor='white', edgecolor='none', pil_kwargs={'optimize': True})
                plt.close('all')  # Close all figures to free memory
                
                # Wait a moment for file system to sync
                import time
                time.sleep(0.1)
                
                # Verify file was created and has content
                if os.path.exists(filename) and os.path.getsize(filename) > 1000:  # At least 1KB
                    # Additional verification - try to open the image
                    try:
                        from PIL import Image
                        with Image.open(filename) as img:
                            # Verify image can be loaded and has reasonable dimensions
                            if img.size[0] > 0 and img.size[1] > 0:
                                logger.info(f"[PREDICTIONS] Successfully saved prediction samples to: {filename} ({img.size[0]}x{img.size[1]})")
                                return filename
                            else:
                                logger.error(f"[PREDICTIONS] Image has invalid dimensions: {img.size}")
                                return None
                    except Exception as img_verify_error:
                        logger.warning(f"[PREDICTIONS] Could not verify image but file exists: {img_verify_error}")
                        # File exists and has size, assume it's okay
                        return filename
                else:
                    logger.error(f"[PREDICTIONS] Failed to create valid prediction file: {filename}")
                    return None
                    
            except Exception as save_error:
                logger.error(f"[PREDICTIONS] Error saving plot: {save_error}")
                plt.close('all')
                return None
                
        except Exception as e:
            logger.error(f"[PREDICTIONS] Error creating sample predictions: {e}")
            import traceback
            logger.error(f"[PREDICTIONS] Traceback: {traceback.format_exc()}")
            plt.close('all')
            
            # Try to create a simple fallback image to ensure we have something
            try:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                error_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
                ax.text(0.5, 0.5, f'Epoch {epoch+1}\nPrediction generation failed\nError: {error_msg}', 
                       ha='center', va='center', fontsize=12, transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                ax.set_title(f'Prediction Error - Epoch {epoch+1}', fontsize=14)
                
                # Determine fallback output path
                if model_dir and os.path.exists(model_dir):
                    pred_dir = os.path.join(model_dir, f'predictions/epoch_{epoch+1:03d}')
                    os.makedirs(pred_dir, exist_ok=True)
                    fallback_filename = os.path.join(pred_dir, f'predictions_epoch_{epoch+1:03d}_error.png')
                else:
                    fallback_dir = os.path.join(CORE_DATA_DIR, 'models', 'artifacts', 'predictions')
                    os.makedirs(fallback_dir, exist_ok=True)
                    fallback_filename = os.path.join(fallback_dir, f'predictions_epoch_{epoch+1:03d}_error.png')
                
                plt.savefig(fallback_filename, dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
                plt.close()
                
                # Verify fallback file was created
                if os.path.exists(fallback_filename) and os.path.getsize(fallback_filename) > 0:
                    logger.info(f"[PREDICTIONS] Created error fallback image: {fallback_filename}")
                    return fallback_filename
                else:
                    logger.error(f"[PREDICTIONS] Failed to create fallback image")
                    return None
                    
            except Exception as fallback_error:
                logger.error(f"[PREDICTIONS] Failed to create even fallback image: {fallback_error}")
                return None

def save_config(args, model_dir=None):
    """Save training configuration"""
    if model_dir:
        # Ensure model directories exist when we actually need them
        ensure_model_directories(model_dir)
        
    # Extract augmentation flags explicitly
    config = {
        "training_params": vars(args),
        "device": str(getattr(args, 'device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))),
        "pytorch_version": torch.__version__,
        "random_flip": getattr(args, 'random_flip', False),
        "random_rotate": getattr(args, 'random_rotate', False),
        "random_scale": getattr(args, 'random_scale', False),
        "random_intensity": getattr(args, 'random_intensity', False),
    }
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        out_path = os.path.join(model_dir, "training_config.json")
    else:
        out_path = "training_config.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=4)
    return out_path

def get_monai_datasets(data_path, val_split=0.2, transform_params=None, dataset_type=None):
    import glob
    # Use default params if none provided
    if transform_params is None:
        transform_params = {
            'use_random_flip': True,
            'use_random_rotate': True,
            'use_random_scale': True,
            'use_random_intensity': True,
            'crop_size': 128
        }
    
    # Initialize directory variables
    images_dir = None
    labels_dir = None
    
    # Support both images/masks and images/labels, prefer imgs/masks if present
    imgs_dir = os.path.join(data_path, "imgs")
    masks_dir = os.path.join(data_path, "masks")
    if os.path.exists(imgs_dir) and os.path.exists(masks_dir):
        images_dir = imgs_dir
        labels_dir = masks_dir
        image_files = sorted(glob.glob(f"{imgs_dir}/*"))
        label_files = sorted(glob.glob(f"{masks_dir}/*"))
    else:
        # Fallback to images/labels or nested data/data/images, data/data/labels
        images_dir = os.path.join(data_path, "images")
        labels_dir = os.path.join(data_path, "labels")
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            images_dir = os.path.join(data_path, "data", "images")
            labels_dir = os.path.join(data_path, "data", "labels")
        image_files = sorted(glob.glob(f"{images_dir}/*"))
        label_files = sorted(glob.glob(f"{labels_dir}/*"))
    
    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(image_files, label_files)]
    n_val = int(len(data_dicts) * val_split)
    train_files, val_files = data_dicts[n_val:], data_dicts[:n_val]
    
    # Enhanced dataset logging
    logger.info(f"[DATASET] Dataset directory: {data_path}")
    logger.info(f"[DATASET] Images directory: {images_dir}")
    logger.info(f"[DATASET] Labels directory: {labels_dir}")
    logger.info(f"[DATASET] Total files found: {len(image_files)} images, {len(label_files)} labels")
    logger.info(f"[DATASET] Validation split: {val_split} ({n_val} samples for validation)")
    logger.info(f"[DATASET] Training samples: {len(train_files)}, Validation samples: {len(val_files)}")
    
    # Log sample file paths for verification
    if train_files:
        logger.info(f"[DATASET] Sample training files:")
        for i, sample in enumerate(train_files[:3]):  # Show first 3 training samples
            logger.info(f"[DATASET]   Train #{i+1}: Image: {os.path.basename(sample['image'])}, Label: {os.path.basename(sample['label'])}")
    
    if val_files:
        logger.info(f"[DATASET] Sample validation files:")
        for i, sample in enumerate(val_files[:3]):  # Show first 3 validation samples
            logger.info(f"[DATASET]   Val #{i+1}: Image: {os.path.basename(sample['image'])}, Label: {os.path.basename(sample['label'])}")
    
    # Log file extension info
    if image_files:
        image_exts = set(os.path.splitext(f)[1].lower() for f in image_files)
        label_exts = set(os.path.splitext(f)[1].lower() for f in label_files)
        logger.info(f"[DATASET] Image file extensions: {sorted(image_exts)}")
        logger.info(f"[DATASET] Label file extensions: {sorted(label_exts)}")
    
    # Create separate transforms for training and validation
    train_transforms = get_monai_transforms(transform_params, for_training=True, dataset_type=dataset_type)
    val_transforms = get_monai_transforms(transform_params, for_training=False, dataset_type=dataset_type)
    
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
    return train_ds, val_ds

def detect_dataset_type(data_path):
    """
    Automatically detect dataset type (ARCADE or standard coronary)
    Returns: tuple (dataset_type, dataset_info)
    """
    logger.info(f"[DATASET] Detecting dataset type for: {data_path}")
    
    # Check for ARCADE dataset structure
    if ARCADE_AVAILABLE:
        # Look for ARCADE dataset structure
        arcade_path = os.path.join(data_path, "arcade_challenge_datasets")
        if os.path.exists(arcade_path):
            logger.info("[DATASET] ARCADE dataset detected")
            info = get_arcade_dataset_info(data_path)
            return "arcade", info
    
    # Check for standard coronary dataset structure
    imgs_path = os.path.join(data_path, "imgs")
    masks_path = os.path.join(data_path, "masks")
    
    if os.path.exists(imgs_path) and os.path.exists(masks_path):
        img_count = len([f for f in os.listdir(imgs_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        mask_count = len([f for f in os.listdir(masks_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        logger.info(f"[DATASET] Standard coronary dataset detected: {img_count} images, {mask_count} masks")
        
        info = {
            "dataset_type": "coronary_standard",
            "images": img_count,
            "masks": mask_count,
            "path": data_path
        }
        return "coronary_standard", info
    
    # Check for MONAI-style structure
    images_dir = os.path.join(data_path, "images")
    labels_dir = os.path.join(data_path, "labels")
    
    if os.path.exists(images_dir) and os.path.exists(labels_dir):
        img_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.nii', '.nii.gz'))])
        label_count = len([f for f in os.listdir(labels_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.nii', '.nii.gz'))])
        
        logger.info(f"[DATASET] MONAI-style dataset detected: {img_count} images, {label_count} labels")
        
        info = {
            "dataset_type": "monai_style",
            "images": img_count,
            "labels": label_count,
            "path": data_path
        }
        return "monai_style", info
    
    logger.warning(f"[DATASET] Unknown dataset structure in: {data_path}")
    return "unknown", {"path": data_path}

def get_datasets_with_auto_detection(data_path, validation_split, transform_params, args):
    """
    Get datasets with automatic detection of dataset type
    Supports both ARCADE and standard coronary datasets
    """
    # Check if dataset type is explicitly specified
    dataset_type_override = getattr(args, 'dataset_type', 'auto')
    
    if dataset_type_override != 'auto':
        logger.info(f"[DATASET] Using specified dataset type: {dataset_type_override}")
        
        if dataset_type_override.startswith('arcade_') and ARCADE_AVAILABLE:
            logger.info("[DATASET] Using ARCADE dataset loader (forced)")
            return get_arcade_datasets(data_path, validation_split, transform_params, args, forced_type=dataset_type_override)
        elif dataset_type_override == 'coronary':
            logger.info("[DATASET] Using MONAI dataset loader (forced)")
            return get_monai_datasets(data_path, validation_split, transform_params, dataset_type_override)
        else:
            logger.warning(f"[DATASET] Unknown dataset type '{dataset_type_override}', falling back to auto-detection")
    
    # Auto-detection
    dataset_type, dataset_info = detect_dataset_type(data_path)
    
    if dataset_type == "arcade" and ARCADE_AVAILABLE:
        logger.info("[DATASET] Using ARCADE dataset loader")
        return get_arcade_datasets(data_path, validation_split, transform_params, args)
    elif dataset_type in ["coronary_standard", "monai_style"]:
        logger.info("[DATASET] Using MONAI dataset loader")
        return get_monai_datasets(data_path, validation_split, transform_params, dataset_type)
    else:
        logger.error(f"[DATASET] Unsupported dataset type: {dataset_type}")
        raise ValueError(f"Unsupported dataset structure in {data_path}")

def get_arcade_datasets(data_path, validation_split, transform_params, args, forced_type=None):
    """
    Create ARCADE datasets for training using torch_arcade_loader
    """
    # Ensure ARCADE support
    if not ARCADE_AVAILABLE:
        raise ImportError("ARCADE dataset support not available. Install pycocotools: pip install pycocotools")
    
    # Log detailed information about dataset path and args
    logger.info(f"[ARCADE] Creating datasets with torch_arcade_loader:")
    logger.info(f"[ARCADE]   data_path: {data_path}")
    logger.info(f"[ARCADE]   forced_type: {forced_type}")
    logger.info(f"[ARCADE]   args.dataset_type: {getattr(args, 'dataset_type', 'NOT_SET')}")
    logger.info(f"[ARCADE]   Directory exists: {os.path.exists(data_path)}")
    if os.path.exists(data_path):
        logger.info(f"[ARCADE]   Directory contents: {os.listdir(data_path)}")
        arcade_path = os.path.join(data_path, "arcade_challenge_datasets")
        logger.info(f"[ARCADE]   ARCADE path exists: {os.path.exists(arcade_path)}")
        if os.path.exists(arcade_path):
            logger.info(f"[ARCADE]   ARCADE contents: {os.listdir(arcade_path)}")
    
    # Determine task type based on forced_type or model_type
    if forced_type:
        # Map GUI dataset types to torch_arcade_loader task types
        mapping = {
            'arcade_binary': 'binary_segmentation',
            'arcade_binary_segmentation': 'binary_segmentation',
            'arcade_semantic': 'semantic_segmentation',
            'arcade_semantic_segmentation': 'semantic_segmentation',
            'arcade_stenosis': 'stenosis_detection',
            'arcade_stenosis_detection': 'stenosis_detection',
            'arcade_classification': 'artery_classification',
            'arcade_artery_classification': 'artery_classification',
            'arcade_semantic_seg_binary': 'semantic_segmentation_binary',
            'arcade_stenosis_segmentation': 'stenosis_segmentation'
        }
        task = mapping.get(forced_type, 'binary_segmentation')
    else:
        # Infer task from model type
        mt = getattr(args, 'model_type', '').lower()
        if 'semantic' in mt:
            task = 'semantic_segmentation'
        elif 'stenosis' in mt:
            task = 'stenosis_detection'
        elif 'artery' in mt or 'classification' in mt:
            task = 'artery_classification'
        else:
            task = 'binary_segmentation'
    
    logger.info(f"[ARCADE] Using torch_arcade_loader task: {task}")
    
    # Get image size from transform params
    crop_size = transform_params.get('crop_size', 128)
    image_size = crop_size
    logger.info(f"[ARCADE] Using image size: {image_size}x{image_size}")
    
    # Use conservative num_workers for Docker environment
    num_workers = min(getattr(args, 'num_workers', 1), 1)
    batch_size = args.batch_size
    
    try:
        # Create train dataloader using torch_arcade_loader
        logger.info(f"[ARCADE] Creating train dataloader for task: {task}")
        train_loader = create_arcade_dataloader(
            root=data_path,
            task=task,
            image_set='train',
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            download=False,
            image_size=image_size,
            side=getattr(args, 'artery_side', None)
        )
        
        # Create validation dataloader using torch_arcade_loader
        logger.info(f"[ARCADE] Creating val dataloader for task: {task}")
        val_loader = create_arcade_dataloader(
            root=data_path,
            task=task,
            image_set='val',
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            download=False,
            image_size=image_size,
            side=getattr(args, 'artery_side', None)
        )
        
        # Log dataset information
        train_dataset_size = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else len(train_loader) * batch_size
        val_dataset_size = len(val_loader.dataset) if hasattr(val_loader, 'dataset') else len(val_loader) * batch_size
        
        logger.info(f"[ARCADE] Dataset created successfully using torch_arcade_loader:")
        logger.info(f"[ARCADE]   Task: {task}")
        logger.info(f"[ARCADE]   Train samples: {train_dataset_size}")
        logger.info(f"[ARCADE]   Val samples: {val_dataset_size}")
        logger.info(f"[ARCADE]   Total samples: {train_dataset_size + val_dataset_size}")
        logger.info(f"[ARCADE]   Image size: {image_size}x{image_size}")
        logger.info(f"[ARCADE]   Batch size: {batch_size}")
        logger.info(f"[ARCADE]   Num workers: {num_workers}")
        
        # Task-specific logging
        if task == 'semantic_segmentation':
            logger.info(f"[ARCADE]   Task type: Semantic Segmentation (multi-class)")
            logger.info(f"[ARCADE]   Expected classes: Multiple coronary artery segments")
        elif task == 'binary_segmentation':
            logger.info(f"[ARCADE]   Task type: Binary Segmentation")
            logger.info(f"[ARCADE]   Expected classes: Background + Artery")
        elif task == 'artery_classification':
            logger.info(f"[ARCADE]   Task type: Artery Classification")
            logger.info(f"[ARCADE]   Input: Binary mask, Output: 0=right, 1=left")
        elif task == 'stenosis_detection':
            logger.info(f"[ARCADE]   Task type: Stenosis Detection (bounding box)")
            logger.info(f"[ARCADE]   Expected output: COCO format bounding boxes")
        elif task == 'stenosis_segmentation':
            logger.info(f"[ARCADE]   Task type: Stenosis Segmentation")
            logger.info(f"[ARCADE]   Expected classes: Background + Stenosis")
        elif task == 'semantic_segmentation_binary':
            logger.info(f"[ARCADE]   Task type: Semantic Segmentation from Binary")
            logger.info(f"[ARCADE]   Input: Binary mask, Output: Multi-class mask")
        
        logger.info(f"[ARCADE] torch_arcade_loader setup completed successfully")
        
    except Exception as e:
        logger.error(f"[ARCADE] Failed to create dataloaders with torch_arcade_loader: {e}")
        logger.error(f"[ARCADE] Exception type: {type(e)}")
        import traceback
        logger.error(f"[ARCADE] Traceback: {traceback.format_exc()}")
        raise
    
    return train_loader, val_loader

def create_optimizer(model, args):
    """Create optimizer based on args.optimizer choice"""
    optimizer_name = getattr(args, 'optimizer', 'adam').lower()
    learning_rate = args.learning_rate
    
    logger.info(f"[OPTIMIZER] Creating {optimizer_name.upper()} optimizer with lr={learning_rate}")
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        # SGD with momentum for better convergence
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-8)
    elif optimizer_name == 'adamw':
        # AdamW with weight decay for better regularization
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    else:
        logger.warning(f"[OPTIMIZER] Unknown optimizer '{optimizer_name}', falling back to Adam")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    logger.info(f"[OPTIMIZER] Created {type(optimizer).__name__} with parameters: {optimizer.defaults}")
    return optimizer

def create_advanced_loss_function(loss_type, **kwargs):
    """Create advanced loss function with enhanced error handling"""
    logger.info(f"[LOSS] Creating advanced loss function: {loss_type}")
    
    # Try advanced losses first if available
    if ADVANCED_LOSSES_AVAILABLE:
        try:
            loss_fn = create_advanced_loss(loss_type, **kwargs)
            logger.info(f"[LOSS] Successfully created advanced loss: {loss_type}")
            return loss_fn
        except ValueError as e:
            logger.warning(f"[LOSS] Advanced loss '{loss_type}' not found: {e}")
        except Exception as e:
            logger.error(f"[LOSS] Failed to create advanced loss '{loss_type}': {e}")
    
    # Try loss manager if available
    if LOSS_MANAGER_AVAILABLE:
        try:
            loss_config = {'type': loss_type, **kwargs}
            loss_fn = LossManager.create_loss_function(loss_config)
            logger.info(f"[LOSS] Successfully created loss via LossManager: {loss_type}")
            return loss_fn
        except Exception as e:
            logger.warning(f"[LOSS] LossManager failed for '{loss_type}': {e}")
    
    # Fallback to standard losses
    logger.info(f"[LOSS] Falling back to standard loss functions")
    
    if loss_type.lower() in ['dice', 'dicelloss']:
        return MonaiDiceLoss(sigmoid=True)
    elif loss_type.lower() in ['bce', 'bceloss']:
        return torch.nn.BCEWithLogitsLoss()
    elif loss_type.lower() in ['mse', 'mseloss']:
        return torch.nn.MSELoss()
    elif loss_type.lower() in ['tversky', 'tversky_precision', 'tversky_recall']:
        # Fallback Tversky implementation
        logger.warning(f"[LOSS] Using fallback Dice loss instead of {loss_type}")
        return MonaiDiceLoss(sigmoid=True)
    elif loss_type.lower() in ['crossentropy', 'cross_entropy']:
        return torch.nn.CrossEntropyLoss()
    else:
        logger.warning(f"[LOSS] Unknown loss type '{loss_type}', using Dice loss")
        return MonaiDiceLoss(sigmoid=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train or run inference with MONAI U-Net model for coronary segmentation')
    parser.add_argument('--save-training-template', action='store_true', help='Save a training config template and exit')
    parser.add_argument('--mode', choices=['train', 'predict'], required=False, help="Mode to run in: train or predict")
    # Model parameters
    parser.add_argument('--model-family', type=str, default=None, help='Model family name for registry and organization (auto-derived from model-type if not set)')
    parser.add_argument('--model-type', type=str, default='unet', help='Model type/architecture')
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', 
                       choices=['adam', 'sgd', 'rmsprop', 'adamw'], 
                       help='Optimizer to use for training')
    parser.add_argument('--data-path', type=str, help='Path to dataset')
    parser.add_argument('--dataset-type', type=str, default='auto', 
                       choices=['auto', 'coronary', 'arcade_binary', 'arcade_semantic', 'arcade_stenosis', 'arcade_classification'],
                       help='Type of dataset to use')
    parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--mlflow-run-id', type=str, help='MLflow run ID')
    parser.add_argument('--celery-task-id', type=str, help='Celery task ID for status updates')
    parser.add_argument('--model-id', type=int, help='Database model ID for callback')
    # Augmentation parameters
    parser.add_argument('--random-flip', action='store_true', help='Enable random flip augmentation')
    parser.add_argument('--random-rotate', action='store_true', help='Enable random rotation augmentation')
    parser.add_argument('--random-scale', action='store_true', help='Enable random scaling augmentation')
    parser.add_argument('--random-intensity', action='store_true', help='Enable random intensity scaling')
    parser.add_argument('--crop-size', type=int, default=128, help='Size of random crop and target resolution')
    parser.add_argument('--threshold', type=lambda x: None if x.lower() == 'none' else float(x), default=0.5, help='Binary segmentation threshold for hard predictions')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of data loading workers (conservative for Docker)')
    
    # Learning rate scheduler parameters
    parser.add_argument('--lr-scheduler', type=str, default='none', 
                       choices=['none', 'plateau', 'step', 'exponential', 'cosine', 'adaptive'],
                       help='Learning rate scheduler type')
    parser.add_argument('--lr-patience', type=int, default=5, help='Patience for plateau scheduler')
    parser.add_argument('--lr-factor', type=float, default=0.5, help='Factor to reduce learning rate')
    parser.add_argument('--lr-step-size', type=int, default=10, help='Step size for step scheduler')
    parser.add_argument('--lr-gamma', type=float, default=0.1, help='Gamma for step/exponential scheduler')
    parser.add_argument('--min-lr', type=float, default=1e-7, help='Minimum learning rate threshold')
    
    # Loss Function parameters
    parser.add_argument('--loss-type', type=str, default='mixed', 
                       choices=['dice', 'bce', 'mixed', 'crossentropy'],
                       help='Type of loss function to use')
    parser.add_argument('--dice-weight', type=float, default=0.7,
                       help='Weight for Dice loss in mixed loss (0.0-1.0)')
    parser.add_argument('--bce-weight', type=float, default=0.3,
                       help='Weight for BCE loss in mixed loss (0.0-1.0)')
    parser.add_argument('--loss-smooth', type=float, default=1e-6,
                       help='Smoothing factor for loss functions')
    
    # Enhanced Checkpointing parameters
    parser.add_argument('--checkpoint-freq', type=str, default='best',
                       choices=['best', 'epoch', 'interval'],
                       help='Frequency of saving checkpoints')
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                       help='Interval for saving checkpoints when using interval mode')
    parser.add_argument('--max-checkpoints', type=int, default=5,
                       help='Maximum number of checkpoints to keep')
    parser.add_argument('--checkpoint-metric', type=str, default='val_dice',
                       choices=['val_dice', 'val_loss', 'train_dice', 'train_loss'],
                       help='Metric to monitor for best checkpoint selection')
    parser.add_argument('--checkpoint-mode', type=str, default='max',
                       choices=['min', 'max'],
                       help='Mode for checkpoint metric (min for loss, max for accuracy/dice)')
    
    # Early Stopping parameters
    parser.add_argument('--use-early-stopping', action='store_true', help='Enable early stopping during training')
    parser.add_argument('--early-stopping-patience', type=int, default=10, help='Number of epochs to wait for improvement before stopping')
    parser.add_argument('--early-stopping-min-epochs', type=int, default=20, help='Minimum number of epochs before early stopping can occur')
    parser.add_argument('--early-stopping-min-delta', type=float, default=1e-4, help='Minimum improvement required to reset patience counter')
    parser.add_argument('--early-stopping-metric', type=str, default='val_dice', 
                       choices=['val_dice', 'val_loss', 'val_accuracy'],
                       help='Metric to monitor for early stopping')
    
    # Enhanced Training parameters
    parser.add_argument('--loss-function', type=str, default='combined',
                       choices=['bce', 'dice', 'iou', 'combined', 'focal', 'focal_segmentation', 'balanced_segmentation', 
                               'dice_focused', 'jaccard_based', 'tversky_recall', 'tversky_precision', 
                               'focal_advanced', 'combo_dice_bce_focal', 'boundary_aware', 'weighted_bce_adaptive',
                               'tversky', 'combo_dice_bce', 'soft_dice', 'weighted_bce', 'boundary', 'stable_bce'],
                       help='Loss function type for training')
    
    parser.add_argument('--primary-metric', type=str, default='dice',
                       choices=['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1'],
                       help='Primary metric for model evaluation and selection (independent of loss function)')
    
    # Medical Preprocessing parameters
    parser.add_argument('--use-medical-preprocessing', action='store_true',
                       help='Enable advanced medical image preprocessing (CLAHE, unsharp masking, etc.)')
    parser.add_argument('--medical-preprocessing-type', type=str, default='angiography',
                       choices=['angiography', 'ct_coronary', 'oct_coronary', 'general'],
                       help='Type of medical preprocessing to apply')
    parser.add_argument('--preprocessing-clahe-clip-limit', type=float, default=3.0,
                       help='CLAHE clip limit for contrast enhancement (1.0-8.0)')
    parser.add_argument('--preprocessing-clahe-tile-size', type=int, default=8,
                       help='CLAHE tile grid size (4-16)')
    parser.add_argument('--preprocessing-use-unsharp-masking', action='store_true',
                       help='Enable unsharp masking for edge enhancement')
    parser.add_argument('--preprocessing-unsharp-radius', type=float, default=1.0,
                       help='Unsharp masking radius (0.5-3.0)')
    parser.add_argument('--preprocessing-unsharp-amount', type=float, default=1.0,
                       help='Unsharp masking amount (0.5-2.0)')
    parser.add_argument('--preprocessing-use-frangi', action='store_true',
                       help='Enable Frangi vesselness filter for vessel enhancement')
    parser.add_argument('--preprocessing-frangi-scale-range', type=str, default='1,10',
                       help='Frangi scale range as min,max (e.g., "1,10")')
    parser.add_argument('--preprocessing-frangi-scale-step', type=float, default=2.0,
                       help='Frangi scale step size (1.0-3.0)')
    parser.add_argument('--preprocessing-use-histogram-equalization', action='store_true',
                       help='Enable histogram equalization for contrast improvement')
    parser.add_argument('--preprocessing-use-denoising', action='store_true',
                       help='Enable denoising filters')
    parser.add_argument('--preprocessing-noise-variance', type=float, default=0.1,
                       help='Denoising variance parameter (0.01-0.5)')
    parser.add_argument('--preprocessing-intensity-range', type=str, default='auto',
                       help='Intensity normalization range (auto, 0-1, 0-255, or min,max)')
    parser.add_argument('--preprocessing-gamma-correction', type=float, default=1.0,
                       help='Gamma correction value (0.5-2.0, 1.0=no correction)')
    parser.add_argument('--preprocessing-vessel-enhancement-sigma', type=float, default=1.0,
                       help='Vessel enhancement sigma parameter (0.5-3.0)')
    parser.add_argument('--preprocessing-custom-pipeline', type=str, default='',
                       help='Custom preprocessing pipeline (comma-separated: clahe,unsharp,frangi,denoise)')
    parser.add_argument('--use-loss-scheduling', action='store_true',
                       help='Enable dynamic loss weight scheduling during training')
    parser.add_argument('--loss-scheduler-type', type=str, default='adaptive',
                       choices=['adaptive', 'cosine', 'step', 'performance'],
                       help='Type of loss weight scheduler to use')
    parser.add_argument('--checkpoint-strategy', type=str, default='best',
                       choices=['best', 'epoch', 'interval', 'all', 'adaptive', 'performance_based'],
                       help='Checkpoint saving strategy')
    parser.add_argument('--monitor-metric', type=str, default='val_dice',
                       choices=['val_dice', 'val_loss', 'val_accuracy', 'val_iou'],
                       help='Metric to monitor for best model selection')
    parser.add_argument('--use-enhanced-training', action='store_true', default=True,
                       help='Enable enhanced training features (checkpointing, loss scheduling)')
    parser.add_argument('--use-mixed-precision', action='store_true',
                       help='Enable mixed precision training for performance')
    
    # Model architecture configuration
    parser.add_argument('--model-size', type=str, default='standard',
                       choices=['micro', 'tiny', 'small', 'standard', 'large', 'xl', 'custom'],
                       help='Model size configuration')
    parser.add_argument('--custom-channels', type=str, default='',
                       help='Custom channel configuration for custom model size (comma-separated)')
    parser.add_argument('--use-attention', action='store_true',
                       help='Enable attention mechanisms in the model')
    parser.add_argument('--use-deep-architecture', action='store_true',
                       help='Enable deeper architecture with more layers')
    parser.add_argument('--use-residual-connections', action='store_true',
                       help='Enable residual connections in the model')
    
    # Prediction parameters
    parser.add_argument('--model-path', type=str, help='Path to trained model weights')
    parser.add_argument('--input-path', type=str, help='Path to input image or directory')
    parser.add_argument('--output-dir', type=str, help='Directory to save predictions')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on')
    parser.add_argument('--weights-path', type=str, help='Optional path to a .pth file for inference')
    args, unknown = parser.parse_known_args()
    if args.save_training_template:
        template = {
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "data_path": "<path>",
            "validation_split": 0.2,
            "mlflow_run_id": "<run_id>",
            "model_id": None,
            "random_flip": False,
            "random_rotate": False,
            "random_scale": False,
            "random_intensity": False,
            "crop_size": 128,
            "num_workers": 1,  # Conservative for Docker
            "mode": "train"
        }
        import json
        print(json.dumps(template, indent=4))
        sys.exit(0)
    if not args.mode:
        parser.print_help()
        print("\nERROR: You must specify --mode=train or --mode=predict.")
        sys.exit(2)
    return args

def detect_num_classes_from_masks(dataset_loaders, dataset_type="auto", max_samples=50):
    """
    Dynamically detect the number of classes from mask data for semantic segmentation
    
    Args:
        dataset_loaders: Either tuple of (train_loader, val_loader) or (train_ds, val_ds)
        dataset_type: Type of dataset ("auto", "arcade_semantic", "binary", etc.)
        max_samples: Maximum number of samples to check
    
    Returns:
        dict with detected class information: {
            'num_classes': int,
            'class_type': str ('binary', 'multi_class', 'semantic'),
            'unique_values': list,
            'max_channels': int (for one-hot encoded masks)
        }
    """
    logger.info(f"[CLASS DETECTION] Detecting number of classes from mask data...")
    
    try:
        # Handle different dataset loader types
        if hasattr(dataset_loaders[0], '__iter__') and hasattr(dataset_loaders[0], 'dataset'):
            # DataLoader objects (ARCADE)
            train_loader, val_loader = dataset_loaders
            train_dataset = train_loader.dataset
        else:
            # Dataset objects (MONAI)
            train_dataset, val_dataset = dataset_loaders
        
        # Special handling for ARCADE dataset types
        dataset_class_name = str(train_dataset.__class__.__name__) if hasattr(train_dataset, '__class__') else ""
        
        if 'ARCADEArteryClassification' in dataset_class_name:
            logger.info(f"[CLASS DETECTION] ARCADEArteryClassification dataset detected")
            logger.info(f"[CLASS DETECTION] This is a classification task: binary mask → left/right artery")
            logger.info(f"[CLASS DETECTION] Output should be 2 classes (0=right, 1=left)")
            return {
                'num_classes': 2,  # Classification task: 2 output classes
                'class_type': 'classification',
                'unique_values': [0, 1],
                'max_channels': 1,
                'task_type': 'artery_classification'
            }
        
        elif 'ARCADESemanticSegmentation' in dataset_class_name and 'Binary' not in dataset_class_name:
            logger.info(f"[CLASS DETECTION] ARCADESemanticSegmentation dataset detected")
            logger.info(f"[CLASS DETECTION] This is multi-class semantic segmentation")
            logger.info(f"[CLASS DETECTION] Expected: 27 classes (background + 26 coronary segments)")
            return {
                'num_classes': 27,  # Semantic segmentation: 27 classes
                'class_type': 'semantic_onehot',
                'unique_values': [0, 1],
                'max_channels': 27,
                'task_type': 'semantic_segmentation'
            }
        
        elif 'ARCADEBinarySegmentation' in dataset_class_name:
            logger.info(f"[CLASS DETECTION] ARCADEBinarySegmentation dataset detected")
            logger.info(f"[CLASS DETECTION] This is binary segmentation: image → binary mask")
            logger.info(f"[CLASS DETECTION] Output should be 1 class (foreground vs background)")
            return {
                'num_classes': 1,  # Binary segmentation: 1 output channel
                'class_type': 'binary',
                'unique_values': [0, 1],
                'max_channels': 1,
                'task_type': 'binary_segmentation'
            }
        
        elif 'ARCADEStenosisDetection' in dataset_class_name:
            logger.info(f"[CLASS DETECTION] ARCADEStenosisDetection dataset detected")
            logger.info(f"[CLASS DETECTION] This is object detection: image → bounding boxes")
            logger.info(f"[CLASS DETECTION] Output should be 1 class (stenosis detection)")
            return {
                'num_classes': 1,  # Object detection: 1 class (stenosis)
                'class_type': 'detection',
                'unique_values': [0, 1],
                'max_channels': 1,
                'task_type': 'stenosis_detection'
            }
        
        elif 'ARCADEStenosisSegmentation' in dataset_class_name:
            logger.info(f"[CLASS DETECTION] ARCADEStenosisSegmentation dataset detected")
            logger.info(f"[CLASS DETECTION] This is stenosis binary segmentation")
            logger.info(f"[CLASS DETECTION] Output should be 1 class (stenosis vs background)")
            return {
                'num_classes': 1,  # Binary stenosis segmentation
                'class_type': 'binary',
                'unique_values': [0, 1],
                'max_channels': 1,
                'task_type': 'stenosis_segmentation'
            }
        
        elif 'ARCADESemanticSegmentationBinary' in dataset_class_name:
            logger.info(f"[CLASS DETECTION] ARCADESemanticSegmentationBinary dataset detected")
            logger.info(f"[CLASS DETECTION] This is binary mask → semantic segmentation")
            logger.info(f"[CLASS DETECTION] Expected: 26 classes (coronary segments without background)")
            return {
                'num_classes': 26,  # Semantic from binary: 26 segments
                'class_type': 'semantic_onehot',
                'unique_values': [0, 1],
                'max_channels': 26,
                'task_type': 'semantic_segmentation_binary'
            }
        
        # Collect unique values and shapes from masks
        all_unique_values = set()
        mask_shapes = []
        mask_channels = []
        samples_checked = 0
        
        logger.info(f"[CLASS DETECTION] Checking up to {max_samples} samples from training dataset...")
        
        # Check training dataset samples
        for i in range(min(len(train_dataset), max_samples)):
            try:
                if hasattr(train_dataset, '__getitem__'):
                    image, mask = train_dataset[i]
                else:
                    # For some dataset implementations
                    sample = train_dataset[i]
                    if isinstance(sample, dict):
                        image = sample.get('image', sample.get('img'))
                        mask = sample.get('label', sample.get('mask'))
                    else:
                        image, mask = sample
                
                # Convert to numpy for analysis
                if hasattr(mask, 'numpy'):
                    mask_array = mask.numpy()
                elif hasattr(mask, 'cpu'):
                    mask_array = mask.cpu().numpy()
                else:
                    mask_array = np.array(mask)
                
                # Record shape and channels
                mask_shapes.append(mask_array.shape)
                
                # Determine number of channels based on shape
                if len(mask_array.shape) == 4:
                    # Batch dimension included (B, C, H, W) or (B, H, W, C)
                    if mask_array.shape[1] < mask_array.shape[3]:  # Likely (B, C, H, W)
                        mask_channels.append(mask_array.shape[1])
                    else:  # Likely (B, H, W, C)
                        mask_channels.append(mask_array.shape[3])
                elif len(mask_array.shape) == 3:
                    # Either (C, H, W) or (H, W, C)
                    if mask_array.shape[0] < min(mask_array.shape[1], mask_array.shape[2]):
                        # Likely (C, H, W) format
                        mask_channels.append(mask_array.shape[0])
                        mask_array = mask_array.transpose(1, 2, 0)  # Convert to (H, W, C)
                    else:
                        # Likely (H, W, C) format
                        mask_channels.append(mask_array.shape[2])
                elif len(mask_array.shape) == 2:
                    # Single channel (H, W)
                    mask_channels.append(1)
                else:
                    # Unknown format, default to 1 channel
                    mask_channels.append(1)
                
                # Debug logging for channel detection
                logger.info(f"[CLASS DETECTION] Sample {i}: shape={mask_array.shape}, detected_channels={mask_channels[-1]}")
                
                # For multi-channel masks, check each channel
                if len(mask_array.shape) == 3 and mask_array.shape[2] > 1:
                    # Multi-channel format (H, W, C) - check each channel
                    logger.info(f"[CLASS DETECTION] Processing multi-channel mask with {mask_array.shape[2]} channels")
                    for c in range(mask_array.shape[2]):
                        channel_data = mask_array[:, :, c]
                        if np.any(channel_data > 0):
                            all_unique_values.update(np.unique(channel_data))
                else:
                    # Single channel or already processed
                    if len(mask_array.shape) > 2:
                        mask_array = mask_array.squeeze()
                    all_unique_values.update(np.unique(mask_array))
                
                samples_checked += 1
                
                # Early termination for clear cases
                if samples_checked >= 10 and len(all_unique_values) > 0:
                    break
                    
            except Exception as e:
                logger.warning(f"[CLASS DETECTION] Error processing sample {i}: {e}")
                continue
        
        # Analyze collected data
        unique_values = sorted(list(all_unique_values))
        max_channels = max(mask_channels) if mask_channels else 1
        
        logger.info(f"[CLASS DETECTION] Analyzed {samples_checked} samples")
        # logger.info(f"[CLASS DETECTION] Unique mask values: {unique_values}")
        logger.info(f"[CLASS DETECTION] Max channels found: {max_channels}")
        logger.info(f"[CLASS DETECTION] Typical mask shape: {mask_shapes[0] if mask_shapes else 'Unknown'}")
        
        # Determine class type and count
        class_info = _analyze_class_distribution(unique_values, max_channels, dataset_type)
        
        logger.info(f"[CLASS DETECTION] Detection result: {class_info}")
        return class_info
        
    except Exception as e:
        logger.error(f"[CLASS DETECTION] Failed to detect classes: {e}")
        # Return safe defaults
        return {
            'num_classes': 1,
            'class_type': 'binary',
            'unique_values': [0, 1],
            'max_channels': 1
        }

def _analyze_class_distribution(unique_values, max_channels, dataset_type):
    """Analyze unique values and channels to determine class configuration"""
    
    num_unique = len(unique_values)
    
    # Handle one-hot encoded semantic segmentation (ARCADE style)
    if max_channels > 2:
        logger.info(f"[CLASS DETECTION] One-hot encoded semantic segmentation detected with {max_channels} channels")
        return {
            'num_classes': max_channels,
            'class_type': 'semantic_onehot',
            'unique_values': unique_values,
            'max_channels': max_channels
        }
    
    # Handle binary segmentation
    elif num_unique == 2 and set(unique_values) <= {0, 1, 255}:
        logger.info(f"[CLASS DETECTION] Binary segmentation detected")
        return {
            'num_classes': 1,
            'class_type': 'binary',
            'unique_values': unique_values,
            'max_channels': 1
        }
    
    # Handle grayscale binary masks that will be auto-thresholded during training
    elif num_unique > 2:
        min_val = min(unique_values)
        max_val = max(unique_values)
        
        # Check if this looks like grayscale binary masks that need thresholding
        if (min_val == 0 and max_val == 255 and num_unique >= 50) or \
           (min_val >= 0 and max_val <= 1.0 and num_unique >= 10):
            # This is a grayscale mask that will be auto-thresholded to binary during training
            logger.info(f"[CLASS DETECTION] Grayscale binary mask detected ({num_unique} values) - will be auto-thresholded to binary")
            logger.info(f"[CLASS DETECTION] Range: [{min_val}, {max_val}] → will become [0, 1] during training")
            return {
                'num_classes': 1,
                'class_type': 'binary',
                'unique_values': [0, 1],  # What it will become after thresholding
                'max_channels': 1
            }
        else:
            # True multi-class semantic segmentation
            num_classes = num_unique if 0 in unique_values else num_unique + 1
            logger.info(f"[CLASS DETECTION] Multi-class semantic segmentation detected with {num_classes} classes")
            return {
                'num_classes': num_classes,
                'class_type': 'semantic_single',
                'unique_values': unique_values,
                'max_channels': 1
            }
    
    # Default to binary
    else:
        logger.info(f"[CLASS DETECTION] Defaulting to binary segmentation")
        return {
            'num_classes': 1,
            'class_type': 'binary',
            'unique_values': unique_values,
            'max_channels': 1
        }

def train_model(args):
    # Setup signal handlers first (must be in main thread)
    setup_signal_handlers()
    
    # Set up logging first thing
    import sys
    import json  # Add json import at the beginning
    
    # Get model directory from Django model if available, otherwise create one
    model_dir = None
    if hasattr(args, 'model_id') and args.model_id is not None and DJANGO_AVAILABLE:
        try:
            # Try multiple import paths for Django model
            try:
                from core.apps.ml_manager.models import MLModel
            except ImportError:
                try:
                    from apps.ml_manager.models import MLModel
                except ImportError:
                    from ml_manager.models import MLModel
                    
            model_obj = MLModel.objects.get(pk=args.model_id)
            if model_obj.model_directory:
                model_dir = model_obj.model_directory
                startup_logger.info(f"[SETUP] Using model directory from database: {model_dir}")
            else:
                startup_logger.warning(f"[SETUP] Model {args.model_id} has no model_directory set")
        except Exception as e:
            startup_logger.warning(f"[SETUP] Could not get model directory from database: {e}")
    
    # Fallback to creating model directory if not set
    if not model_dir:
        # Use model_type as model_family if not explicitly set
        model_family = getattr(args, 'model_family', None)
        if not model_family:
            model_family = getattr(args, 'model_type', 'unet').upper().replace('_', '-')
        
        mlflow_run_id = getattr(args, 'mlflow_run_id', None)
        model_dir, unique_id = create_organized_model_directory(
            model_id=args.model_id, 
            model_family=model_family, 
            version="1.0.0",
            mlflow_run_id=mlflow_run_id
        )
        startup_logger.info(f"[SETUP] Created new model directory: {model_dir}")
        
        # Save the model directory to the database for future log loading
        if DJANGO_AVAILABLE and args.model_id:
            try:
                # Try multiple import paths for Django model
                try:
                    from core.apps.ml_manager.models import MLModel
                except ImportError:
                    try:
                        from apps.ml_manager.models import MLModel
                    except ImportError:
                        from ml_manager.models import MLModel
                        
                model_obj = MLModel.objects.get(id=args.model_id)
                # Save absolute path that works both in container and local environments
                absolute_model_dir = os.path.abspath(model_dir)
                # Extract the actual folder name from the path to use as unique_identifier
                actual_folder_name = os.path.basename(absolute_model_dir)
                
                startup_logger.info(f"[SETUP] Before update - model_directory: {model_obj.model_directory}")
                startup_logger.info(f"[SETUP] Before update - unique_identifier: {model_obj.unique_identifier}")
                
                model_obj.model_directory = absolute_model_dir
                model_obj.unique_identifier = actual_folder_name
                model_obj.save(update_fields=['model_directory', 'unique_identifier'])
                
                startup_logger.info(f"[SETUP] After update - model_directory: {model_obj.model_directory}")
                startup_logger.info(f"[SETUP] After update - unique_identifier: {model_obj.unique_identifier}")
                startup_logger.info(f"[SETUP] Saved model directory to database: {absolute_model_dir}")
                startup_logger.info(f"[SETUP] Updated unique_identifier to match folder name: {actual_folder_name}")
            except Exception as e:
                startup_logger.warning(f"[SETUP] Could not save model directory to database: {e}")
    
    # Set up logging to both model-specific and global locations
    model_log_path = os.path.join(model_dir, 'logs', 'training.log')
    global_log_path = os.path.join(CORE_DATA_DIR, 'logs', 'training.log')  # Use CORE_DATA_DIR for global logs
    
    # Ensure both log directories exist
    os.makedirs(os.path.dirname(model_log_path), exist_ok=True)
    os.makedirs(os.path.dirname(global_log_path), exist_ok=True)
    
    # Clear any existing handlers to avoid duplication
    training_logger = logging.getLogger('training')
    for handler in training_logger.handlers[:]:
        training_logger.removeHandler(handler)
    
    # Create a training-specific logger 
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent propagation to root logger
    
    # Create formatter with detailed information
    formatter = logging.Formatter('%(asctime)s %(levelname)s [MODEL-%(process)d] %(message)s')
    
    # Add file handlers for training logs (write mode to start fresh)
    model_handler = logging.FileHandler(model_log_path, mode='w', encoding='utf-8', delay=False)
    model_handler.setFormatter(formatter)
    model_handler.setLevel(logging.INFO)
    logger.addHandler(model_handler)
    
    global_handler = logging.FileHandler(global_log_path, mode='a', encoding='utf-8', delay=False)
    global_handler.setFormatter(formatter) 
    global_handler.setLevel(logging.INFO)
    logger.addHandler(global_handler)
    
    # Add console handler for training logs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # Force flush all handlers
    for handler in logger.handlers:
        if hasattr(handler, 'flush'):
            handler.flush()
    
    # Test log to ensure it's working
    logger.info(f"[SETUP] Logging initialized - Model log: {model_log_path}")
    logger.info(f"[SETUP] Global log: {global_log_path}")
    logger.info(f"[SETUP] Model directory: {model_dir}")
    
    # Verify log file was created
    if os.path.exists(model_log_path):
        logger.info(f"[SETUP] ✅ Model log file created successfully: {model_log_path}")
    else:
        logger.error(f"[SETUP] ❌ Failed to create model log file: {model_log_path}")
    
    # Setup MLflow experiment before handling run
    try:
        # Import MLflow utils with fallback paths using flexible import
        mlflow_utils_mod, MLFLOW_UTILS_AVAILABLE = import_module_flexibly("utils.mlflow_utils", "setup_mlflow", "MLFLOW_UTILS")
        if MLFLOW_UTILS_AVAILABLE:
            setup_mlflow = mlflow_utils_mod.setup_mlflow
        else:
            setup_mlflow = None
        
        # Additional MLflow path patches before connection
        def patch_mlflow_paths():
            """Additional aggressive MLflow path patches"""
            try:
                import mlflow.store.artifact.local_artifact_repo
                import mlflow.utils.file_utils
                
                # Patch mlflow file_utils.mkdir to prevent host path access
                original_mkdir = mlflow.utils.file_utils.mkdir
                def safe_mkdir(path):
                    if isinstance(path, str) and '/home/rafal' in path:
                        safe_path = path.replace('/home/rafal', '/app/core/data/mlflow')
                        logger.warning(f"[MLFLOW_PATCH] Redirected mkdir from {path} to {safe_path}")
                        return original_mkdir(safe_path)
                    return original_mkdir(path)
                mlflow.utils.file_utils.mkdir = safe_mkdir
                
                # Also patch os.makedirs at MLflow level
                original_os_makedirs = os.makedirs
                def mlflow_safe_makedirs(name, mode=0o777, exist_ok=False):
                    if isinstance(name, str) and '/home/rafal' in name:
                        safe_name = name.replace('/home/rafal', '/app/core/data/mlflow')
                        logger.warning(f"[MLFLOW_PATCH] Redirected os.makedirs from {name} to {safe_name}")
                        return original_os_makedirs(safe_name, mode, exist_ok)
                    return original_os_makedirs(name, mode, exist_ok)
                
                # Apply the patch to os module that MLflow uses
                import mlflow.utils.file_utils
                if hasattr(mlflow.utils.file_utils, 'os'):
                    mlflow.utils.file_utils.os.makedirs = mlflow_safe_makedirs
                
                logger.info("[MLFLOW_PATCH] Applied aggressive path patches")
            except Exception as e:
                logger.warning(f"[MLFLOW_PATCH] Could not apply patches: {e}")
        
        patch_mlflow_paths()
        
        # Setup MLflow connection (experiment is already set by Celery task)
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
        
        # Verify MLflow connection by listing experiments
        try:
            experiments = mlflow.search_experiments()
            logger.info(f"[MLFLOW] Connection verified - found {len(experiments)} experiments")
            
            # Get current experiment info (should be set by Celery task)
            current_experiment = mlflow.get_experiment_by_name('coronary-experiments-fixed')
            if current_experiment:
                logger.info(f"[MLFLOW] Using experiment: {current_experiment.name} (ID: {current_experiment.experiment_id})")
                logger.info(f"[MLFLOW] Artifact location: {current_experiment.artifact_location}")
            else:
                logger.warning("[MLFLOW] Could not find 'coronary-experiments-fixed' experiment")
        except Exception as verify_error:
            logger.warning(f"[MLFLOW] Connection verification failed: {verify_error}")
            
    except Exception as e:
        logger.warning(f"[MLFLOW] Failed to setup MLflow connection: {e}")
    
    # Handle MLflow run in subprocess context
    # The run should already be active from Celery task, just connect to it
    current_run = mlflow.active_run()
    if current_run and current_run.info.run_id == args.mlflow_run_id:
        logger.info(f"[MLFLOW] Continuing active run {args.mlflow_run_id}")
    elif args.mlflow_run_id:
        # Connect to the existing run created by Celery task
        if current_run:
            mlflow.end_run()  # End any existing run first
        try:
            mlflow.start_run(run_id=args.mlflow_run_id)
            logger.info(f"[MLFLOW] Connected to existing run {args.mlflow_run_id} from Celery task")
            
            # Enable MLflow autologging for system metrics and other automated logging
            try:
                mlflow.autolog(
                    log_models=False,  # We handle model logging manually
                    log_datasets=False,  # We handle dataset info manually
                    disable=False,
                    exclusive=False,
                    log_traces=False
                )
                logger.info("[MLFLOW] Autologging enabled")
            except Exception as autolog_error:
                logger.warning(f"[MLFLOW] Failed to enable autologging: {autolog_error}")
                
        except Exception as e:
            logger.error(f"[MLFLOW] Failed to connect to run {args.mlflow_run_id}: {e}")
            logger.error("[MLFLOW] Train.py cannot create new runs - run must be created by Celery task")
            return "error", f"MLflow run connection failed: {e}"
    else:
        logger.error("[MLFLOW] No MLflow run ID provided - run must be created by Celery task")
        return "error", "No MLflow run ID provided"
    
    training_start_time = time.time()  # Track total training duration

    # Initialize system monitoring for MLflow
    system_monitor = None
    try:
        # Import system monitor with fallback paths
        system_monitor_mod, SYSTEM_MONITOR_AVAILABLE = import_module_flexibly("utils.system_monitor", "SystemMonitor", "SYSTEM_MONITOR")
        if SYSTEM_MONITOR_AVAILABLE:
            SystemMonitor = system_monitor_mod.SystemMonitor
            system_monitor = SystemMonitor(log_interval=30, enable_gpu=True)  # Log every 30 seconds
        else:
            logger.warning("[MONITORING] SystemMonitor not available")
            system_monitor = None
        
        # Check if system monitor is enabled
        if system_monitor.enabled:
            system_monitor.start_monitoring()
            logger.info("[MONITORING] ✅ System monitoring started - logging to MLflow every 30 seconds")
        else:
            logger.warning("[MONITORING] ⚠️ System monitoring disabled (psutil not available)")
            system_monitor = None
    except Exception as e:
        logger.warning(f"[MONITORING] ❌ Failed to start system monitoring: {e}")
        system_monitor = None

    # Parameters are already logged by Celery task, just update training status
    try:
        mlflow.set_tag('training_status', 'training_active')
        mlflow.set_tag('subprocess_started', 'true')
        logger.info("[MLFLOW] Updated training status tags")
    except Exception as e:
        logger.warning(f"[MLFLOW] Failed to update tags: {e}")
    
    # Set threshold for use in training (already logged as parameter by Celery)
    threshold = args.threshold if args.threshold is not None else 0.5
    
    callback = None
    if hasattr(args, 'model_id') and args.model_id is not None:
        logger.info(f"[CALLBACK SETUP] Creating callback for model_id: {args.model_id}")
        try:
            # Use TrainingCallback that was imported at the module level
            # This will either be the real callback or our dummy implementation
            callback = TrainingCallback(args.model_id, args.mlflow_run_id)
            # Store the model directory path in Django model
            callback.set_model_directory(model_dir)
            # Set status to 'loading' when training script starts
            callback.on_training_start()
            logger.info(f"[CALLBACK SETUP] Callback created and initialized successfully")
        except Exception as e:
            logger.error(f"[CALLBACK SETUP] Failed to create callback: {e}")
            import traceback
            logger.error(f"[CALLBACK SETUP] Traceback: {traceback.format_exc()}")
            callback = None
    else:
        logger.info("[CALLBACK SETUP] No model_id provided, callback will not be created")

    try:
        logger.info("[TRAINING] Starting training with parameters: %s", vars(args))
        
        # Handle device selection based on args.device parameter
        if hasattr(args, 'device') and args.device:
            if args.device == 'auto':
                # Auto-detect best available device: MPS > CUDA > CPU
                if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                    device = torch.device("mps")
                elif torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    device = torch.device("cpu")
            elif args.device == 'mps':
                if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                    device = torch.device("mps")
                else:
                    logger.warning("MPS requested but not available, falling back to CPU")
                    device = torch.device("cpu")
            elif args.device == 'cuda':
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    device = torch.device("cpu")
            else:
                device = torch.device(args.device)  # cpu or specific device
        else:
            # Fallback to auto-detection if no device specified
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            
        logger.info(f"[TRAINING] Using device: {device}")
        
        # Log dataset path verification with auto-detection and fallback creation
        logger.info(f"[DATASET] Data path: {args.data_path}")
        
        # Try to resolve and create the dataset path if it doesn't exist
        original_data_path = args.data_path
        resolved_data_path = None
        
        # First try the original path
        if os.path.exists(args.data_path):
            resolved_data_path = args.data_path
            logger.info(f"[DATASET] Using existing data path: {resolved_data_path}")
        else:
            logger.warning(f"[DATASET] Original data path does not exist: {args.data_path}")
            
            # Try alternative paths for container/local environment compatibility
            # Priority order: relative paths first (for local dev), then absolute container paths
            alternative_paths = []
            
            # Convert absolute container paths to relative paths (for local development)
            if args.data_path.startswith('/app/core/'):
                relative_path = args.data_path.replace('/app/core/', '', 1)
                alternative_paths.append(relative_path)
                alternative_paths.append(f'core/{relative_path}')
                logger.info(f"[DATASET] Trying relative paths for local dev: {relative_path}, core/{relative_path}")
            elif args.data_path.startswith('/app/'):
                relative_path = args.data_path.replace('/app/', '', 1)
                alternative_paths.append(relative_path)
                alternative_paths.append(f'core/{relative_path}')
                logger.info(f"[DATASET] Trying relative paths: {relative_path}, core/{relative_path}")
            
            # If it's already a relative path, try with different base directories
            elif not os.path.isabs(args.data_path):
                alternative_paths.append(args.data_path)  # Try as-is first
                alternative_paths.append(os.path.join('core', args.data_path))
                alternative_paths.append(os.path.join('/app/core', args.data_path))
                alternative_paths.append(os.path.join('/app', args.data_path))
                alternative_paths.append(os.path.abspath(args.data_path))
                logger.info(f"[DATASET] Relative path provided, trying variations")
            
            # Add working directory based paths
            alternative_paths.extend([
                os.path.join(os.getcwd(), args.data_path.lstrip('/')),
                os.path.join(os.getcwd(), 'core', args.data_path.lstrip('/')),
            ])
            
            # Add common dataset directory patterns based on basename
            base_name = os.path.basename(args.data_path.rstrip('/'))
            if base_name and base_name != 'datasets':  # Don't duplicate if base_name is just 'datasets'
                alternative_paths.extend([
                    os.path.join('core/data/datasets', base_name),
                    os.path.join('/app/core/data/datasets', base_name),
                    os.path.join(os.getcwd(), 'core/data/datasets', base_name),
                    # Legacy patterns for backwards compatibility
                    os.path.join('data/datasets', base_name),
                    os.path.join('/app/data/datasets', base_name),
                ])
            
            # Try alternative paths
            for alt_path in alternative_paths:
                logger.debug(f"[DATASET] Checking alternative path: {alt_path}")
                if os.path.exists(alt_path):
                    resolved_data_path = alt_path
                    logger.info(f"[DATASET] Found alternative data path: {resolved_data_path}")
                    break
            
            # If still not found, try to create the directory structure
            if not resolved_data_path:
                logger.warning(f"[DATASET] No existing path found, attempting to create directories")
                
                # Try to create common dataset paths
                paths_to_create = [
                    'core/data/datasets',
                    '/app/core/data/datasets',
                    'data/datasets'
                ]
                
                for create_path in paths_to_create:
                    try:
                        os.makedirs(create_path, exist_ok=True)
                        logger.info(f"[DATASET] Created dataset directory: {create_path}")
                        resolved_data_path = create_path
                        break
                    except Exception as e:
                        logger.debug(f"[DATASET] Could not create path {create_path}: {e}")
        
        if not resolved_data_path or not os.path.exists(resolved_data_path):
            logger.error(f"[DATASET] No valid data path found. Original: {original_data_path}")
            if 'alternative_paths' in locals() and alternative_paths:
                logger.error(f"[DATASET] Alternative paths tried: {alternative_paths[:5]}...")  # Show first 5 to avoid spam
            logger.error(f"[DATASET] Current working directory: {os.getcwd()}")
            logger.error(f"[DATASET] Available core/data contents: {os.listdir('core/data') if os.path.exists('core/data') else 'core/data not found'}")
            raise FileNotFoundError(f"Dataset path not found: {original_data_path}. Please ensure the dataset directory exists. Available datasets: {os.listdir('core/data/datasets') if os.path.exists('core/data/datasets') else 'No datasets found'}")
        
        # Update args with resolved path
        args.data_path = resolved_data_path
        logger.info(f"[DATASET] Using resolved data path: {args.data_path}")

        # Get transforms with augmentation and preprocessing parameters
        transform_params = {
            'use_random_flip': getattr(args, 'random_flip', True),
            'use_random_rotate': getattr(args, 'random_rotate', True),
            'use_random_scale': getattr(args, 'random_scale', True),
            'use_random_intensity': getattr(args, 'random_intensity', True),
            'crop_size': getattr(args, 'crop_size', 128),
            'use_medical_preprocessing': getattr(args, 'use_medical_preprocessing', False),
            'medical_preprocessing_type': getattr(args, 'medical_preprocessing_type', 'angiography'),
            # Add all detailed preprocessing parameters
            'preprocessing_clahe_clip_limit': getattr(args, 'preprocessing_clahe_clip_limit', 3.0),
            'preprocessing_clahe_tile_size': getattr(args, 'preprocessing_clahe_tile_size', 8),
            'preprocessing_use_unsharp_masking': getattr(args, 'preprocessing_use_unsharp_masking', False),
            'preprocessing_unsharp_radius': getattr(args, 'preprocessing_unsharp_radius', 1.0),
            'preprocessing_unsharp_amount': getattr(args, 'preprocessing_unsharp_amount', 1.0),
            'preprocessing_use_frangi': getattr(args, 'preprocessing_use_frangi', False),
            'preprocessing_frangi_scale_range': getattr(args, 'preprocessing_frangi_scale_range', '1,10'),
            'preprocessing_frangi_scale_step': getattr(args, 'preprocessing_frangi_scale_step', 2.0),
            'preprocessing_use_histogram_equalization': getattr(args, 'preprocessing_use_histogram_equalization', False),
            'preprocessing_use_denoising': getattr(args, 'preprocessing_use_denoising', False),
            'preprocessing_noise_variance': getattr(args, 'preprocessing_noise_variance', 0.1),
            'preprocessing_intensity_range': getattr(args, 'preprocessing_intensity_range', 'auto'),
            'preprocessing_gamma_correction': getattr(args, 'preprocessing_gamma_correction', 1.0),
            'preprocessing_vessel_enhancement_sigma': getattr(args, 'preprocessing_vessel_enhancement_sigma', 1.0),
            'preprocessing_custom_pipeline': getattr(args, 'preprocessing_custom_pipeline', '')
        }
        
        logger.info("[DATASET] Loading datasets with auto-detection...")
        
        # Use auto-detection to determine dataset type and create appropriate loaders
        try:
            dataset_loaders = get_datasets_with_auto_detection(
                args.data_path, 
                args.validation_split, 
                transform_params, 
                args
            )
            
            # Handle different return types (ARCADE returns DataLoaders, MONAI returns Datasets)
            if isinstance(dataset_loaders[0], MonaiDataLoader) or hasattr(dataset_loaders[0], '__iter__'):
                # ARCADE or pre-built DataLoaders
                train_loader, val_loader = dataset_loaders
                train_samples = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else len(train_loader) * train_loader.batch_size
                val_samples = len(val_loader.dataset) if hasattr(val_loader, 'dataset') else len(val_loader) * val_loader.batch_size
            else:
                # MONAI Datasets - need to create DataLoaders
                train_ds, val_ds = dataset_loaders
                # Use conservative num_workers to avoid shared memory issues in Docker
                num_workers = min(getattr(args, 'num_workers', 1), 1)  # Conservative: max 1 worker
                train_loader = MonaiDataLoader(
                    train_ds, 
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    num_workers=num_workers,
                    persistent_workers=False,  # Disable persistent workers to prevent zombie processes
                    pin_memory=False  # Disable pin_memory to reduce memory pressure
                )
                val_loader = MonaiDataLoader(
                    val_ds, 
                    batch_size=args.batch_size, 
                    shuffle=False, 
                    num_workers=num_workers,
                    persistent_workers=False,  # Disable persistent workers to prevent zombie processes
                    pin_memory=False  # Disable pin_memory to reduce memory pressure
                )
                train_samples = len(train_ds)
                val_samples = len(val_ds)
                
        except Exception as e:
            logger.error(f"[DATASET] Failed to load datasets: {e}")
            # Fallback to original MONAI method
            logger.info("[DATASET] Falling back to MONAI dataset loader...")
            train_ds, val_ds = get_monai_datasets(args.data_path, args.validation_split, transform_params, dataset_type=getattr(args, 'dataset_type', None))
            # Use conservative num_workers to avoid shared memory issues in Docker  
            num_workers = min(getattr(args, 'num_workers', 1), 1)  # Conservative: max 1 worker
            train_loader = MonaiDataLoader(
                train_ds, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=num_workers,
                persistent_workers=False,  # Disable persistent workers to prevent zombie processes
                pin_memory=False  # Disable pin_memory to reduce memory pressure
            )
            val_loader = MonaiDataLoader(
                val_ds, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=num_workers,
                persistent_workers=False,  # Disable persistent workers to prevent zombie processes
                pin_memory=False  # Disable pin_memory to reduce memory pressure
            )
            train_samples = len(train_ds)
            val_samples = len(val_ds)

        # Log dataset info
        if 'train_ds' in locals() and 'val_ds' in locals():
            logger.info(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
        elif 'train_loader' in locals() and 'val_loader' in locals():
            logger.info(f"Training samples: {train_samples}, Validation samples: {val_samples}")
        else:
            logger.warning("[DATASET] Could not determine dataset sizes - dataset loading failed.")
        
        # Dataset loaded successfully, now update status to 'training'
        if callback:
            callback.on_dataset_loaded()
            logger.info("Status updated to 'training' - starting model training")
            
            # Update training data info with detailed dataset statistics  
            detailed_training_info = {
                'data_path': args.data_path,
                'total_samples': train_samples + val_samples,
                'training_samples': train_samples,
                'validation_samples': val_samples,
                'validation_split': args.validation_split,
                'transform_params': transform_params,
                'batch_size': args.batch_size,
                'crop_size': getattr(args, 'crop_size', 128),
                'threshold': getattr(args, 'threshold', 0.5),
                'dataset_type': getattr(args, 'dataset_type', 'auto'),
                'num_workers': getattr(args, 'num_workers', 1)  # Conservative for Docker
            }
            
            # Add dataset-specific paths and file information
            try:
                # Debug logging to track available variables
                available_vars = [k for k in locals().keys() if any(x in k for x in ['ds', 'loader', 'dataset'])]
                logger.info(f"[DATASET] Available variables for metadata: {available_vars}")
                
                # Check if we have MONAI datasets (train_ds, val_ds exist)
                if 'train_ds' in locals() and 'val_ds' in locals():
                    logger.info("[DATASET] Processing MONAI dataset metadata...")
                    if hasattr(train_ds, 'data') and train_ds.data:
                        # For MONAI datasets, extract sample paths
                        train_sample_paths = []
                        val_sample_paths = []
                        
                        # Get sample paths from training dataset
                        for i, sample in enumerate(train_ds.data[:5]):  # First 5 samples
                            if isinstance(sample, dict) and 'image' in sample:
                                train_sample_paths.append({
                                    'image': os.path.basename(sample['image']) if isinstance(sample['image'], str) else f"sample_{i}",
                                    'label': os.path.basename(sample['label']) if isinstance(sample, dict) and 'label' in sample and isinstance(sample['label'], str) else f"label_{i}"
                                })
                        
                        # Get sample paths from validation dataset  
                        if hasattr(val_ds, 'data') and val_ds.data:
                            for i, sample in enumerate(val_ds.data[:5]):  # First 5 samples
                                if isinstance(sample, dict) and 'image' in sample:
                                    val_sample_paths.append({
                                        'image': os.path.basename(sample['image']) if isinstance(sample['image'], str) else f"sample_{i}",
                                        'label': os.path.basename(sample['label']) if isinstance(sample, dict) and 'label' in sample and isinstance(sample['label'], str) else f"label_{i}"
                                    })
                        
                        detailed_training_info.update({
                            'sample_train_files': train_sample_paths,
                            'sample_val_files': val_sample_paths,
                            'train_file_count': len(train_ds.data) if hasattr(train_ds, 'data') else len(train_ds),
                            'val_file_count': len(val_ds.data) if hasattr(val_ds, 'data') else len(val_ds),
                            'dataset_framework': 'MONAI'
                        })
                        
                        logger.info(f"[DATASET] MONAI sample training files: {train_sample_paths[:3]}")
                        logger.info(f"[DATASET] MONAI sample validation files: {val_sample_paths[:3]}")
                        
                # Check if we have ARCADE loaders (train_loader, val_loader exist)  
                elif 'train_loader' in locals() and 'val_loader' in locals():
                    logger.info("[DATASET] Processing ARCADE dataset metadata...")
                    # For ARCADE datasets, get info from DataLoader
                    train_dataset = getattr(train_loader, 'dataset', None)
                    val_dataset = getattr(val_loader, 'dataset', None)
                    
                    if train_dataset and val_dataset:
                        detailed_training_info.update({
                            'dataset_framework': 'ARCADE',
                            'dataset_class': str(type(train_dataset).__name__),
                            'arcade_root': getattr(train_dataset, 'root', 'Unknown'),
                            'arcade_image_set_train': getattr(train_dataset, 'image_set', 'Unknown'),
                            'arcade_image_set_val': getattr(val_dataset, 'image_set', 'Unknown'),
                            'train_file_count': len(train_dataset) if train_dataset else 0,
                            'val_file_count': len(val_dataset) if val_dataset else 0
                        })
                        
                        logger.info(f"[DATASET] ARCADE dataset detected: {type(train_dataset).__name__}")
                        logger.info(f"[DATASET] ARCADE root: {getattr(train_dataset, 'root', 'Unknown')}")
                        logger.info(f"[DATASET] ARCADE train count: {len(train_dataset) if train_dataset else 0}")
                        logger.info(f"[DATASET] ARCADE val count: {len(val_dataset) if val_dataset else 0}")
                else:
                    logger.info("[DATASET] Neither MONAI datasets nor ARCADE loaders found for detailed logging")
                    
            except Exception as e:
                logger.warning(f"[DATASET] Could not extract detailed file information: {e}")
                logger.warning(f"[DATASET] Available variables: {[k for k in locals().keys() if 'ds' in k or 'loader' in k]}")
                
            callback.update_model_metadata(training_data_info=detailed_training_info)
        
        # Get sample batch for model signature - with safe handling
        sample_batch = None
        sample_images = None
        try:
            sample_batch = next(iter(train_loader))
            if isinstance(sample_batch, dict):
                logger.info(f"Sample batch shapes - Image: {sample_batch['image'].shape}, Label: {sample_batch['label'].shape}")
                images = sample_batch['image']
                labels = sample_batch['label']
                sample_images = images
            elif isinstance(sample_batch, (list, tuple)) and len(sample_batch) == 2:
                logger.info(f"Sample batch shapes - Image: {sample_batch[0].shape}, Label: {sample_batch[1].shape}")
                images = sample_batch[0]
                labels = sample_batch[1]
                sample_images = images
                # Create dict-like structure for MLflow signature
                sample_batch = {"image": images, "label": labels}
            else:
                logger.warning(f"Unexpected sample batch type: {type(sample_batch)}, content: {sample_batch}")
                images, labels = None, None
                sample_images = None
                sample_batch = None
        except Exception as e:
            logger.error(f"Failed to get sample batch for model signature: {e}")
            sample_batch = None
            sample_images = None
            images, labels = None, None
        # --- Wizualizacja przykładowych danych wejściowych i masek ---
        try:
            import matplotlib.pyplot as plt
            if images is not None and labels is not None:
                # Log data ranges for debugging normalization issues
                img_min, img_max = images.min().item(), images.max().item()
                label_min, label_max = labels.min().item(), labels.max().item()
                logger.info(f"[DATASET] Sample batch data ranges:")
                logger.info(f"[DATASET]   Images: min={img_min:.4f}, max={img_max:.4f} (dtype: {images.dtype})")
                logger.info(f"[DATASET]   Labels: min={label_min:.4f}, max={label_max:.4f} (dtype: {labels.dtype})")
                
                # Determine and log data range types
                if img_max <= 1.0 and img_min >= 0.0:
                    logger.info(f"[DATASET]   Images appear to be normalized to [0-1] range")
                elif img_max <= 255 and img_min >= 0:
                    logger.info(f"[DATASET]   Images appear to be in [0-255] range")
                else:
                    logger.info(f"[DATASET]   Images in custom range [{img_min:.2f}-{img_max:.2f}]")
                
                if label_max <= 1.0 and label_min >= 0.0:
                    logger.info(f"[DATASET]   Binary masks in [0-1] range (normalized)")
                elif label_max <= 255 and label_min >= 0:
                    logger.info(f"[DATASET]   Binary masks in [0-255] range (needs normalization)")
                else:
                    logger.info(f"[DATASET]   Binary masks in custom range [{label_min:.2f}-{label_max:.2f}]")
                
                # Check for unique values in masks to confirm binary nature
                unique_labels = torch.unique(labels)
                # logger.info(f"[DATASET]   Unique mask values: {unique_labels.cpu().numpy()}")
                
                fig, axes = plt.subplots(2, min(4, images.shape[0]), figsize=(12, 6))
                for i in range(min(4, images.shape[0])):
                    axes[0, i].imshow(images[i, 0].cpu().numpy(), cmap='gray')
                    axes[0, i].set_title(f'Input #{i+1}')
                    axes[0, i].axis('off')
                    axes[1, i].imshow(labels[i, 0].cpu().numpy(), cmap='gray')
                    axes[1, i].set_title(f'Segmentacja #{i+1}')
                    axes[1, i].axis('off')
                plt.tight_layout()
                sample_vis_dir = os.path.join(model_dir, 'artifacts')
                os.makedirs(sample_vis_dir, exist_ok=True)
                sample_vis_path = os.path.join(sample_vis_dir, 'sample_inputs_and_masks.png')
                plt.savefig(sample_vis_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"[DATASET] Saved sample input/mask visualization: {sample_vis_path}")
        except Exception as e:
            logger.warning(f"[DATASET] Could not create sample input/mask visualization: {e}")
        
        # --- DYNAMIC CLASS AND CHANNEL DETECTION ---
        # Detect input channels from batch data
        input_channels = images.shape[1] if images is not None else 1
        logger.info(f"[MODEL CONFIG] Detected input channels: {input_channels}")
        
        # Detect output classes from mask data
        class_info = None
        if 'train_loader' in locals() and 'val_loader' in locals():
            # ARCADE dataset loaders
            logger.info("[MODEL CONFIG] Detecting classes from ARCADE dataset...")
            class_info = detect_num_classes_from_masks((train_loader, val_loader), dataset_type="arcade")
        elif 'train_ds' in locals() and 'val_ds' in locals():
            # MONAI datasets
            logger.info("[MODEL CONFIG] Detecting classes from MONAI dataset...")
            class_info = detect_num_classes_from_masks((train_ds, val_ds), dataset_type="monai")
        else:
            logger.warning("[MODEL CONFIG] No dataset available for class detection, using defaults")
        
        # Prepare model architecture configuration from command line arguments
        model_architecture = {
            'model_size': getattr(args, 'model_size', 'standard'),
            'custom_channels': getattr(args, 'custom_channels', ''),
            'use_attention': getattr(args, 'use_attention', False),
            'use_deep_architecture': getattr(args, 'use_deep_architecture', False),
            'use_residual_connections': getattr(args, 'use_residual_connections', False)
        }
        
        # Add model_architecture to args for get_default_model_config to use
        args.model_architecture = model_architecture
        logger.info(f"[MODEL CONFIG] Model architecture configuration: {model_architecture}")
        
        # Configure model based on detected parameters
        model_config = get_default_model_config(args.model_type, args)
        model_config["in_channels"] = input_channels  # Set input channels
        
        # Set output channels based on class detection
        if class_info:
            output_channels = class_info['num_classes']
            logger.info(f"[MODEL CONFIG] Detected {output_channels} output classes ({class_info['class_type']})")
            logger.info(f"[MODEL CONFIG] Class values found: {class_info['unique_values']}")
            logger.info(f"[MODEL CONFIG] Max channels in masks: {class_info['max_channels']}")
            
            # For one-hot encoded semantic segmentation, use the channel count
            if class_info['class_type'] == 'semantic_onehot':
                model_config["out_channels"] = class_info['max_channels']
                logger.info(f"[MODEL CONFIG] Using {class_info['max_channels']} output channels for one-hot semantic segmentation")
            else:
                model_config["out_channels"] = output_channels
                logger.info(f"[MODEL CONFIG] Using {output_channels} output channels for {class_info['class_type']} segmentation")
        else:
            # Fallback to default
            default_out_channels = 1
            model_config["out_channels"] = default_out_channels
            logger.warning(f"[MODEL CONFIG] Using default {default_out_channels} output channels")
        
        # Create model with dynamically configured parameters
        logger.info(f"[MODEL CONFIG] Final model configuration: {model_config}")
        
        # Pass task type information to model creation
        task_type = class_info.get('task_type') if class_info else None
        logger.info(f"[MODEL CONFIG] Task type for model creation: {task_type}")
        
        model, arch_info = create_model_from_registry(
            args.model_type, 
            device,
            task_type=task_type,
            **model_config
        )

        # Update model metadata if callback is available
        if callback:
            callback.update_model_metadata(
                model_family=args.model_family,
                model_type=args.model_type,
                architecture_info=arch_info
            )
            
            # Prepare training config with class detection info
            training_config = {
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'optimizer': getattr(args, 'optimizer', 'adam'),
                'crop_size': args.crop_size,
                'validation_split': args.validation_split,
                'num_workers': getattr(args, 'num_workers', 2),
                'random_flip': getattr(args, 'random_flip', False),
                'random_rotate': getattr(args, 'random_rotate', False),
                'random_scale': getattr(args, 'random_scale', False),
                'random_intensity': getattr(args, 'random_intensity', False)
            }
            
            # Add class detection information to training config
            if class_info:
                training_config.update({
                    'detected_num_classes': class_info['num_classes'],
                    'detected_class_type': class_info['class_type'],
                    'detected_unique_values': class_info['unique_values'],
                    'detected_max_channels': class_info['max_channels']
                })
                logger.info(f"[MODEL CONFIG] Added class detection info to training config")
            
            callback.update_training_config(training_config)
            callback.update_architecture_info(model, model_config)

        # Model directory was already created during logging setup
        # Save and log model architecture summary
        summary_file = save_model_summary(model, model_dir=model_dir)
        log_artifact_to_model_directory(summary_file, model_dir, "model_summary")
        
        # Save and log training configuration
        config_file = save_config(args, model_dir=model_dir)
        log_artifact_to_model_directory(config_file, model_dir, "training_config")

        # --- VALIDATE MODEL CONFIGURATION WITH DETECTED CLASSES ---
        # Get model output channels to validate against detected classes
        model_output_channels = None
        if hasattr(model, 'outc') and hasattr(model.outc, 'conv'):
            model_output_channels = model.outc.conv.out_channels
        elif hasattr(model, 'out_conv'):
            model_output_channels = model.out_conv.out_channels
        elif hasattr(model, 'segmentation_head'):
            model_output_channels = model.segmentation_head.out_channels
        
        if model_output_channels and class_info:
            logger.info(f"[MODEL VALIDATION] Model output channels: {model_output_channels}")
            logger.info(f"[MODEL VALIDATION] Detected classes: {class_info['num_classes']} ({class_info['class_type']})")
            
            # Validate channel match
            expected_channels = class_info['max_channels'] if class_info['class_type'] == 'semantic_onehot' else class_info['num_classes']
            if model_output_channels == expected_channels:
                logger.info(f"[MODEL VALIDATION] ✅ Model output channels match detected classes")
            else:
                logger.warning(f"[MODEL VALIDATION] ⚠️  Model output channels ({model_output_channels}) don't match expected ({expected_channels})")
                logger.warning(f"[MODEL VALIDATION] This may cause training issues - check model configuration")

        # Configure loss function based on detected class information and args
        logger.info(f"[LOSS CONFIG] Received loss function argument: {args.loss_function}")
        
        # Use our new advanced loss function creation with fallback support
        logger.info(f"[LOSS CONFIG] Using advanced loss configuration with fallback")
        
        # Determine loss parameters based on task and arguments
        loss_kwargs = {}
        
        if class_info and class_info.get('task_type') == 'artery_classification':
            # Classification task
            logger.info(f"[LOSS CONFIG] Using classification loss for artery classification")
            loss_function = create_advanced_loss_function('crossentropy', **loss_kwargs)
        elif args.loss_function == 'combined':
            # Mixed Dice + BCE loss (combined mode)
            logger.info(f"[LOSS CONFIG] Using mixed loss: {getattr(args, 'dice_weight', 0.5):.1%} Dice + {getattr(args, 'bce_weight', 0.5):.1%} BCE")
            loss_kwargs.update({
                'dice_weight': getattr(args, 'dice_weight', 0.5),
                'bce_weight': getattr(args, 'bce_weight', 0.5),
                'smooth': getattr(args, 'loss_smooth', 1e-5)
            })
            loss_function = create_advanced_loss_function('combo_dice_bce', **loss_kwargs)
        elif args.loss_function == 'dice':
            # Pure Dice loss
            logger.info(f"[LOSS CONFIG] Using Dice loss")
            if class_info and class_info['class_type'] == 'semantic_onehot' and class_info['max_channels'] > 1:
                logger.info(f"[LOSS CONFIG] Using multi-class Dice loss for {class_info['max_channels']} classes")
                loss_kwargs.update({'sigmoid': False, 'softmax': True})
            else:
                loss_kwargs.update({'sigmoid': True, 'smooth': getattr(args, 'loss_smooth', 1e-5)})
                logger.info(f"[LOSS CONFIG] Using binary Dice loss")
            loss_function = create_advanced_loss_function('dice', **loss_kwargs)
        elif args.loss_function == 'iou':
            # IoU loss
            logger.info(f"[LOSS CONFIG] Using IoU loss")
            if class_info and class_info['class_type'] == 'semantic_onehot' and class_info['max_channels'] > 1:
                logger.info(f"[LOSS CONFIG] Using multi-class IoU loss for {class_info['max_channels']} classes")
                # Use 1 - IoU as loss (since IoU is a metric, not a loss)
                from monai.losses import GeneralizedDiceLoss
                loss_function = GeneralizedDiceLoss(sigmoid=False, softmax=True)
            else:
                logger.info(f"[LOSS CONFIG] Using binary IoU loss")
                # For binary IoU, we can use Dice loss as they are closely related
                # Or create a custom IoU loss wrapper
                loss_kwargs.update({'sigmoid': True, 'smooth': getattr(args, 'loss_smooth', 1e-5)})
                loss_function = create_advanced_loss_function('dice', **loss_kwargs)
        elif args.loss_function == 'tversky':
            # Tversky loss with balanced alpha/beta
            logger.info(f"[LOSS CONFIG] Using Tversky loss (balanced)")
            loss_kwargs.update({'alpha': 0.5, 'beta': 0.5})
            loss_function = create_advanced_loss_function('tversky', **loss_kwargs)
        elif args.loss_function == 'tversky_recall':
            # Tversky loss optimized for recall (missing fewer arteries)
            logger.info(f"[LOSS CONFIG] Using Tversky loss (recall-focused)")
            loss_kwargs.update({'alpha': 0.3, 'beta': 0.7})  # Emphasize recall
            loss_function = create_advanced_loss_function('tversky', **loss_kwargs)
        elif args.loss_function == 'tversky_precision':
            # Tversky loss optimized for precision (cleaner segmentations)
            logger.info(f"[LOSS CONFIG] Using Tversky loss (precision-focused)")
            loss_kwargs.update({'alpha': 0.7, 'beta': 0.3})  # Emphasize precision
            loss_function = create_advanced_loss_function('tversky', **loss_kwargs)
        elif args.loss_function == 'focal':
            # Focal loss
            logger.info(f"[LOSS CONFIG] Using Focal loss")
            loss_kwargs.update({'alpha': 0.25, 'gamma': 2.0})
            loss_function = create_advanced_loss_function('focal', **loss_kwargs)
        elif args.loss_function == 'combo_dice_bce':
            # Combined Dice + BCE loss
            logger.info(f"[LOSS CONFIG] Using Combined Dice + BCE loss")
            loss_kwargs.update({'dice_weight': 0.5, 'bce_weight': 0.5})
            loss_function = create_advanced_loss_function('combo_dice_bce', **loss_kwargs)
        elif args.loss_function == 'soft_dice':
            # Soft Dice loss
            logger.info(f"[LOSS CONFIG] Using Soft Dice loss")
            loss_kwargs.update({'smooth': getattr(args, 'loss_smooth', 1e-5)})
            loss_function = create_advanced_loss_function('soft_dice', **loss_kwargs)
        elif args.loss_function == 'weighted_bce':
            # Weighted BCE loss
            logger.info(f"[LOSS CONFIG] Using Weighted BCE loss")
            loss_function = create_advanced_loss_function('weighted_bce', **loss_kwargs)
        elif args.loss_function == 'boundary':
            # Boundary loss
            logger.info(f"[LOSS CONFIG] Using Boundary loss")
            loss_function = create_advanced_loss_function('boundary', **loss_kwargs)
        elif args.loss_function == 'stable_bce':
            # Stable BCE loss
            logger.info(f"[LOSS CONFIG] Using Stable BCE loss")
            loss_function = create_advanced_loss_function('stable_bce', **loss_kwargs)
        elif args.loss_function == 'bce':
            # Pure BCE loss
            logger.info(f"[LOSS CONFIG] Using Binary Cross Entropy loss")
            loss_function = create_advanced_loss_function('bce', **loss_kwargs)
        else:
            # Default to dice loss for unknown selection
            logger.info(f"[LOSS CONFIG] Using default Dice loss (unknown loss function: {args.loss_function})")
            loss_kwargs.update({'sigmoid': True, 'smooth': getattr(args, 'loss_smooth', 1e-5)})
            loss_function = create_advanced_loss_function('dice', **loss_kwargs)
        
        # Create optimizer based on args.optimizer choice
        optimizer = create_optimizer(model, args)
        
        # Initialize dynamic learning rate scheduler
        lr_scheduler = DynamicLearningRateScheduler(
            initial_lr=args.learning_rate,
            model_id=args.model_id if hasattr(args, 'model_id') and args.model_id else None
        )
        logger.info(f"[LR_SCHEDULER] Dynamic learning rate scheduler initialized with LR: {args.learning_rate}")
        
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        iou_metric = MeanIoU(include_background=True, reduction="mean")
        scaler = torch.cuda.amp.GradScaler()
        
        # Initialize best metrics based on primary metric (not loss function)
        if args.primary_metric == 'iou':
            best_val_metric = -1  # Primary metric (IoU)
            best_val_dice = -1    # Secondary metric
            best_val_iou = -1     # Primary metric (same as best_val_metric)
            primary_metric_name = 'IoU'
        else:
            best_val_metric = -1  # Primary metric (Dice for other metrics)
            best_val_dice = -1    # Primary metric (same as best_val_metric)
            best_val_iou = -1     # Secondary metric
            primary_metric_name = 'Dice'
        
        best_model_path = None
        
        # Initialize Early Stopping if enabled
        early_stopping = None
        if getattr(args, 'use_early_stopping', False):
            # Determine mode based on metric
            mode = 'max' if args.early_stopping_metric in ['val_dice', 'val_accuracy'] else 'min'
            early_stopping = EarlyStopping(
                patience=args.early_stopping_patience,
                min_epochs=args.early_stopping_min_epochs,
                min_delta=args.early_stopping_min_delta,
                monitor_metric=args.early_stopping_metric,
                mode=mode,
                restore_best_weights=True,
                verbose=True
            )
            logger.info(f"[EARLY_STOPPING] Early stopping enabled - monitoring {args.early_stopping_metric}, patience: {args.early_stopping_patience}")
        else:
            logger.info("[EARLY_STOPPING] Early stopping disabled")
        
        epoch_history = []  # Track metrics for each epoch
        training_stopped_early = False  # Track if training was stopped by user
        
        # Set total number of batches per epoch for progress tracking
        if callback:
            callback.set_epoch_batches(len(train_loader))
            
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            logger.info(f"[EPOCH] Starting epoch {epoch+1}/{args.epochs}")
            
            # Update Celery task status if task ID is available
            if hasattr(args, 'celery_task_id') and args.celery_task_id:
                update_celery_task_status(
                    args.celery_task_id,
                    state='PROGRESS',
                    meta={
                        'status': 'training',
                        'current_epoch': epoch + 1,
                        'total_epochs': args.epochs,
                        'progress_percent': int((epoch / args.epochs) * 100),
                        'message': f'Training epoch {epoch+1}/{args.epochs}'
                    }
                )
            
            if epoch == 0:
                logger.info("[MODEL] Model Architecture Summary:")
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"[MODEL] Total parameters: {total_params:,}")
                logger.info(f"[MODEL] Trainable parameters: {trainable_params:,}")
                logger.info(f"[MODEL] Model size: {total_params * 4 / (1024**2):.2f} MB")
                
                # Log training configuration
                logger.info(f"[CONFIG] Batch size: {args.batch_size}")
                if 'train_ds' in locals() and 'val_ds' in locals():
                    logger.info(f"[CONFIG] Dataset: Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
                elif 'train_loader' in locals() and 'val_loader' in locals():
                    logger.info(f"[CONFIG] Dataset: Train samples: {train_samples}, Val samples: {val_samples}")
                else:
                    logger.info("[CONFIG] Dataset: Could not determine sample counts.")
                logger.info(f"[CONFIG] Optimizer: {type(optimizer).__name__}, Loss: DiceLoss, Device: {device}")
            
            # Check for stop_requested flag using callback system or global signal
            if STOP_TRAINING.is_set():
                logger.info("Global stop signal received. Exiting training loop.")
                training_stopped_early = True
                break
            elif callback and not callback.on_epoch_start(epoch, args.epochs):
                logger.info("[CALLBACK] Stop requested via callback. Exiting training loop.")
                training_stopped_early = True
                break
            elif hasattr(args, 'model_id') and args.model_id is not None and callback is None and DJANGO_AVAILABLE:
                # Fallback for stop checking if callback is not available and Django is available
                try:
                    # Try multiple import paths for Django model
                    try:
                        from core.apps.ml_manager.models import MLModel
                    except ImportError:
                        try:
                            from apps.ml_manager.models import MLModel
                        except ImportError:
                            from ml_manager.models import MLModel
                            
                    model_obj = MLModel.objects.get(pk=args.model_id)
                    if getattr(model_obj, 'stop_requested', False):
                        logger.info("Stop requested. Exiting training loop.")
                        training_stopped_early = True
                        break
                except Exception as e:
                    logger.warning(f"Could not check stop_requested flag: {e}")
            model.train()
            epoch_loss = 0
            train_dice = 0
            train_iou = 0
            
            batch_stopped_early = False
            for batch_idx, batch_data in enumerate(train_loader):
                # Check for global stop signal
                if STOP_TRAINING.is_set():
                    logger.info("Global stop signal received during batch. Exiting training loop.")
                    training_stopped_early = True
                    batch_stopped_early = True
                    break
                    
                # Call batch start callback
                if callback and not callback.on_batch_start(batch_idx, len(train_loader)):
                    logger.info("Stop requested during batch. Exiting training loop.")
                    training_stopped_early = True
                    batch_stopped_early = True
                    break
                    
                if isinstance(batch_data, dict):
                    inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                else:
                    raise TypeError(f"Unsupported batch_data type: {type(batch_data)}")
                # Debug tensor shapes on first batch
                if epoch == 0 and batch_idx == 0:
                    logger.info(f"Training batch shapes - Inputs: {inputs.shape}, Labels: {labels.shape}")
                    
                    # Enhanced range logging for first training batch
                    input_min, input_max = inputs.min().item(), inputs.max().item()
                    input_mean, input_std = inputs.mean().item(), inputs.std().item()
                    label_min, label_max = labels.min().item(), labels.max().item()
                    
                    # Handle both classification and segmentation labels
                    if labels.dtype in [torch.int64, torch.long, torch.int32, torch.int]:
                        # Classification labels - convert to float for statistics
                        label_mean = labels.float().mean().item()
                        label_std = labels.float().std().item()
                        is_classification = True
                    else:
                        # Segmentation labels - already float
                        label_mean, label_std = labels.mean().item(), labels.std().item()
                        is_classification = False
                    
                    logger.info(f"[TRAIN BATCH] 📊 COMPREHENSIVE DATA ANALYSIS:")
                    logger.info(f"[TRAIN BATCH]   🖼️  INPUT IMAGES:")
                    logger.info(f"[TRAIN BATCH]     Range: [{input_min:.4f}, {input_max:.4f}] (dtype: {inputs.dtype})")
                    logger.info(f"[TRAIN BATCH]     Stats: mean={input_mean:.4f}, std={input_std:.4f}")
                    
                    # Determine and log data range types for training images
                    if input_max <= 1.0 and input_min >= 0.0:
                        if input_mean < 0.1:
                            logger.info(f"[TRAIN BATCH]     ✅ Normalized [0-1] range (likely medical images with dark background)")
                        else:
                            logger.info(f"[TRAIN BATCH]     ✅ Normalized [0-1] range (standard normalization)")
                    elif input_max <= 255 and input_min >= 0:
                        if input_max == 255:
                            logger.info(f"[TRAIN BATCH]     ⚠️  [0-255] range detected (8-bit images, consider normalization)")
                        else:
                            logger.info(f"[TRAIN BATCH]     ⚠️  [0-{input_max:.0f}] range (partial 8-bit scale)")
                    elif input_max > 1000:
                        logger.info(f"[TRAIN BATCH]     🔍 High-value range [{input_min:.0f}-{input_max:.0f}] (DICOM/medical data?)")
                    else:
                        logger.info(f"[TRAIN BATCH]     ❓ Custom range [{input_min:.2f}-{input_max:.2f}]")
                    
                    logger.info(f"[TRAIN BATCH]   🎭 LABELS:")
                    logger.info(f"[TRAIN BATCH]     Range: [{label_min:.4f}, {label_max:.4f}] (dtype: {labels.dtype})")
                    logger.info(f"[TRAIN BATCH]     Stats: mean={label_mean:.4f}, std={label_std:.4f}")
                    
                    if is_classification:
                        # Classification labels
                        logger.info(f"[TRAIN BATCH]     Type: Classification labels")
                        unique_labels = torch.unique(labels)
                        unique_values_array = unique_labels.cpu().numpy()
                        logger.info(f"[TRAIN BATCH]     Classes: {unique_values_array}")
                        
                        # Enhanced class distribution with names for artery classification
                        if class_info and class_info.get('task_type') == 'artery_classification':
                            class_names = class_info.get('class_names', ['Right Artery', 'Left Artery'])
                            logger.info(f"[TRAIN BATCH]     🫀 ARTERY CLASSIFICATION STATISTICS:")
                            for class_idx in unique_values_array:
                                class_count = (labels == class_idx).sum().item()
                                class_ratio = class_count / labels.numel()
                                class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
                                logger.info(f"[TRAIN BATCH]       {class_name} (class {class_idx}): {class_count} samples ({class_ratio:.2%})")
                        else:
                            # Generic classification statistics
                            for class_idx in unique_values_array:
                                class_count = (labels == class_idx).sum().item()
                                class_ratio = class_count / labels.numel()
                                logger.info(f"[TRAIN BATCH]     Class {class_idx}: {class_count} samples ({class_ratio:.2%})")
                    else:
                        # Segmentation labels
                        logger.info(f"[TRAIN BATCH]     Type: Segmentation masks")
                        
                        # Determine and log data range types for training masks
                        if label_max <= 1.0 and label_min >= 0.0:
                            logger.info(f"[TRAIN BATCH]     ✅ Binary normalized [0-1] range")
                        elif label_max <= 255 and label_min >= 0:
                            logger.info(f"[TRAIN BATCH]     ⚠️  [0-255] range (needs binary normalization)")
                        else:
                            logger.info(f"[TRAIN BATCH]     ❓ Custom range [{label_min:.2f}-{label_max:.2f}]")
                        
                        # Check for unique values in training masks with intelligent binary detection
                        unique_train_labels = torch.unique(labels)
                        unique_values_array = unique_train_labels.cpu().numpy()
                        # logger.info(f"[TRAIN BATCH]     Raw unique values: {unique_values_array}")
                        
                        # Intelligent binary segmentation detection
                        if len(unique_train_labels) == 2:
                            # Check if it's standard binary (0,1) or 8-bit binary (0,255)
                            min_val, max_val = unique_values_array.min(), unique_values_array.max()
                            if (min_val == 0 and max_val == 1):
                                logger.info(f"[TRAIN BATCH]     ✅ Binary segmentation confirmed (0,1 format)")
                                positive_ratio = (labels == 1).float().mean().item()
                            elif (min_val == 0 and max_val == 255):
                                logger.info(f"[TRAIN BATCH]     ✅ Binary segmentation confirmed (0,255 format - will be normalized)")
                                positive_ratio = (labels == 255).float().mean().item()
                            else:
                                logger.info(f"[TRAIN BATCH]     ✅ Binary segmentation confirmed (custom {min_val},{max_val} format)")
                                positive_ratio = (labels == max_val).float().mean().item()
                            logger.info(f"[TRAIN BATCH]     📈 Class distribution: {positive_ratio:.2%} positive, {1-positive_ratio:.2%} background")
                        elif len(unique_train_labels) == 1:
                            single_val = unique_values_array[0]
                            if single_val == 0:
                                logger.info(f"[TRAIN BATCH]     ⚠️  Single class detected (all background) - check data!")
                            elif single_val == 1 or single_val == 255:
                                logger.info(f"[TRAIN BATCH]     ⚠️  Single class detected (all foreground) - check data!")
                            else:
                                logger.info(f"[TRAIN BATCH]     ⚠️  Single class detected (all {single_val}) - check data!")
                        else:
                            # Check if it's grayscale values that need thresholding
                            if len(unique_train_labels) > 2:
                                # Check if it's normalized grayscale (many values between 0-1) or 8-bit grayscale (0-255)
                                if label_max <= 1.0 and label_min >= 0 and len(unique_train_labels) >= 50:
                                    # Many values in [0-1] range = normalized grayscale masks
                                    logger.info(f"[TRAIN BATCH]     📝 Normalized grayscale mask detected ({len(unique_train_labels)} values)")
                                    logger.info(f"[TRAIN BATCH]     💡 Recommend thresholding: values > 0.5 → 1, else → 0")
                                    # Show sample distribution for debugging
                                    sample_values = sorted(unique_values_array)
                                    logger.info(f"[TRAIN BATCH]     🔍 Sample values: {sample_values[:5]}...{sample_values[-5:]}")
                                elif label_max <= 255 and label_min >= 0:
                                    # Values in [0-255] range = 8-bit grayscale masks
                                    logger.info(f"[TRAIN BATCH]     📝 8-bit grayscale mask detected ({len(unique_train_labels)} values)")
                                    logger.info(f"[TRAIN BATCH]     💡 Recommend thresholding: values > 127 → 1, else → 0")
                                    if len(unique_train_labels) <= 10:
                                        logger.info(f"[TRAIN BATCH]     🔍 All values: {sorted(unique_values_array)}")
                                else:
                                    logger.info(f"[TRAIN BATCH]     ⚠️  Multi-class ({len(unique_train_labels)} classes) - not binary segmentation")
                            else:
                                logger.info(f"[TRAIN BATCH]     ⚠️  Multi-class ({len(unique_train_labels)} classes) - not binary segmentation")
                    
                    # Additional data quality checks
                    nan_inputs = torch.isnan(inputs).sum().item()
                    inf_inputs = torch.isinf(inputs).sum().item()
                    nan_labels = torch.isnan(labels).sum().item()
                    inf_labels = torch.isinf(labels).sum().item()
                    
                    if nan_inputs > 0 or inf_inputs > 0:
                        logger.warning(f"[TRAIN BATCH]     ❌ Data quality issues - NaN: {nan_inputs}, Inf: {inf_inputs} in inputs")
                    if nan_labels > 0 or inf_labels > 0:
                        logger.warning(f"[TRAIN BATCH]     ❌ Data quality issues - NaN: {nan_labels}, Inf: {inf_labels} in labels")
                    if nan_inputs == 0 and inf_inputs == 0 and nan_labels == 0 and inf_labels == 0:
                        logger.info(f"[TRAIN BATCH]     ✅ Data quality check passed (no NaN/Inf values)")
                
                # Handle label preprocessing based on task type
                is_classification_task = class_info and class_info.get('class_type') == 'classification'
                
                if is_classification_task:
                    # Classification task - labels should remain as Long (int64)
                    labels = labels.long()
                    if epoch == 0 and batch_idx == 0:
                        logger.info(f"[TRAIN BATCH] 🎯 Classification task detected - labels kept as Long type")
                        logger.info(f"[TRAIN BATCH]     Label shape: {labels.shape}, dtype: {labels.dtype}")
                        logger.info(f"[TRAIN BATCH]     Label range: [{labels.min().item()}, {labels.max().item()}]")
                else:
                    # Segmentation task - convert to float and apply normalization
                    labels = labels.float()
                    
                    # Auto-normalize and threshold masks if they're not binary
                    if labels.max() > 1:
                        if labels.max() <= 255 and labels.min() >= 0:
                            logger.info(f"[TRAIN BATCH] 🔧 Auto-normalizing masks from [0-255] to [0-1] range")
                            labels = labels / 255.0
                            # Ensure binary values after normalization
                            labels = (labels > 0.5).float()
                            logger.info(f"[TRAIN BATCH] ✅ Masks normalized to range [{labels.min().item():.1f}-{labels.max().item():.1f}]")
                        else:
                            logger.warning(f"[TRAIN BATCH] ❌ Labels out of expected range! min: {labels.min().item()}, max: {labels.max().item()}")
                            # Try to normalize anyway for custom ranges
                            labels_max = labels.max()
                            if labels_max > 0:
                                labels = labels / labels_max
                                labels = (labels > 0.5).float()
                                logger.info(f"[TRAIN BATCH] 🔧 Normalized custom range to binary [0-1]")
                    elif labels.max() <= 1 and labels.min() >= 0:
                        # Check if it's grayscale masks that need thresholding (many values between 0-1)
                        unique_for_threshold = torch.unique(labels)
                        if len(unique_for_threshold) > 10:  # More than 10 unique values = grayscale
                            logger.info(f"[TRAIN BATCH] 🔧 Auto-thresholding grayscale masks ({len(unique_for_threshold)} values → binary)")
                            labels = (labels > 0.5).float()
                            final_unique = torch.unique(labels)
                            logger.info(f"[TRAIN BATCH] ✅ Thresholded to {len(final_unique)} values: {final_unique.cpu().numpy()}")
                
                # Additional validation for segmentation tasks only
                if not is_classification_task:
                    if labels.max() > 1 or labels.min() < 0:
                        logger.warning(f"[TRAIN BATCH] ❌ Labels still out of [0,1] range after normalization! min: {labels.min().item()}, max: {labels.max().item()}")
                
                # Verify normalization worked correctly for first batch
                if epoch == 0 and batch_idx == 0:
                    # Re-check unique values after normalization
                    normalized_unique_labels = torch.unique(labels)
                    logger.info(f"[TRAIN BATCH] 🔍 POST-NORMALIZATION VERIFICATION:")
                    logger.info(f"[TRAIN BATCH]     Final unique values: {normalized_unique_labels.cpu().numpy()}")
                    logger.info(f"[TRAIN BATCH]     Final range: [{labels.min().item():.4f}, {labels.max().item():.4f}]")
                    
                    if len(normalized_unique_labels) == 2 and labels.min() >= 0 and labels.max() <= 1:
                        logger.info(f"[TRAIN BATCH]     ✅ Successfully normalized to binary [0-1] format")
                        positive_ratio_final = (labels == 1).float().mean().item()
                        logger.info(f"[TRAIN BATCH]     📈 Final class distribution: {positive_ratio_final:.2%} positive, {1-positive_ratio_final:.2%} background")
                    else:
                        logger.warning(f"[TRAIN BATCH]     ⚠️  Normalization may have failed - {len(normalized_unique_labels)} unique values")
                
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    if epoch == 0 and batch_idx == 0:
                        logger.info(f"Model output shape: {outputs.shape}")
                    loss = loss_function(outputs, labels)
                
                if loss.item() < 0:
                    logger.warning(f"Negative loss detected! outputs min/max: {outputs.min().item()}/{outputs.max().item()}, labels min/max: {labels.min().item()}/{labels.max().item()}")
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                
                # Calculate training metrics
                with torch.no_grad():
                    if class_info and class_info.get('task_type') == 'artery_classification':
                        # Classification task - calculate accuracy (no dice metric needed)
                        predicted_classes = torch.argmax(outputs, dim=1)
                        batch_accuracy = (predicted_classes == labels).float().mean().item()
                        batch_dice = batch_accuracy  # Use accuracy as the metric
                    else:
                        # Segmentation task - calculate dice and iou scores
                        # Apply proper thresholding for binary segmentation
                        if outputs.shape[1] == 1:
                            val_outputs_soft = torch.sigmoid(outputs)
                            threshold = args.threshold if args.threshold is not None else 0.5
                            val_outputs_hard = (val_outputs_soft > threshold).float()
                            dice_metric(y_pred=val_outputs_hard, y=labels)
                            iou_metric(y_pred=val_outputs_hard, y=labels)
                        else:
                            # Multi-class - apply softmax and argmax
                            val_outputs = torch.softmax(outputs, dim=1)
                            val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True).float()
                            dice_metric(y_pred=val_outputs, y=labels)
                            iou_metric(y_pred=val_outputs, y=labels)
                        
                        batch_dice = dice_metric.aggregate().item()
                        batch_iou = iou_metric.aggregate().item()
                        dice_metric.reset()
                        iou_metric.reset()
                    train_dice += batch_dice
                    train_iou += batch_iou if 'batch_iou' in locals() else batch_dice
                
                # Call batch end callback with current metrics
                if callback:
                    # Use appropriate metric name for callback
                    metric_name = 'train_accuracy' if class_info and class_info.get('task_type') == 'artery_classification' else 'train_dice'
                    batch_logs = {
                        'train_loss': loss.item(),
                        metric_name: batch_dice,
                        'batch_progress': (batch_idx + 1) / len(train_loader)
                    }
                    callback.on_batch_end(batch_idx, batch_logs)
                
                if batch_idx % 10 == 0:  # More frequent logging for GUI
                    progress_pct = (batch_idx / len(train_loader)) * 100
                    # Use appropriate metric name in logging
                    metric_display = "Accuracy" if class_info and class_info.get('task_type') == 'artery_classification' else "Dice"
                    logger.info(f"[TRAIN] Epoch {epoch+1}/{args.epochs} - "
                              f"Batch {batch_idx}/{len(train_loader)} ({progress_pct:.1f}%) - "
                              f"Loss: {loss.item():.4f}, {metric_display}: {batch_dice:.4f}")
                              
            # Check if training was stopped during batch processing
            if batch_stopped_early:
                logger.info("Training stopped during batch processing.")
                break
            
            if callback and callback.model.stop_requested:
                logger.info("Training stopped during batch processing.")
                training_stopped_early = True
                break
            
            epoch_loss /= len(train_loader)
            train_dice /= len(train_loader)
            train_iou /= len(train_loader)
            
            # Validation
            logger.info(f"[VAL] Starting validation for epoch {epoch+1}")
            model.eval()
            val_loss = 0
            val_dice = 0
            val_iou = 0
            
            # --- Enhanced validation data analysis ---
            try:
                val_batch = next(iter(val_loader))
                if isinstance(val_batch, dict):
                    val_images = val_batch["image"]
                    val_labels = val_batch["label"]
                elif isinstance(val_batch, (list, tuple)) and len(val_batch) == 2:
                    val_images, val_labels = val_batch
                else:
                    val_images, val_labels = None, None
                    
                # Only log validation data analysis on first epoch
                if epoch == 0 and val_images is not None and val_labels is not None:
                    logger.info(f"[VAL DATA] Validation batch: images shape: {val_images.shape}, masks shape: {val_labels.shape}")
                    
                    # Enhanced validation data range logging
                    val_img_min, val_img_max = val_images.min().item(), val_images.max().item()
                    val_img_mean, val_img_std = val_images.mean().item(), val_images.std().item()
                    val_label_min, val_label_max = val_labels.min().item(), val_labels.max().item()
                    val_label_mean, val_label_std = val_labels.mean().item(), val_labels.std().item()
                    
                    logger.info(f"[VAL DATA] 📊 VALIDATION DATA ANALYSIS:")
                    logger.info(f"[VAL DATA]   🖼️  VALIDATION IMAGES:")
                    logger.info(f"[VAL DATA]     Range: [{val_img_min:.4f}, {val_img_max:.4f}] (dtype: {val_images.dtype})")
                    logger.info(f"[VAL DATA]     Stats: mean={val_img_mean:.4f}, std={val_img_std:.4f}")
                    
                    # Determine and log data range types for validation images
                    if val_img_max <= 1.0 and val_img_min >= 0.0:
                        if val_img_mean < 0.1:
                            logger.info(f"[VAL DATA]     ✅ Normalized [0-1] range (likely medical images with dark background)")
                        else:
                            logger.info(f"[VAL DATA]     ✅ Normalized [0-1] range (standard normalization)")
                    elif val_img_max <= 255 and val_img_min >= 0:
                        if val_img_max == 255:
                            logger.info(f"[VAL DATA]     ⚠️  [0-255] range detected (8-bit images, consider normalization)")
                        else:
                            logger.info(f"[VAL DATA]     ⚠️  [0-{val_img_max:.0f}] range (partial 8-bit scale)")
                    elif val_img_max > 1000:
                        logger.info(f"[VAL DATA]     🔍 High-value range [{val_img_min:.0f}-{val_img_max:.0f}] (DICOM/medical data?)")
                    else:
                        logger.info(f"[VAL DATA]     ❓ Custom range [{val_img_min:.2f}-{val_img_max:.2f}]")
                    
                    logger.info(f"[VAL DATA]   🎭 VALIDATION MASKS:")
                    logger.info(f"[VAL DATA]     Range: [{val_label_min:.4f}, {val_label_max:.4f}] (dtype: {val_labels.dtype})")
                    logger.info(f"[VAL DATA]     Stats: mean={val_label_mean:.4f}, std={val_label_std:.4f}")
                    
                    # Determine and log data range types for validation masks
                    if val_label_max <= 1.0 and val_label_min >= 0.0:
                        logger.info(f"[VAL DATA]     ✅ Binary normalized [0-1] range")
                    elif val_label_max <= 255 and val_label_min >= 0:
                        logger.info(f"[VAL DATA]     ⚠️  [0-255] range (needs binary normalization)")
                    else:
                        logger.info(f"[VAL DATA]     ❓ Custom range [{val_label_min:.2f}-{val_label_max:.2f}]")
                    
                    # Check for unique values in validation masks with intelligent binary detection
                    unique_val_labels = torch.unique(val_labels)
                    unique_val_values_array = unique_val_labels.cpu().numpy()
                    # logger.info(f"[VAL DATA]     Raw unique values: {unique_val_values_array}")
                    
                    # Intelligent binary segmentation detection for validation
                    if len(unique_val_labels) == 2:
                        # Check if it's standard binary (0,1) or 8-bit binary (0,255)
                        min_val, max_val = unique_val_values_array.min(), unique_val_values_array.max()
                        if (min_val == 0 and max_val == 1):
                            logger.info(f"[VAL DATA]     ✅ Binary segmentation confirmed (0,1 format)")
                            val_positive_ratio = (val_labels == 1).float().mean().item()
                        elif (min_val == 0 and max_val == 255):
                            logger.info(f"[VAL DATA]     ✅ Binary segmentation confirmed (0,255 format - will be normalized)")
                            val_positive_ratio = (val_labels == 255).float().mean().item()
                        else:
                            logger.info(f"[VAL DATA]     ✅ Binary segmentation confirmed (custom {min_val},{max_val} format)")
                            val_positive_ratio = (val_labels == max_val).float().mean().item()
                        logger.info(f"[VAL DATA]     📈 Class distribution: {val_positive_ratio:.2%} positive, {1-val_positive_ratio:.2%} background")
                    elif len(unique_val_labels) == 1:
                        single_val = unique_val_values_array[0]
                        if single_val == 0:
                            logger.info(f"[VAL DATA]     ⚠️  Single class detected (all background) - check data!")
                        elif single_val == 1 or single_val == 255:
                            logger.info(f"[VAL DATA]     ⚠️  Single class detected (all foreground) - check data!")
                        else:
                            logger.info(f"[VAL DATA]     ⚠️  Single class detected (all {single_val}) - check data!")
                    else:
                        # Check if it's grayscale values that need thresholding
                        if len(unique_val_labels) > 2:
                            # Check if it's normalized grayscale (many values between 0-1) or 8-bit grayscale (0-255)
                            if val_label_max <= 1.0 and val_label_min >= 0 and len(unique_val_labels) >= 50:
                                # Many values in [0-1] range = normalized grayscale masks
                                logger.info(f"[VAL DATA]     📝 Normalized grayscale mask detected ({len(unique_val_labels)} values)")
                                logger.info(f"[VAL DATA]     💡 Recommend thresholding: values > 0.5 → 1, else → 0")
                                # Show sample distribution for debugging
                                sample_values = sorted(unique_val_values_array)
                                logger.info(f"[VAL DATA]     🔍 Sample values: {sample_values[:5]}...{sample_values[-5:]}")
                            elif val_label_max <= 255 and val_label_min >= 0:
                                # Values in [0-255] range = 8-bit grayscale masks
                                logger.info(f"[VAL DATA]     📝 8-bit grayscale mask detected ({len(unique_val_labels)} values)")
                                logger.info(f"[VAL DATA]     💡 Recommend thresholding: values > 127 → 1, else → 0")
                                if len(unique_val_labels) <= 10:
                                    logger.info(f"[VAL DATA]     🔍 All values: {sorted(unique_val_values_array)}")
                            else:
                                logger.info(f"[VAL DATA]     ⚠️  Multi-class ({len(unique_val_labels)} classes) - not binary segmentation")
                        else:
                            logger.info(f"[VAL DATA]     ⚠️  Multi-class ({len(unique_val_labels)} classes) - not binary segmentation")
                    
                    # Additional validation data quality checks
                    val_nan_inputs = torch.isnan(val_images).sum().item()
                    val_inf_inputs = torch.isinf(val_images).sum().item()
                    val_nan_labels = torch.isnan(val_labels).sum().item()
                    val_inf_labels = torch.isinf(val_labels).sum().item()
                    
                    if val_nan_inputs > 0 or val_inf_inputs > 0:
                        logger.warning(f"[VAL DATA]     ❌ Data quality issues - NaN: {val_nan_inputs}, Inf: {val_inf_inputs} in inputs")
                    if val_nan_labels > 0 or val_inf_labels > 0:
                        logger.warning(f"[VAL DATA]     ❌ Data quality issues - NaN: {val_nan_labels}, Inf: {val_inf_labels} in labels")
                    if val_nan_inputs == 0 and val_inf_inputs == 0 and val_nan_labels == 0 and val_inf_labels == 0:
                        logger.info(f"[VAL DATA]     ✅ Data quality check passed (no NaN/Inf values)")
                
                # Always generate visualization (moved after conditional logging)
                if val_images is not None and val_labels is not None:
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(2, min(4, val_images.shape[0]), figsize=(12, 6))
                    for i in range(min(4, val_images.shape[0])):
                        axes[0, i].imshow(val_images[i, 0].cpu().numpy(), cmap='gray')
                        axes[0, i].set_title(f'Val input #{i+1}')
                        axes[0, i].axis('off')
                        axes[1, i].imshow(val_labels[i, 0].cpu().numpy(), cmap='gray')
                        axes[1, i].set_title(f'Val mask #{i+1}')
                        axes[1, i].axis('off')
                    plt.tight_layout()
                    vis_dir = os.path.join(model_dir, 'artifacts')
                    os.makedirs(vis_dir, exist_ok=True)
                    vis_path = os.path.join(vis_dir, 'val_sample_inputs_and_masks.png')
                    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    logger.info(f"[VAL DATA] Zapisano wizualizację walidacyjnych wejść/masek: {vis_path}")
            except Exception as e:
                logger.warning(f"[VAL DATA] Nie udało się zwizualizować batcha walidacyjnego: {e}")

            with torch.no_grad():
                for val_idx, val_data in enumerate(val_loader):
                    if isinstance(val_data, dict):
                        val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    elif isinstance(val_data, (list, tuple)) and len(val_data) == 2:
                        val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
                    else:
                        raise TypeError(f"Unsupported val_data type: {type(val_data)}")
                    
                    # Apply same normalization and thresholding to validation labels as training
                    # Handle label preprocessing based on task type (same as training loop)
                    is_classification_task = class_info and class_info.get('class_type') == 'classification'
                    
                    if is_classification_task:
                        # Classification task - labels should remain as Long (int64)
                        val_labels = val_labels.long()
                    else:
                        # Segmentation task - convert to float and apply normalization
                        val_labels = val_labels.float()
                        if val_labels.max() > 1:
                            if val_labels.max() <= 255 and val_labels.min() >= 0:
                                val_labels = val_labels / 255.0
                                val_labels = (val_labels > 0.5).float()
                            else:
                                # Normalize custom ranges
                                labels_max = val_labels.max()
                                if labels_max > 0:
                                    val_labels = val_labels / labels_max
                                    val_labels = (val_labels > 0.5).float()
                        elif val_labels.max() <= 1 and val_labels.min() >= 0:
                            # Check if it's grayscale masks that need thresholding
                            unique_for_threshold = torch.unique(val_labels)
                            if len(unique_for_threshold) > 10:  # More than 10 unique values = grayscale
                                val_labels = (val_labels > 0.5).float()
                    
                    val_outputs = model(val_inputs)
                    batch_val_loss = loss_function(val_outputs, val_labels).item()
                    val_loss += batch_val_loss
                    
                    # Apply appropriate post-processing and calculate metrics based on task type
                    if class_info and class_info.get('task_type') == 'artery_classification':
                        # Classification task - calculate accuracy directly (no dice metric)
                        predicted_classes = torch.argmax(val_outputs, dim=1)
                        batch_accuracy = (predicted_classes == val_labels).float().mean().item()
                        
                        # Accumulate accuracy values for averaging later
                        if 'val_accuracies' not in locals():
                            val_accuracies = []
                        val_accuracies.append(batch_accuracy)
                    else:
                        # Segmentation task - apply appropriate post-processing
                        num_output_channels = val_outputs.shape[1]
                        if num_output_channels == 1:
                            # Binary segmentation - apply sigmoid then threshold
                            val_outputs_soft = torch.sigmoid(val_outputs)
                            threshold = args.threshold if args.threshold is not None else 0.5
                            val_outputs_hard = (val_outputs_soft > threshold).float()
                            # Use hard predictions for metric calculation
                            val_outputs = val_outputs_hard
                        else:
                            # Multi-class semantic segmentation
                            val_outputs = torch.softmax(val_outputs, dim=1)
                            val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True).float()
                        
                        dice_metric(y_pred=val_outputs, y=val_labels)
                        iou_metric(y_pred=val_outputs, y=val_labels)
                    if val_idx % 5 == 0:
                        val_progress_pct = (val_idx / len(val_loader)) * 100
                        logger.info(f"[VAL] Batch {val_idx}/{len(val_loader)} ({val_progress_pct:.1f}%) - Loss: {batch_val_loss:.4f}")
                
                val_loss /= len(val_loader)
                
                # Calculate final validation metric based on task type
                if class_info and class_info.get('task_type') == 'artery_classification':
                    # For classification, use average accuracy (stored in val_accuracies)
                    val_dice = sum(val_accuracies) / len(val_accuracies) if 'val_accuracies' in locals() and val_accuracies else 0.0
                    val_iou = val_dice  # For classification, IoU equals accuracy
                else:
                    # For segmentation, use dice and iou metrics
                    val_dice = dice_metric.aggregate().item()
                    val_iou = iou_metric.aggregate().item()
                    dice_metric.reset()
                    iou_metric.reset()
            
            # Create metrics dictionary with appropriate names based on task type and loss function
            if class_info and class_info.get('task_type') == 'artery_classification':
                metrics = {
                    "train_loss": epoch_loss,
                    "train_accuracy": train_dice,  # train_dice contains accuracy for classification
                    "val_loss": val_loss,
                    "val_accuracy": val_dice,      # val_dice contains accuracy for classification
                }
                train_metric_name = "Train Accuracy"
                val_metric_name = "Val Accuracy"
                best_metric_name = "Best Val Accuracy"
            else:
                # Determine metric names based on primary metric (not loss function)
                if args.primary_metric == 'iou':
                    train_metric_name = "Training IoU"
                    val_metric_name = "Validation IoU"
                    best_metric_name = "Best Val IoU"
                    # For IoU primary metric, IoU is the main evaluation metric
                    metrics = {
                        "train_loss": epoch_loss,
                        "train_iou": train_iou,    # Primary metric for IoU
                        "train_dice": train_dice,  # Secondary metric
                        "val_loss": val_loss,
                        "val_iou": val_iou,        # Primary metric for IoU
                        "val_dice": val_dice,      # Secondary metric
                    }
                    # Use IoU as the primary metric value for logging
                    primary_train_metric = train_iou
                    primary_val_metric = val_iou
                else:
                    # Default to Dice for all other primary metrics
                    train_metric_name = "Training Dice"
                    val_metric_name = "Validation Dice"
                    best_metric_name = "Best Val Dice"
                    metrics = {
                        "train_loss": epoch_loss,
                        "train_dice": train_dice,  # Primary metric for Dice
                        "train_iou": train_iou,    # Secondary metric
                        "val_loss": val_loss,
                        "val_dice": val_dice,      # Primary metric for Dice
                        "val_iou": val_iou,        # Secondary metric
                    }
                    # Use Dice as the primary metric value for logging
                    primary_train_metric = train_dice
                    primary_val_metric = val_dice
            
            # Create metadata about primary metric for callback (separate from MLflow metrics)
            metrics_metadata = {
                'primary_metric_name': primary_metric_name,
                'primary_train_metric': primary_train_metric,
                'primary_val_metric': primary_val_metric,
                'train_metric_name': train_metric_name,
                'val_metric_name': val_metric_name,
                'loss_function': args.loss_function,
                'primary_metric_type': args.primary_metric,
            }
            
            # Add numeric metadata to metrics for MLflow (only numeric values)
            metrics.update({
                'primary_train_metric': primary_train_metric,
                'primary_val_metric': primary_val_metric,
            })
            
            logger.info(f"[EPOCH] {epoch+1}/{args.epochs} COMPLETED - "
                       f"Train Loss: {epoch_loss:.4f}, {train_metric_name}: {primary_train_metric:.4f}, "
                       f"Val Loss: {val_loss:.4f}, {val_metric_name}: {primary_val_metric:.4f}")
            
            # Log learning rate and other training details
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"[METRICS] Learning Rate: {current_lr:.6f}, "
                       f"{primary_metric_name}: {best_val_metric:.4f}")
            
            # Log batch statistics
            logger.info(f"[STATS] Total batches processed: {len(train_loader)} train, {len(val_loader)} val")
            
            # Ensure all metrics are numeric before logging to MLflow
            numeric_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_metrics[key] = value
                else:
                    logger.warning(f"[MLFLOW] Skipping non-numeric metric: {key}={value} (type: {type(value)})")
            
            mlflow.log_metrics(numeric_metrics, step=epoch+1)  # Use 1-based epoch numbering
            
            # Enhanced MLflow synchronization - log additional training state and system metrics
            try:
                # 1. Training progress and state
                mlflow.log_metric('training_progress_percent', ((epoch + 1) / args.epochs) * 100, step=epoch+1)
                mlflow.log_metric('current_epoch', epoch + 1, step=epoch+1)
                mlflow.log_metric('total_epochs', args.epochs, step=epoch+1)
                mlflow.log_metric('learning_rate', current_lr, step=epoch+1)
                
                # 2. System metrics using SystemMonitor for comprehensive metrics
                if system_monitor and system_monitor.enabled:
                    try:
                        system_metrics = system_monitor.get_system_metrics()
                        if system_metrics:
                            logger.info(f"[SYSTEM_METRICS] Logging {len(system_metrics)} system metrics to MLflow")
                            for metric_name, metric_value in system_metrics.items():
                                mlflow.log_metric(metric_name, metric_value, step=epoch+1)
                        else:
                            logger.warning("[SYSTEM_METRICS] No system metrics returned from SystemMonitor")
                    except Exception as e:
                        logger.warning(f"[SYSTEM_METRICS] Failed to log system metrics: {e}")
                        # Fallback to basic system metrics
                        try:
                            import psutil
                            cpu_percent = psutil.cpu_percent()
                            memory = psutil.virtual_memory()
                            disk = psutil.disk_usage('.')
                            
                            mlflow.log_metric('system_cpu_percent', cpu_percent, step=epoch+1)
                            mlflow.log_metric('system_memory_percent', memory.percent, step=epoch+1)
                            mlflow.log_metric('system_memory_used_gb', memory.used / (1024**3), step=epoch+1)
                            mlflow.log_metric('system_disk_percent', disk.percent, step=epoch+1)
                            logger.info("[SYSTEM_METRICS] Used fallback basic system metrics")
                        except Exception as fallback_e:
                            logger.warning(f"[SYSTEM_METRICS] Failed to log even basic system metrics: {fallback_e}")
                else:
                    # Fallback to basic system metrics when system_monitor is not available
                    try:
                        import psutil
                        cpu_percent = psutil.cpu_percent()
                        memory = psutil.virtual_memory()
                        disk = psutil.disk_usage('.')
                        
                        mlflow.log_metric('system_cpu_percent', cpu_percent, step=epoch+1)
                        mlflow.log_metric('system_memory_percent', memory.percent, step=epoch+1)
                        mlflow.log_metric('system_memory_used_gb', memory.used / (1024**3), step=epoch+1)
                        mlflow.log_metric('system_disk_percent', disk.percent, step=epoch+1)
                        logger.info("[SYSTEM_METRICS] Used basic system metrics (SystemMonitor not available)")
                    except Exception as e:
                        logger.warning(f"[SYSTEM_METRICS] Failed to log basic system metrics: {e}")
                
                # 3. Model metrics
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                mlflow.log_metric('model_total_params', total_params, step=epoch+1)
                mlflow.log_metric('model_trainable_params', trainable_params, step=epoch+1)
                
                # 4. Training state tags
                mlflow.set_tag('training_status', 'in_progress')
                mlflow.set_tag('current_epoch', f"{epoch + 1}/{args.epochs}")
                mlflow.set_tag('training_progress', f"{((epoch + 1) / args.epochs) * 100:.1f}%")
                mlflow.set_tag(f'best_val_{primary_metric_name.lower()}', f"{best_val_metric:.4f}")
                
                # 5. Time metrics
                epoch_duration = time.time() - epoch_start_time
                mlflow.log_metric('epoch_duration_seconds', epoch_duration, step=epoch+1)
                
                if epoch == 0:
                    training_start_time = epoch_start_time  # Store start time
                
                if 'training_start_time' in locals():
                    total_training_time = time.time() - training_start_time
                    mlflow.log_metric('total_training_time_seconds', total_training_time, step=epoch+1)
                    mlflow.log_metric('avg_epoch_duration', total_training_time / (epoch + 1), step=epoch+1)
                
                # 6. Dataset metrics (log once)
                if epoch == 0:
                    if 'train_ds' in locals() and 'val_ds' in locals():
                        mlflow.log_metric('train_samples', len(train_ds))
                        mlflow.log_metric('val_samples', len(val_ds))
                        mlflow.log_metric('total_samples', len(train_ds) + len(val_ds))
                    elif 'train_loader' in locals() and 'val_loader' in locals():
                        mlflow.log_metric('train_batches', len(train_loader))
                        mlflow.log_metric('val_batches', len(val_loader))
                        if 'train_samples' in locals() and 'val_samples' in locals():
                            mlflow.log_metric('train_samples', train_samples)
                            mlflow.log_metric('val_samples', val_samples)
                            mlflow.log_metric('total_samples', train_samples + val_samples)
                
                logger.info(f"[MLFLOW] Synchronized metrics and state for epoch {epoch+1}")
                
            except Exception as mlflow_error:
                logger.warning(f"[MLFLOW] Failed to sync additional metrics for epoch {epoch+1}: {mlflow_error}")
            
            epoch_history.append(metrics)  # Save metrics for this epoch
            
            # Call epoch end callback with metrics (including metadata)
            if callback:
                logger.info(f"[CALLBACK] Calling callback.on_epoch_end for epoch {epoch}")
                # Combine numeric metrics with metadata for callback
                callback_metrics = metrics.copy()
                callback_metrics.update(metrics_metadata)
                callback.on_epoch_end(epoch, callback_metrics)
                
                # Check if sample generation was triggered by callback
                try:
                    # Always generate samples for the first and last epochs, and every few epochs
                    should_generate = (epoch == 0 or epoch == args.epochs - 1 or (epoch + 1) % max(1, args.epochs // 5) == 0)
                    
                    # Also check if callback specifically requested generation
                    try:
                        # Try multiple import paths for Django model
                        try:
                            from core.apps.ml_manager.models import MLModel
                        except ImportError:
                            try:
                                from apps.ml_manager.models import MLModel
                            except ImportError:
                                from ml_manager.models import MLModel
                                
                        model_obj = MLModel.objects.get(id=callback.model_id)
                        callback_requested = (hasattr(model_obj, 'training_data_info') and 
                                            model_obj.training_data_info and 
                                            model_obj.training_data_info.get('generate_samples_epoch') == epoch + 1)
                    except (ImportError, Exception) as e:
                        logger.warning(f"[SAMPLE_GENERATION] Could not check callback request: {e}")
                        callback_requested = False
                    
                    if should_generate or callback_requested:
                        logger.info(f"[SAMPLE_GENERATION] Generating training samples for epoch {epoch+1} (auto: {should_generate}, callback: {callback_requested})")
                        
                        # Create predictions directory if it doesn't exist
                        pred_dir = os.path.join(model_dir, f'predictions/epoch_{epoch+1:03d}')
                        os.makedirs(pred_dir, exist_ok=True)
                        logger.info(f"[SAMPLE_GENERATION] Created predictions directory: {pred_dir}")
                        
                        pred_file = save_sample_predictions(
                            model, val_loader, device, epoch, 
                            model_dir=model_dir, 
                            class_info=class_info if 'class_info' in locals() else None, 
                            threshold=threshold if 'threshold' in locals() else 0.5
                        )
                        
                        if pred_file and os.path.exists(pred_file):
                            # Log prediction samples to MLflow and model directory
                            log_artifact_to_model_directory(pred_file, model_dir, f"predictions/epoch_{epoch+1:03d}")
                            logger.info(f"[SAMPLE_GENERATION] Successfully generated and logged samples for epoch {epoch+1}: {pred_file}")
                            
                            # Clear the generation flag if it was callback-requested
                            if callback_requested:
                                model_obj.training_data_info['generate_samples_epoch'] = None
                                model_obj.save(update_fields=['training_data_info'])
                        else:
                            logger.warning(f"[SAMPLE_GENERATION] Failed to generate samples for epoch {epoch+1}")
                            
                except Exception as sample_error:
                    logger.warning(f"[SAMPLE_GENERATION] Error generating samples for epoch {epoch+1}: {sample_error}")
                    import traceback
                    logger.error(f"[SAMPLE_GENERATION] Traceback: {traceback.format_exc()}")
                    
            else:
                logger.info(f"[CALLBACK] No callback available for epoch {epoch}")
                
                # Generate samples anyway if no callback (fallback)
                try:
                    logger.info(f"[SAMPLE_GENERATION] Fallback: Generating training samples for epoch {epoch+1}")
                    pred_file = save_sample_predictions(
                        model, val_loader, device, epoch, 
                        model_dir=model_dir, 
                        class_info=class_info if 'class_info' in locals() else None, 
                        threshold=threshold if 'threshold' in locals() else 0.5
                    )
                    
                    if pred_file and os.path.exists(pred_file):
                        # Log prediction samples to MLflow and model directory
                        log_artifact_to_model_directory(pred_file, model_dir, f"predictions/epoch_{epoch+1:03d}")
                        logger.info(f"[SAMPLE_GENERATION] Fallback: Successfully generated and logged samples for epoch {epoch+1}: {pred_file}")
                    else:
                        logger.warning(f"[SAMPLE_GENERATION] Fallback: Failed to generate samples for epoch {epoch+1}")
                        
                except Exception as sample_error:
                    logger.warning(f"[SAMPLE_GENERATION] Fallback error generating samples for epoch {epoch+1}: {sample_error}")
            
            # Check early stopping condition
            if early_stopping is not None:
                # Create metrics dict for early stopping
                early_stopping_metrics = {
                    'val_dice': val_dice,
                    'val_loss': val_loss,
                    'val_accuracy': val_dice  # Use dice as accuracy proxy for consistency
                }
                
                should_stop_result = early_stopping(epoch, early_stopping_metrics, model)
                should_stop = should_stop_result.get('should_stop', False) if isinstance(should_stop_result, dict) else should_stop_result
                
                if should_stop:
                    logger.info(f"[EARLY_STOPPING] Early stopping triggered at epoch {epoch+1}")
                    logger.info(f"[EARLY_STOPPING] Best {args.early_stopping_metric}: {early_stopping.best_metric:.6f} at epoch {early_stopping.best_epoch + 1}")
                    logger.info(f"[EARLY_STOPPING] No improvement for {early_stopping.epochs_without_improvement} epochs (patience: {early_stopping.patience})")
                    
                    # Log early stopping metrics to MLflow
                    mlflow.log_metric('early_stopping_epoch', epoch + 1)
                    mlflow.log_metric('early_stopping_best_metric', early_stopping.best_metric)
                    mlflow.log_metric('early_stopping_best_epoch', early_stopping.best_epoch + 1)
                    
                    # Break the training loop
                    break
            
            # Enhanced MLflow artifact logging using the new artifact manager
            try:
                try:
                    # Try multiple import paths for Django models and utilities
                    try:
                        from core.apps.ml_manager.utils.mlflow_artifact_manager import log_epoch_artifacts
                    except ImportError:
                        try:
                            from apps.ml_manager.utils.mlflow_artifact_manager import log_epoch_artifacts
                        except ImportError:
                            from ml_manager.utils.mlflow_artifact_manager import log_epoch_artifacts
                except ImportError as import_error:
                    logger.warning(f"[MLFLOW] Could not import log_epoch_artifacts: {import_error}")
                    logger.warning("[MLFLOW] Skipping epoch artifact logging")
                    log_epoch_artifacts = None
                
                if log_epoch_artifacts is not None:
                    # Prepare artifacts dictionary for this epoch
                    epoch_artifacts = {}
                    
                    # Save sample predictions every epoch (not just every 5 epochs)
                    try:
                        threshold = args.threshold if args.threshold is not None else 0.5
                        pred_file = save_sample_predictions(model, val_loader, device, epoch, model_dir=model_dir, class_info=class_info, threshold=threshold)
                        if pred_file and os.path.exists(pred_file):
                            epoch_artifacts['predictions'] = pred_file
                            log_artifact_to_model_directory(pred_file, model_dir, f"predictions/epoch_{epoch+1:03d}")
                            logger.info(f"[MLFLOW] Successfully logged prediction samples: {pred_file}")
                        else:
                            logger.warning(f"[MLFLOW] Prediction file was not created or does not exist: {pred_file}")
                    except Exception as pred_error:
                        logger.error(f"[MLFLOW] Failed to save/log sample predictions: {pred_error}")
                
                # Save enhanced training curves if we have enough data
                if len(epoch_history) >= 2:
                    # Use the existing model directory instead of creating a new one
                    
                    enhanced_curves_file = save_enhanced_training_curves(
                        epoch_history, model_dir, epoch
                    )
                    if enhanced_curves_file:
                        epoch_artifacts['training_curves'] = enhanced_curves_file
                        # Log training curves with organized path
                        log_artifact_to_model_directory(enhanced_curves_file, model_dir, f"visualizations/training_curves/epoch_{epoch+1:03d}")
                    
                    # Save model comparison artifacts
                    comparison_artifacts = save_model_comparison_artifacts(
                        model_dir, 
                        {**metrics, 'epoch_history': epoch_history}, 
                        epoch
                    )
                    if comparison_artifacts:
                        for i, artifact in enumerate(comparison_artifacts):
                            epoch_artifacts[f'comparison_{i}'] = artifact
                            # Log comparison artifacts with organized paths
                            log_artifact_to_model_directory(artifact, model_dir, f"visualizations/comparisons/epoch_{epoch+1:03d}")
                
                # Save current epoch configuration and log it
                epoch_config = {
                    'epoch': epoch + 1,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'batch_size': args.batch_size,
                    'device': str(device),
                    'metrics': metrics
                }
                config_file = os.path.join(model_dir, 'logs', f'epoch_{epoch+1:03d}_config.json')
                os.makedirs(os.path.dirname(config_file), exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(epoch_config, f, indent=2)
                epoch_artifacts['config'] = config_file
                log_artifact_to_model_directory(config_file, model_dir, f"config/epoch_{epoch+1:03d}")
                
                # Prepare metadata for this epoch
                epoch_metadata = {
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'batch_size': args.batch_size,
                    'total_batches': len(train_loader),
                    'model_family': getattr(args, 'model_family', None) or getattr(args, 'model_type', 'unet').upper().replace('_', '-'),
                    'device': str(device),
                    'optimizer': 'Adam',
                    'loss_function': loss_function.__class__.__name__,
                    'epoch_duration': time.time() - epoch_start_time if 'epoch_start_time' in locals() else 0
                }
                
                # Log using enhanced artifact manager
                logged_paths = log_epoch_artifacts(
                    epoch=epoch + 1,  # 1-based epoch numbering
                    model_state=model.state_dict() if val_dice > best_val_dice else None,
                    metrics=metrics,
                    artifacts=epoch_artifacts,
                    metadata=epoch_metadata
                )
                
                logger.info(f"[MLFLOW] Enhanced artifact logging completed for epoch {epoch+1}: {len(logged_paths)} artifacts")
                logger.info(f"[MLFLOW] Artifact paths: {list(logged_paths.keys())}")
                
            except Exception as e:
                logger.warning(f"[MLFLOW] Enhanced artifact logging failed, falling back to basic logging: {e}")
                
                # Fallback to original artifact logging - Generate predictions every epoch
                threshold = args.threshold if args.threshold is not None else 0.5
                pred_file = save_sample_predictions(model, val_loader, device, epoch, model_dir=model_dir, class_info=class_info, threshold=threshold)
                if pred_file:
                    log_artifact_to_model_directory(pred_file, model_dir, f"predictions/epoch_{epoch+1:03d}")
                    logger.info(f"[MLFLOW] Fallback: Successfully logged prediction samples for epoch {epoch+1}")
                
                # Save training curves when we have enough data
                if len(epoch_history) >= 2:
                    # Use the existing model directory instead of creating a new one
                    
                    enhanced_curves_file = save_enhanced_training_curves(
                        epoch_history, model_dir, epoch
                    )
                    if enhanced_curves_file:
                        log_artifact_to_model_directory(enhanced_curves_file, model_dir)
                    
                    comparison_artifacts = save_model_comparison_artifacts(
                        model_dir, 
                        {**metrics, 'epoch_history': epoch_history}, 
                        epoch
                    )
                    for artifact in comparison_artifacts:
                        log_artifact_to_model_directory(artifact, model_dir)
                else:
                    # Fallback to original training curves
                    training_curves_file = save_training_curves(epoch, metrics, logger, model_dir=model_dir)
                    if training_curves_file:
                        log_artifact_to_model_directory(training_curves_file, model_dir)
            
            # Dynamic Learning Rate Adjustment
            if 'lr_scheduler' in locals() and lr_scheduler:
                try:
                    # Provide current metrics to the scheduler
                    lr_adjustment = lr_scheduler.check_and_adjust(
                        epoch=epoch + 1,  # 1-based epoch numbering
                        metrics={
                            'val_dice': val_dice,
                            'val_loss': val_loss,
                            'train_dice': train_dice,
                            'train_loss': epoch_loss
                        }
                    )
                    
                    if lr_adjustment['adjusted']:
                        # Apply the new learning rate to the optimizer
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_adjustment['new_lr']
                        
                        logger.info(f"[LR_SCHEDULER] Learning rate adjusted from {lr_adjustment['old_lr']:.6f} to {lr_adjustment['new_lr']:.6f}")
                        logger.info(f"[LR_SCHEDULER] Reason: {lr_adjustment['reason']}")
                        
                        # Log the adjustment to MLflow
                        mlflow.log_metric('lr_adjustment_epoch', epoch + 1, step=epoch+1)
                        mlflow.log_metric('lr_adjustment_reason', hash(lr_adjustment['reason']), step=epoch+1)
                    
                except Exception as lr_error:
                    logger.warning(f"[LR_SCHEDULER] Error in learning rate adjustment: {lr_error}")
            
            # Track learning rate
            current_lr = optimizer.param_groups[0]['lr']
            mlflow.log_metric('learning_rate', current_lr, step=epoch+1)
            
            # Use the appropriate primary metric for model selection
            current_metric = val_iou if args.primary_metric == 'iou' else val_dice
            
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                best_val_dice = val_dice
                best_val_iou = val_iou
                
                # Use existing model directory instead of creating a new one
                model_name = f"model_{args.model_id if args.model_id else 'unknown'}_epoch_{epoch+1}_{primary_metric_name}_{current_metric:.3f}"
                
                best_model_path = os.path.join(model_dir, "weights", "model.pth")
                
                # Create enhanced model checkpoint with metadata
                model_checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'model_metadata': {
                        'model_type': 'classification' if class_info and class_info.get('class_type') == 'classification' else 'segmentation',
                        'task_type': class_info.get('task_type') if class_info else 'segmentation',
                        'input_channels': class_info.get('input_channels', 3) if class_info else 3,
                        'num_classes': class_info.get('num_classes', 1) if class_info else 1,
                        'model_architecture': getattr(args, 'model_architecture', 'unet'),
                        'epoch': epoch + 1,
                        'validation_metric': current_metric,
                        'class_names': class_info.get('class_names', []) if class_info else []
                    },
                    'training_args': {
                        'epochs': args.epochs,
                        'batch_size': args.batch_size,
                        'learning_rate': args.learning_rate,
                        'crop_size': getattr(args, 'crop_size', 256)
                    }
                }
                
                torch.save(model_checkpoint, best_model_path)
                
                # Save model metadata
                metadata_path = save_enhanced_model_metadata(
                    model_dir=model_dir,
                    model_id=args.model_id,
                    unique_id=unique_id,
                    args=args,
                    model_info=model,
                    training_metrics=metrics,
                    model_family=model_family,
                    arch_info=arch_info
                )
                
                # Enhanced MLflow artifact logging for best model
                log_artifact_to_model_directory(best_model_path, model_dir, f"checkpoints/best_model/epoch_{epoch+1:03d}")
                log_artifact_to_model_directory(metadata_path, model_dir, f"checkpoints/best_model/metadata")
                
                # Log current epoch metrics as best model context
                best_model_context = {
                    'epoch': epoch +  1,
                    'validation_dice': val_dice,
                    'validation_loss': val_loss,
                    'train_dice': train_dice,
                    'train_loss': epoch_loss,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'improvement': current_metric - best_val_metric if best_val_metric != -1 else current_metric,
                    'model_path': best_model_path,
                    'timestamp': time.time()
                }
                
                context_file = os.path.join(model_dir, "weights", "best_model_context.json")
                with open(context_file, 'w') as f:
                    json.dump(best_model_context, f, indent=2, default=str)
                
                log_artifact_to_model_directory(context_file, model_dir, f"checkpoints/best_model/context")
                
                logger.info(f"Saved new best model with {primary_metric_name.lower()}: {current_metric:.4f} at {best_model_path}")
                logger.info(f"[MLFLOW] Best model artifacts logged to checkpoints/best_model/epoch_{epoch+1:03d}")
            
            # Save epoch checkpoint (for inference selection)
            epoch_checkpoint_dir = os.path.join(model_dir, "checkpoints", f"epoch_{epoch+1:03d}")
            os.makedirs(epoch_checkpoint_dir, exist_ok=True)
            epoch_checkpoint_path = os.path.join(epoch_checkpoint_dir, "model.pth")
            
            epoch_checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'train_loss': epoch_loss,
                'train_dice': train_dice,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'is_best': val_dice > best_val_dice,
                'model_metadata': {
                    'model_type': 'classification' if class_info and class_info.get('class_type') == 'classification' else 'segmentation',
                    'task_type': class_info.get('task_type') if class_info else 'segmentation',
                    'input_channels': class_info.get('input_channels', 3) if class_info else 3,
                    'num_classes': class_info.get('num_classes', 1) if class_info else 1,
                    'model_architecture': getattr(args, 'model_architecture', 'unet'),
                    'crop_size': getattr(args, 'crop_size', 256)
                }
            }
            
            torch.save(epoch_checkpoint, epoch_checkpoint_path)
            logger.info(f"[CHECKPOINT] Saved epoch {epoch+1} checkpoint: {epoch_checkpoint_path}")
            
            # Log to MLflow and model directory
            log_artifact_to_model_directory(epoch_checkpoint_path, model_dir, f"checkpoints/epoch_{epoch+1:03d}")
            
            # --- ZAPIS ŚCIEŻEK DO WIZUALIZACJI DLA GUI ---
            try:
                vis_json_path = os.path.join(model_dir, 'artifacts', 'visualizations.json')
                # Zbierz ścieżki do wizualizacji
                vis_data = {}
                # Wejścia i maski z treningu
                sample_vis_path = os.path.join(model_dir, 'artifacts', 'sample_inputs_and_masks.png')
                if os.path.exists(sample_vis_path):
                    vis_data['train_sample_inputs_and_masks'] = sample_vis_path
                # Wejścia i maski z walidacji
                val_vis_path = os.path.join(model_dir, 'artifacts', 'val_sample_inputs_and_masks.png')
                if os.path.exists(val_vis_path):
                    vis_data['val_sample_inputs_and_masks'] = val_vis_path
                # Predykcje z tej epoki
                pred_dir = os.path.join(model_dir, f'predictions/epoch_{epoch+1:03d}')
                if os.path.exists(pred_dir):
                    pred_files = [os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.endswith('.png')]
                    if pred_files:
                        vis_data['predictions'] = pred_files
                # Zapisz lub zaktualizuj plik JSON
                if os.path.exists(vis_json_path):
                    with open(vis_json_path, 'r') as f:
                        old_data = json.load(f)
                    old_data.update({f'epoch_{epoch+1}': vis_data})
                    with open(vis_json_path, 'w') as f:
                        json.dump(old_data, f, indent=2)
                else:
                    with open(vis_json_path, 'w') as f:
                        json.dump({f'epoch_{epoch+1}': vis_data}, f, indent=2)
                logger.info(f"[GUI] Zaktualizowano artifacts/visualizations.json dla GUI")
            except Exception as e:
                logger.warning(f"[GUI] Nie udało się zaktualizować visualizations.json: {e}")
            # ...existing code...

        logger.info(f"Training completed. Best validation {primary_metric_name} score: {best_val_metric:.4f}")
        
        # Enhanced final model logging using the new artifact manager
        try:
            try:
                from core.apps.ml_manager.utils.mlflow_artifact_manager import log_final_model
            except ImportError as import_error:
                logger.warning(f"[MLFLOW] Could not import log_final_model: {import_error}")
                logger.warning("[MLFLOW] Skipping final model artifact logging")
                log_final_model = None
            
            if log_final_model is not None:
                # Prepare comprehensive model information
                model_info = {
                'architecture': getattr(arch_info, 'display_name', 'MONAI UNet') if arch_info else 'MONAI UNet',
                'framework': getattr(arch_info, 'framework', 'PyTorch') if arch_info else 'PyTorch',
                'model_family': getattr(args, 'model_family', None) or getattr(args, 'model_type', 'unet').upper().replace('_', '-'),
                'architecture_key': getattr(arch_info, 'key', 'monai_unet') if arch_info else 'monai_unet',
                'version': getattr(arch_info, 'version', '1.0.0') if arch_info else '1.0.0',
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
                'training_config': {
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'optimizer': 'Adam',
                    'loss_function': loss_function.__class__.__name__,
                    'device': str(device),
                    'validation_split': args.validation_split
                },
                'data_info': {
                    'total_training_batches': len(train_loader),
                    'total_validation_batches': len(val_loader),
                    'crop_size': args.crop_size,
                    'augmentations': {
                        'random_flip': getattr(args, 'random_flip', False),
                        'random_rotate': getattr(args, 'random_rotate', False),
                        'random_scale': getattr(args, 'random_scale', False),
                        'random_intensity': getattr(args, 'random_intensity', False)
                    }
                }
            }
            
            # Best metrics summary
            best_metrics = {
                'best_val_dice': best_val_dice,
                'best_val_iou': best_val_iou,
                'best_val_metric': best_val_metric,
                'primary_metric_name': primary_metric_name,
                'final_train_loss': epoch_history[-1]['train_loss'] if epoch_history else 0.0,
                'final_val_loss': epoch_history[-1]['val_loss'] if epoch_history else 0.0,
                'final_train_dice': epoch_history[-1]['train_dice'] if epoch_history else 0.0,
                'final_val_dice': epoch_history[-1]['val_dice'] if epoch_history else 0.0,
                'total_epochs_trained': len(epoch_history),
                'convergence_epoch': next((i+1 for i, h in enumerate(epoch_history) if h.get(f'val_{primary_metric_name.lower()}', h.get('val_dice', 0)) == best_val_metric), len(epoch_history))
            }
            
            # Use the model directory that was created during training
            final_model_dir = model_dir if 'model_dir' in locals() else None
            
            if final_model_dir:
                # Log using enhanced artifact manager
                final_logged_paths = log_final_model(
                    model_info=model_info,
                    model_directory=final_model_dir,
                    best_metrics=best_metrics
                )
                
                # Additional comprehensive artifact logging using MLflow APIs
                logger.info("[MLFLOW] Logging additional comprehensive artifacts...")
                
                # Log final training history as JSON
                history_file = os.path.join(final_model_dir, "training_history.json")
                with open(history_file, 'w') as f:
                    json.dump({
                        'epoch_history': epoch_history,
                        'best_metrics': best_metrics,
                        'training_summary': {
                            'total_epochs': len(epoch_history),
                            'best_epoch': next((i+1 for i, h in enumerate(epoch_history) if h.get(f'val_{primary_metric_name.lower()}', h.get('val_dice', 0)) == best_val_metric), len(epoch_history)),
                            'final_lr': optimizer.param_groups[0]['lr'],
                            'model_parameters': sum(p.numel() for p in model.parameters())
                        }
                    }, f, indent=2)
                log_artifact_to_model_directory(history_file, final_model_dir, "final_model/training_history")
                
                # Log final model state with comprehensive metadata
                if best_model_path and os.path.exists(best_model_path):
                    log_artifact_to_model_directory(best_model_path, final_model_dir, "final_model/weights")
                
                # Log complete training configuration
                final_config = {
                    'model_info': model_info,
                    'training_args': vars(args),
                    'final_metrics': best_metrics,
                    'device_info': {
                        'device': str(device),
                        'cuda_available': torch.cuda.is_available(),
                        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
                    }
                }
                config_file = os.path.join(final_model_dir, "complete_config.json")
                with open(config_file, 'w') as f:
                    json.dump(final_config, f, indent=2, default=str)
                log_artifact_to_model_directory(config_file, final_model_dir, "final_model/configuration")
                
                # Log PyTorch model using MLflow's model logging - with safe signature handling
                try:
                    if sample_batch is not None and "image" in sample_batch and sample_images is not None:
                        input_example = sample_images[:1].detach().cpu().numpy()
                        try:
                            # Try to create signature with model inference
                            model_output = model(sample_images[:1].to(device))
                            # Apply appropriate post-processing based on model output channels
                            if model_output.shape[1] == 1:
                                # Binary segmentation - apply sigmoid
                                processed_output = torch.sigmoid(model_output)
                            else:
                                # Multi-class - apply softmax
                                processed_output = torch.softmax(model_output, dim=1)
                            
                            signature = mlflow.models.infer_signature(
                                input_example,
                                processed_output.detach().cpu().numpy()
                            )
                            mlflow.pytorch.log_model(
                                model,
                                "pytorch_model",
                                input_example=input_example,
                                signature=signature
                            )
                            logger.info("[MLFLOW] PyTorch model logged with signature and input example")
                        except Exception as sig_error:
                            logger.warning(f"[MLFLOW] Failed to create model signature: {sig_error}")
                            # Fallback without signature but with input example
                            mlflow.pytorch.log_model(
                                model,
                                "pytorch_model",
                                input_example=input_example
                            )
                            logger.info("[MLFLOW] PyTorch model logged with input example only (no signature)")
                    else:
                        logger.warning("[MLFLOW] No valid sample batch available for model signature")
                        # Final fallback without any examples
                        mlflow.pytorch.log_model(model, "pytorch_model_fallback")
                        logger.info("[MLFLOW] PyTorch model logged without signature or input example")
                except Exception as model_log_error:
                    logger.warning(f"[MLFLOW] Failed to log PyTorch model: {model_log_error}")
                    # Fallback without signature
                    mlflow.pytorch.log_model(model, "pytorch_model_fallback")
                    logger.info("[MLFLOW] PyTorch model logged with fallback method")
                
                logger.info(f"[MLFLOW] Enhanced final model logging completed: {len(final_logged_paths)} artifacts")
            else:
                logger.warning("[MLFLOW] No model directory available for enhanced logging, using fallback")
                raise Exception("No model directory for enhanced logging")
                
        except Exception as e:
            logger.warning(f"[MLFLOW] Enhanced final model logging failed, using fallback: {e}")
            
            # Fallback to original final model logging - with safe sample batch handling
            try:
                if sample_batch is not None and "image" in sample_batch and sample_images is not None:
                    input_example = sample_images[:1].detach().cpu().numpy()
                    try:
                        # Try with signature first
                        model_output = model(sample_images[:1].to(device))
                        # Apply appropriate post-processing based on model output channels
                        if model_output.shape[1] == 1:
                            # Binary segmentation - apply sigmoid
                            processed_output = torch.sigmoid(model_output)
                        else:
                            # Multi-class - apply softmax
                            processed_output = torch.softmax(model_output, dim=1)
                        
                        signature = mlflow.models.infer_signature(
                            input_example,
                            processed_output.detach().cpu().numpy()
                        )
                        mlflow.pytorch.log_model(
                            model,
                            "model",
                            input_example=input_example,
                            signature=signature
                        )
                        logger.info("[MLFLOW] Model logged to MLflow under 'model' artifact (fallback with signature).")
                    except Exception as fallback_signature_error:
                        logger.warning(f"[MLFLOW] Failed to log fallback model with signature: {fallback_signature_error}")
                        # Final fallback with input example but no signature
                        mlflow.pytorch.log_model(
                            model,
                            "model",
                            input_example=input_example,
                        )
                        logger.info("[MLFLOW] Model logged to MLflow under 'model' artifact (fallback without signature).")
                else:
                    logger.warning("[MLFLOW] No valid sample batch for fallback model logging")
                    # Ultimate fallback without any examples
                    mlflow.pytorch.log_model(model, "model")
                    logger.info("[MLFLOW] Model logged to MLflow under 'model' artifact (no examples).")
            except Exception as fallback_error:
                logger.error(f"[MLFLOW] All fallback model logging attempts failed: {fallback_error}")
                # Final emergency fallback
                try:
                    mlflow.pytorch.log_model(model, "model_emergency")
                    logger.info("[MLFLOW] Emergency model logging successful")
                except Exception as emergency_error:
                    logger.error(f"[MLFLOW] Emergency model logging also failed: {emergency_error}")
        
        # Enhanced training artifacts and summary logging
        try:
            # Log training logs with comprehensive organization
            try:
                from core.apps.ml_manager.utils.mlflow_artifact_manager import MLflowArtifactManager
            except ImportError as import_error:
                logger.warning(f"[MLFLOW] Could not import MLflowArtifactManager: {import_error}")
                logger.warning("[MLFLOW] Skipping artifact manager logging")
                MLflowArtifactManager = None
            
            if MLflowArtifactManager is not None:
                with MLflowArtifactManager() as artifact_manager:
                    # Collect all training logs
                    log_files = [
                        os.path.join(CORE_DATA_DIR, 'models', 'artifacts', 'training.log'),
                        os.path.join(model_dir, 'logs', 'training.log') if 'model_dir' in locals() else None
                    ]
                    log_files = [f for f in log_files if f and os.path.exists(f)]
                    
                    if log_files:
                        logged_log_paths = artifact_manager.log_training_logs(log_files, "training")
                    else:
                        logged_log_paths = []
                    logger.info(f"[MLFLOW] Logged {len(logged_log_paths)} training log files")
                    
                    # Also log individual log files with organized paths
                    for log_file in log_files:
                        log_artifact_to_model_directory(log_file, final_model_dir, "logs/training")
                
                # Log complete training summary
                training_summary = {
                    'training_completed': True,
                    'total_epochs': len(epoch_history),
                    'best_validation_dice': best_val_dice,
                    'final_model_path': best_model_path,
                    'model_directory': model_dir if 'model_dir' in locals() else None,
                    'training_duration': time.time() - training_start_time if 'training_start_time' in locals() else None,
                    'device_used': str(device),
                    'dataset_info': {
                        'training_samples': len(train_ds) if 'train_ds' in locals() else train_samples,
                        'validation_samples': len(val_ds) if 'val_ds' in locals() else val_samples,
                        'total_batches_per_epoch': len(train_loader),
                        'validation_batches_per_epoch': len(val_loader)
                    }
                }
                
                summary_file = os.path.join(CORE_DATA_DIR, 'models', 'artifacts', 'training_summary.json')
                os.makedirs(os.path.dirname(summary_file), exist_ok=True)
                with open(summary_file, 'w') as f:
                    json.dump(training_summary, f, indent=2, default=str)
                
                log_artifact_to_model_directory(summary_file, final_model_dir, "summaries/training")
                logger.info("[MLFLOW] Comprehensive training summary logged")
                
        except Exception as e:
            logger.warning(f"[MLFLOW] Failed to log training logs with enhanced manager: {e}")
            
            # Fallback: Log training logs using basic MLflow artifact APIs
            try:
                # Log available training logs
                fallback_logs = [
                    'artifacts/training_logs/training.log',
                    'data/logs/training.log'
                ]
                
                for log_path in fallback_logs:
                    if os.path.exists(log_path):
                        log_artifact_to_model_directory(log_path, final_model_dir, "logs/training_fallback")
                        logger.info(f"[MLFLOW] Fallback logged: {log_path}")
                        
                # Create and log basic training summary
                basic_summary = {
                    'training_completed': True,
                    'epochs': len(epoch_history),
                    'best_dice': best_val_dice,
                    'device': str(device)
                }
                
                fallback_summary_file = 'data/artifacts/basic_training_summary.json'
                os.makedirs(os.path.dirname(fallback_summary_file), exist_ok=True)
                with open(fallback_summary_file, 'w') as f:
                    json.dump(basic_summary, f, indent=2)
                
                log_artifact_to_model_directory(fallback_summary_file, final_model_dir, "summaries/basic")
                        
            except Exception as fallback_e:
                logger.warning(f"[MLFLOW] Fallback log artifact also failed: {fallback_e}")
        
        # Log interactive training plot if available
        try:
            interactive_plot_file = save_interactive_training_plot(epoch_history, model_dir)
            if interactive_plot_file:
                log_artifact_to_model_directory(interactive_plot_file, final_model_dir)
        except Exception as e:
            logger.warning(f"[MLFLOW] Failed to log interactive plot: {e}")
        if hasattr(args, 'model_id') and args.model_id is not None:
            try:
                # Import registry functions using absolute import
                import sys
                current_script_dir = os.path.dirname(os.path.abspath(__file__))
                core_dir = os.path.abspath(os.path.join(current_script_dir, '..', '..', 'core', 'apps'))
                if core_dir not in sys.path:
                    sys.path.append(core_dir)
                from core.apps.ml_manager.utils.mlflow_utils import register_model, transition_model_stage
                
                # Determine model family/type for registry naming and tags
                model_family = getattr(args, 'model_family', None) or getattr(args, 'model_type', None) or 'generic-model'
                # Use the same unique_id for consistent naming between artifacts and registry
                registry_model_name = f"{unique_id}_v1.0.0"
                registry_tags = {
                    "model_family": model_family,
                    "framework": "PyTorch",
                    "task": "image_segmentation",
                    "best_val_dice": str(best_val_dice),
                    "training_epochs": str(args.epochs),
                    "batch_size": str(args.batch_size),
                    "learning_rate": str(args.learning_rate)
                }
                # Register the model
                model_info = register_model(
                    run_id=args.mlflow_run_id,
                    model_name=registry_model_name,
                    model_description=f"{model_family} model for coronary segmentation trained with {args.epochs} epochs, best val {primary_metric_name.lower()}: {best_val_metric:.4f}",
                    tags=registry_tags
                )
                
                if model_info:
                    logger.info(f"[REGISTRY] Model registered: {model_info['name']} v{model_info['version']}")
                    
                    # Update Django model with registry information
                    if callback:
                        callback.update_registry_info(
                            registry_model_name=model_info['name'],
                            registry_model_version=model_info['version'],
                            is_registered=True
                        )
                    
                    # Auto-promote to Staging if performance is good
                    if best_val_dice > 0.8:  # Threshold for auto-promotion
                        success = transition_model_stage(
                            model_name=model_info['name'],
                            version=model_info['version'],
                            stage="Staging"
                        )
                        if success:
                            logger.info(f"[REGISTRY] Model auto-promoted to Staging (val_dice: {best_val_dice:.4f})")
                            if callback:
                                callback.update_registry_stage("Staging")
                        else:
                            logger.warning("[REGISTRY] Failed to auto-promote model to Staging")
                    
                else:
                    logger.warning("[REGISTRY] Failed to register model")
                    
            except Exception as e:
                logger.warning(f"[REGISTRY] Model registration failed: {e}")
        
        if callback:
            callback.on_training_end({
                "best_val_dice": best_val_dice,
                "best_val_iou": best_val_iou,
                "best_val_metric": best_val_metric,
                "primary_metric_name": primary_metric_name
            })
        
        # === COMPREHENSIVE TRAINING SUMMARY WITH DATA RANGE ANALYSIS ===
        logger.info("=" * 80)
        logger.info("🏁 TRAINING COMPLETED - COMPREHENSIVE SUMMARY REPORT")
        logger.info("=" * 80)
        
        # Training Performance Summary
        logger.info(f"📊 TRAINING PERFORMANCE:")
        logger.info(f"   📈 Best Validation {primary_metric_name} Score: {best_val_metric:.4f}")
        logger.info(f"   📈 Best Validation Dice Score: {best_val_dice:.4f}")
        logger.info(f"   📈 Best Validation IoU Score: {best_val_iou:.4f}")
        logger.info(f"   🔄 Total Epochs Completed: {len(epoch_history)}")
        logger.info(f"   ⏱️  Training Duration: {time.time() - training_start_time:.1f} seconds" if 'training_start_time' in locals() else "   ⏱️  Training Duration: Not tracked")
        logger.info(f"   🖥️  Device Used: {device}")
        
        # Dataset Summary  
        logger.info(f"📁 DATASET SUMMARY:")
        if 'train_ds' in locals() and 'val_ds' in locals():
            logger.info(f"   🏋️  Training Samples: {len(train_ds)}")
            logger.info(f"   ✅ Validation Samples: {len(val_ds)}")
            logger.info(f"   📦 Total Dataset Size: {len(train_ds) + len(val_ds)}")
        elif 'train_samples' in locals() and 'val_samples' in locals():
            logger.info(f"   🏋️  Training Samples: {train_samples}")
            logger.info(f"   ✅ Validation Samples: {val_samples}")  
            logger.info(f"   📦 Total Dataset Size: {train_samples + val_samples}")
        else:
            logger.info(f"   ❓ Dataset size information not available")
        
        # Batch Processing Summary
        logger.info(f"   🔢 Batches per Epoch: {len(train_loader)} train, {len(val_loader)} validation")
        logger.info(f"   📏 Batch Size: {args.batch_size}")
        logger.info(f"   👷 Workers: {getattr(args, 'num_workers', 'Unknown')}")
        
        # Model Architecture Summary
        if 'total_params' in locals():
            logger.info(f"🏗️  MODEL ARCHITECTURE:")
            logger.info(f"   🔧 Parameters: {total_params:,} total ({trainable_params:,} trainable)")
            logger.info(f"   💾 Model Size: {total_params * 4 / (1024**2):.2f} MB")
        
        # Training Configuration Summary
        logger.info(f"⚙️  TRAINING CONFIGURATION:")
        logger.info(f"   📚 Learning Rate: {args.learning_rate}")
        logger.info(f"   🎯 Loss Function: {loss_function.__class__.__name__}")
        logger.info(f"   🔄 Optimizer: {getattr(args, 'optimizer', 'adam').upper()}")
        logger.info(f"   📐 Target Size: {getattr(args, 'crop_size', '128')}px")
        logger.info(f"   🎚️  Threshold: {getattr(args, 'threshold', 0.5)}")
        
        # Data Range Analysis Summary (if data was analyzed)
        logger.info(f"🔍 DATA CHARACTERISTICS SUMMARY:")
        logger.info(f"   💽 Dataset Type: {getattr(args, 'dataset_type', 'auto-detected')}")
        logger.info(f"   📊 Data Path: {args.data_path}")
        
        # MLflow Integration Summary
        if hasattr(args, 'mlflow_run_id') and args.mlflow_run_id:
            logger.info(f"📈 MLFLOW INTEGRATION:")
            logger.info(f"   🆔 Run ID: {args.mlflow_run_id}")
            logger.info(f"   📝 Experiment: {mlflow.get_experiment(mlflow.active_run().info.experiment_id).name if mlflow.active_run() else 'N/A'}")
        
        # File Artifacts Summary
        if 'model_dir' in locals() and model_dir:
            logger.info(f"📁 GENERATED ARTIFACTS:")
            logger.info(f"   📂 Model Directory: {model_dir}")
            if 'best_model_path' in locals():
                logger.info(f"   🏆 Best Model: {os.path.basename(best_model_path)}")
            
            # List key artifact files
            artifacts_dir = os.path.join(model_dir, 'artifacts')
            if os.path.exists(artifacts_dir):
                artifact_files = [f for f in os.listdir(artifacts_dir) if f.endswith(('.png', '.json', '.txt', '.html'))]
                if artifact_files:
                    logger.info(f"   📊 Artifacts Generated: {len(artifact_files)} files")
                    for artifact in sorted(artifact_files)[:5]:  # Show first 5
                        logger.info(f"     • {artifact}")
                    if len(artifact_files) > 5:
                        logger.info(f"     • ... and {len(artifact_files) - 5} more")
                    
                    # Enhanced MLflow artifact logging with retry mechanism
                    try:
                        from core.apps.ml_manager.utils.mlflow_artifact_logger import force_log_all_artifacts
                        logged_artifacts = force_log_all_artifacts(model_dir, args.mlflow_run_id)
                        if logged_artifacts:
                            logger.info(f"   ✅ MLflow: {len(logged_artifacts)} artifacts logged successfully")
                            logger.info(f"   📊 Artifacts available in MLflow UI at experiment artifacts section")
                        else:
                            logger.warning(f"   ⚠️  No artifacts were logged to MLflow")
                    except Exception as mlflow_error:
                        logger.error(f"   ❌ MLflow artifact logging failed: {mlflow_error}")
                        logger.info(f"   💡 Check MLflow connection and experiment configuration")
                        # Continue execution even if MLflow logging fails
        
        # Performance Recommendations
        logger.info(f"💡 PERFORMANCE INSIGHTS:")
        if best_val_dice < 0.3:
            logger.info(f"   ⚠️  Low Dice Score detected - Consider:")
            logger.info(f"     • Increasing training epochs")
            logger.info(f"     • Adjusting learning rate")
            logger.info(f"     • Verifying data quality and ranges")
            logger.info(f"     • Checking mask normalization (0-1 vs 0-255)")
        elif best_val_dice < 0.7:
            logger.info(f"   🔄 Moderate performance - Potential improvements:")
            logger.info(f"     • Fine-tuning hyperparameters")
            logger.info(f"     • Adding data augmentation")
            logger.info(f"     • Increasing model complexity")
        else:
            logger.info(f"   ✅ Excellent performance achieved!")
            logger.info(f"   🚀 Model ready for production consideration")
        
        # Next Steps
        logger.info(f"🎯 NEXT STEPS:")
        logger.info(f"   • Review training curves and sample predictions")
        logger.info(f"   • Test model on independent test set")
        logger.info(f"   • Consider model deployment if performance is satisfactory")
        if hasattr(args, 'model_id') and args.model_id is not None:
            logger.info(f"   • Check MLflow for detailed metrics and artifacts")
        
        logger.info("=" * 80)
        logger.info("🎉 TRAINING SUMMARY COMPLETE")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        if callback:
            callback.on_training_failed(str(e))
        raise
    finally:
        # Stop system monitoring
        if system_monitor:
            try:
                system_monitor.stop_monitoring()
                logger.info("[MONITORING] System monitoring stopped")
            except Exception as e:
                logger.warning(f"[MONITORING] Failed to stop system monitoring: {e}")
        
        # Log essential training artifacts to MLflow (detailed logging handled by Celery task)
        if mlflow.active_run() and 'model_dir' in locals() and model_dir:
            try:
                # Basic artifact logging 
                logger.info(f"[MLFLOW] Logging essential training artifacts from: {model_dir}")
                
                # Log training log file
                log_path = os.path.join(model_dir, 'logs', 'training.log')
                if os.path.exists(log_path):
                    log_artifact_to_model_directory(log_path, model_dir, "logs")
                    logger.info(f"[MLFLOW] Logged training log as artifact")
                
                # Log model artifacts from model directory  
                artifact_count = 0
                for item in Path(model_dir).rglob('*'):
                    if item.is_file() and item.suffix in ['.pth', '.json', '.txt']:
                        try:
                            log_artifact_to_model_directory(str(item), model_dir, "model_artifacts")
                            artifact_count += 1
                        except Exception as e:
                            logger.debug(f"Failed to log artifact {item}: {e}")
                
                logger.info(f"[MLFLOW] Logged {artifact_count} model artifacts")
                
                # Log final training configuration
                config_path = os.path.join(model_dir, 'training_config.json')
                if os.path.exists(config_path):
                    log_artifact_to_model_directory(config_path, model_dir, "config")
                    logger.info(f"[MLFLOW] Logged final training configuration")
                
            except Exception as e:
                logger.warning(f"[MLFLOW] Failed to log training artifacts: {e}")
        
        # Handle MLflow run completion with proper status
        if mlflow.active_run():
            try:
                # Log training completion status
                if 'training_stopped_early' in locals() and training_stopped_early:
                    logger.info("[MLFLOW] Training was stopped early by user request")
                    mlflow.set_tag("training_status", "stopped_by_user")
                    mlflow.set_tag("completion_reason", "user_requested_stop")
                    
                    # Log partial training metrics if available
                    if 'epoch_history' in locals() and epoch_history:
                        mlflow.log_metric("final_epoch_completed", len(epoch_history))
                        mlflow.log_metric("epochs_trained", len(epoch_history))
                        mlflow.log_metric("training_completion_percentage", (len(epoch_history) / args.epochs) * 100)
                        
                        # Log best metrics achieved before stopping
                        if epoch_history:
                            best_val_dice = max([epoch.get('val_dice', 0) for epoch in epoch_history])
                            best_epoch = [i for i, epoch in enumerate(epoch_history) if epoch.get('val_dice', 0) == best_val_dice][0]
                            mlflow.log_metric("best_val_dice_before_stop", best_val_dice)
                            mlflow.log_metric("best_epoch_before_stop", best_epoch + 1)
                    
                    # Call callback to update model status
                    if callback:
                        callback.on_training_stopped()
                        
                else:
                    logger.info("[MLFLOW] Training completed normally")
                    mlflow.set_tag("training_status", "completed")
                    mlflow.set_tag("completion_reason", "normal_completion")
                    
                    # Log comprehensive final metrics
                    if 'epoch_history' in locals() and epoch_history:
                        mlflow.log_metric("total_epochs_completed", len(epoch_history))
                        mlflow.log_metric("training_completion_percentage", 100.0)
                        
                        # Log final training summary
                        final_metrics = epoch_history[-1]
                        mlflow.log_metric("final_train_loss", final_metrics.get('train_loss', 0))
                        mlflow.log_metric("final_val_loss", final_metrics.get('val_loss', 0))
                        mlflow.log_metric("final_train_dice", final_metrics.get('train_dice', 0))
                        mlflow.log_metric("final_val_dice", final_metrics.get('val_dice', 0))
                        
                        # Log best metrics summary
                        if 'best_val_dice' in locals():
                            mlflow.log_metric("best_val_dice_final", best_val_dice)
                            best_epoch_idx = [i for i, epoch in enumerate(epoch_history) if epoch.get('val_dice', 0) == best_val_dice]
                            if best_epoch_idx:
                                mlflow.log_metric("best_epoch_final", best_epoch_idx[0] + 1)
                
                # Log training artifacts and model info
                if 'model_dir' in locals() and model_dir:
                    try:
                        # Log model directory structure
                        mlflow.set_tag("model_directory", model_dir)
                        mlflow.set_tag("model_artifacts_logged", "true")
                        
                        # Log model files if they exist
                        model_files = []
                        for file_pattern in ['*.pth', '*.pt', '*.pkl']:
                            model_files.extend(glob.glob(os.path.join(model_dir, '**', file_pattern), recursive=True))
                        
                        if model_files:
                            mlflow.set_tag("model_files_count", len(model_files))
                            for i, model_file in enumerate(model_files[:5]):  # Log first 5 files
                                rel_path = os.path.relpath(model_file, model_dir)
                                mlflow.set_tag(f"model_file_{i+1}", rel_path)
                        
                        # Log training logs as artifact
                        model_log_path = os.path.join(model_dir, 'logs', 'training.log')
                        if os.path.exists(model_log_path):
                            log_artifact_to_model_directory(model_log_path, model_dir, "logs")
                            logger.info(f"[MLFLOW] Logged training log as artifact: {model_log_path}")
                        
                        # Log config files if they exist
                        config_dir = os.path.join(model_dir, 'configs')
                        if os.path.exists(config_dir):
                            for config_file in os.listdir(config_dir):
                                config_path = os.path.join(config_dir, config_file)
                                if os.path.isfile(config_path):
                                    log_artifact_to_model_directory(config_path, model_dir, "configs")
                        
                        logger.info(f"[MLFLOW] Logged model artifacts from: {model_dir}")
                        
                    except Exception as artifact_error:
                        logger.warning(f"[MLFLOW] Failed to log some artifacts: {artifact_error}")
                
                # Log final training configuration summary
                try:
                    config_summary = {
                        'model_type': getattr(args, 'model_type', 'unknown'),
                        'batch_size': getattr(args, 'batch_size', 0),
                        'learning_rate': getattr(args, 'learning_rate', 0),
                        'epochs_requested': getattr(args, 'epochs', 0),
                        'epochs_completed': len(epoch_history) if 'epoch_history' in locals() else 0,
                        'device': str(device) if 'device' in locals() else 'unknown'
                    }
                    
                    for key, value in config_summary.items():
                        mlflow.set_tag(f"config_{key}", str(value))
                    
                    logger.info("[MLFLOW] Logged final training configuration")
                    
                except Exception as config_error:
                    logger.warning(f"[MLFLOW] Failed to log final configuration: {config_error}")
                
                # Log final training duration
                if 'start_time' in locals():
                    total_duration = time.time() - start_time
                    mlflow.log_metric("total_training_duration_seconds", total_duration)
                    mlflow.log_metric("total_training_duration_minutes", total_duration / 60)
                    
                logger.info("[MLFLOW] Ending MLflow run with proper status")
                
            except Exception as e:
                logger.warning(f"[MLFLOW] Error updating MLflow run status: {e}")
            
            finally:
                # Properly end MLflow run with status information
                if mlflow.active_run():
                    try:
                        # Mark run as completed but don't end it here - Celery task will handle that
                        mlflow.set_tag("run_status", "FINISHED")
                        mlflow.set_tag("completion_time", datetime.now().isoformat())
                        mlflow.set_tag("subprocess_completed", "true")
                        
                        # Log final status based on how training ended
                        if 'training_stopped_early' in locals() and training_stopped_early:
                            mlflow.set_tag("training_result", "stopped_early")
                        elif 'best_val_dice' in locals():
                            mlflow.set_tag("training_result", "completed_successfully")
                        else:
                            mlflow.set_tag("training_result", "completed")
                            
                        logger.info("[MLFLOW] Run marked as completed - Celery task will finalize and end the run")
                    except Exception as tag_error:
                        logger.warning(f"[MLFLOW] Error setting final tags: {tag_error}")
                    
                    # DO NOT end the run here - let Celery task handle final artifacts and run closure
                    logger.info("[MLFLOW] ✅ Training subprocess completed - run remains active for Celery finalization")
                else:
                    logger.warning("[MLFLOW] No active run to end")
    
    # Return training results for tasks.py
    result = {
        'success': True,
        'best_val_dice': best_val_dice if 'best_val_dice' in locals() else 0,
        'best_val_iou': best_val_iou if 'best_val_iou' in locals() else 0,
        'val_dice': best_val_dice if 'best_val_dice' in locals() else 0,  # Alias for compatibility
        'final_epoch': len(epoch_history) if 'epoch_history' in locals() else 0,
        'total_epochs': args.epochs,
        'model_dir': model_dir if 'model_dir' in locals() else None
    }
    
    logger.info(f"[TRAIN] Returning result: {result}")
    
    # CRITICAL: Print result as JSON for subprocess parsing
    print("TRAINING_RESULT_JSON:", json.dumps(result))
    
    return result

# Placeholder for save_interactive_training_plot
def save_interactive_training_plot(epoch_history, model_dir):
    logger.info("[PLOT] `save_interactive_training_plot` called, placeholder implementation.")
    # Example: Create a dummy file to satisfy logging if needed
    # dummy_plot_path = os.path.join(model_dir, "interactive_plot_placeholder.html")
    # with open(dummy_plot_path, "w") as f:
    #     f.write("<html><body>Placeholder for interactive plot</body></html>")
    # return dummy_plot_path
    return None

def get_inference_transforms(image_size=(256, 256), use_original_size=False, input_channels=3):
    """Get transforms for inference - uses PIL-compatible orientation
    
    Args:
        image_size: Target image size as (height, width). Ignored if use_original_size=True
        use_original_size: If True, don't resize the image, keep original dimensions
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
    """
    def load_pil_compatible_image(filepath):
        """Load image using PIL to maintain browser-compatible orientation"""
        from PIL import Image
        import numpy as np
        import torch
        
        # Check if file exists before trying to open it
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image file not found: {filepath}")
        
        # Log file info for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Loading image from: {filepath}")
        logger.info(f"File size: {os.path.getsize(filepath)} bytes")
        logger.info(f"File exists: {os.path.exists(filepath)}")
        
        try:
            img = Image.open(filepath)
        except Exception as e:
            logger.error(f"Failed to open image {filepath}: {e}")
            raise IOError(f"Failed to open image {filepath}: {e}")
        
        # Convert based on required channels
        if input_channels == 1:
            # Convert to grayscale for single channel models
            if img.mode != 'L':
                img = img.convert('L')
            img_array = np.array(img)
            # Add channel dimension for grayscale
            if len(img_array.shape) == 2:
                img_array = img_array[np.newaxis, :, :]  # Add channel dim: HW -> CHW
        else:
            # Keep RGB format for multi-channel models
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)
            if len(img_array.shape) == 3:
                # Convert HWC to CHW format
                img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
        
        # Convert to torch tensor and normalize to [0, 1]
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        
        # Apply appropriate normalization
        if input_channels == 1:
            # Grayscale normalization
            mean = torch.tensor([0.5]).view(1, 1, 1)
            std = torch.tensor([0.5]).view(1, 1, 1)
        else:
            # RGB normalization to match training
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor
    
    if use_original_size:
        # Don't resize, just load with RGB normalization
        return Compose([
            Lambda(load_pil_compatible_image),
        ])
    else:
        return Compose([
            Lambda(load_pil_compatible_image),
            Resize(spatial_size=image_size, mode="bilinear"),
        ])

def get_display_oriented_image(input_file, target_size=(256, 256)):
    """Load image in display orientation (same as browser shows) for comparison"""
    # Load image using PIL to maintain browser-compatible orientation
    img = Image.open(input_file)
    
    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
    
    # Resize to target size
    img = img.resize(target_size, Image.Resampling.BILINEAR)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    return img_array

def run_inference(model_path, input_path, output_dir, device="cuda", weights_path=None, model_type="unet", crop_size=128, threshold=0.5):
    """Run inference on input images using a trained model
    
    Args:
        model_path: Path to the trained model weights (MLflow or .pth)
        input_path: Path to input image or directory of images
        output_dir: Directory to save predictions
        device: Device to run inference on
        weights_path: Optional path to a .pth file to load weights from
        model_type: Type of model architecture to use
        crop_size: Target size for input images (128, 256, 384, 512)
        threshold: Binary segmentation threshold for hard predictions
    """
    import json
    import time
    from PIL import Image
    
    logger = logging.getLogger(__name__)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load checkpoint first to get model metadata
    if weights_path:
        checkpoint = torch.load(weights_path, map_location=device)
    else:
        checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats and extract metadata
    state_dict = None
    metadata = {}
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            metadata = checkpoint.get('model_metadata', {})
            logger.info("Loading model from 'model_state_dict' key")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            metadata = checkpoint.get('metadata', {})
            logger.info("Loading model from 'state_dict' key")
        else:
            # Assume checkpoint is the state_dict directly
            state_dict = checkpoint
            logger.info("Loading model from checkpoint directly")
    else:
        state_dict = checkpoint
        logger.info("Loading model from checkpoint directly (not dict)")
    
    # Get input channels - prioritize state_dict over metadata
    input_channels = None
    
    # First try to infer from state_dict (most reliable)
    first_layer_key = None
    # Look specifically for conv weight layers, not normalization or bias
    for key in sorted(state_dict.keys()):
        if 'weight' in key and 'conv' in key and not 'adn' in key and not 'norm' in key:
            first_layer_key = key
            break
    
    if first_layer_key:
        weight_shape = state_dict[first_layer_key].shape
        if len(weight_shape) >= 2:
            input_channels = weight_shape[1]  # Input channels dimension
            logger.info(f"Inferred input_channels={input_channels} from layer {first_layer_key} with shape {weight_shape}")
        else:
            logger.warning(f"Layer {first_layer_key} has unexpected shape {weight_shape}, cannot infer input channels")
            input_channels = metadata.get('input_channels', 3)
            logger.info(f"Fallback to metadata/default input_channels={input_channels}")
    else:
        # Fallback to metadata if state_dict parsing fails
        input_channels = metadata.get('input_channels')
        if input_channels:
            logger.info(f"Using input_channels={input_channels} from metadata")
        else:
            # Debug: show all keys to understand structure
            all_keys = list(state_dict.keys())[:10]
            logger.warning(f"Could not find first conv layer. Available keys (first 10): {all_keys}")
            input_channels = 3  # Default fallback
            logger.warning("Could not infer input channels, using default 3")
    
    logger.info(f"Final input_channels={input_channels}")
    
    # Create model using architecture registry with correct channels
    model_config = get_default_model_config(model_type)
    model_config["in_channels"] = input_channels
    logger.info(f"Creating model with config: {model_config}")
    model, arch_info = create_model_from_registry(model_type, device, **model_config)
    
    # Debug: show model's first layer to verify correct creation
    if hasattr(model, 'model') and hasattr(model.model, '0'):
        first_layer = model.model[0]
        if hasattr(first_layer, 'conv') and hasattr(first_layer.conv, 'unit0'):
            first_conv = first_layer.conv.unit0.conv
            logger.info(f"Created model first conv layer expects: {first_conv.weight.shape}")
    
    # Load the state dict
    try:
        model.load_state_dict(state_dict)
        logger.info("Model state_dict loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load state_dict: {e}")
        # Show the problematic layer info for debugging
        if 'model.0.conv.unit0.conv.weight' in state_dict:
            checkpoint_shape = state_dict['model.0.conv.unit0.conv.weight'].shape
            logger.error(f"Checkpoint first layer shape: {checkpoint_shape}")
        raise
    model.eval()
    
    # Get transforms based on crop_size and detected input channels
    transforms = get_inference_transforms(image_size=(crop_size, crop_size), use_original_size=False, input_channels=input_channels)
    
    if os.path.isdir(input_path):
        input_files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    else:
        input_files = [input_path]
    logger.info(f"Processing {len(input_files)} files...")
    
    with torch.no_grad():
        for input_file in input_files:
            try:
                logger.info(f"Processing: {input_file}")
                
                # Load and preprocess image using transforms
                img = transforms(input_file)
                
                # Add batch dimension and move to device
                img = img.unsqueeze(0).to(device)
                logger.info(f"Input tensor shape: {img.shape}")
                
                # Run inference
                output = model(img)
                logger.info(f"Model output shape: {output.shape}")
                
                # Apply appropriate post-processing based on number of output channels
                num_output_channels = output.shape[1]
                if num_output_channels == 1:
                    # Binary segmentation
                    output_soft = torch.sigmoid(output)  # Soft predictions
                    pred = (output_soft > threshold).float()   # Hard predictions
                    logger.info(f"Applied binary segmentation post-processing (sigmoid + threshold)")
                else:
                    # Multi-class semantic segmentation
                    output = torch.softmax(output, dim=1)
                    pred = torch.argmax(output, dim=1, keepdim=True).float()
                    logger.info(f"Applied multi-class segmentation post-processing (softmax + argmax) for {num_output_channels} classes")
                
                # Save prediction - ensure correct tensor dimensions
                pred_np = pred.squeeze().cpu().numpy()
                # Ensure 2D array (height, width) for proper image creation
                if pred_np.ndim == 3 and pred_np.shape[0] == 1:
                    pred_np = pred_np.squeeze(0)
                
                # Debug: Log prediction statistics
                logger.info(f"Prediction stats - Shape: {pred_np.shape}, Min: {pred_np.min():.3f}, Max: {pred_np.max():.3f}, Mean: {pred_np.mean():.3f}")
                
                # Convert to visible image - if prediction is all zeros, create a test pattern
                if pred_np.max() == 0:
                    logger.warning("Prediction is all zeros - no segmentation detected")
                    # Create a semi-transparent overlay to show the model ran
                    pred_image = np.zeros_like(pred_np, dtype=np.uint8)
                    # Add a small indicator that inference ran but found no segments
                    pred_image[10:30, 10:30] = 128  # Small gray square as indicator
                else:
                    pred_image = (pred_np * 255).astype(np.uint8)
                    logger.info(f"Segmentation detected - {np.sum(pred_np > 0)} pixels")
                
                output_filename = os.path.join(output_dir, f"pred_{os.path.basename(input_file)}")
                
                # Create a side-by-side comparison with consistent orientation
                # Both model input and display now use the same PIL orientation
                display_input = get_display_oriented_image(input_file, (pred_image.shape[1], pred_image.shape[0]))
                
                # Create side-by-side comparison
                comparison = np.hstack([display_input, pred_image])
                comparison_img = Image.fromarray(comparison)
                comparison_img.save(output_filename)
                
                # Save the display-oriented input separately for web display consistency
                input_only_filename = os.path.join(output_dir, f"input_{os.path.basename(input_file)}")
                input_only_img = Image.fromarray(display_input)
                input_only_img.save(input_only_filename)
                
                # Save prediction only
                pred_only_filename = os.path.join(output_dir, f"pred_only_{os.path.basename(input_file)}")
                pred_only_img = Image.fromarray(pred_image)
                pred_only_img.save(pred_only_filename)
                
                logger.info(f"Saved prediction to {output_filename}")
                
                # Enhanced MLflow artifact logging for predictions
                try:
                    if mlflow.active_run():
                        # Log all prediction outputs with organized structure - get model_dir from context if available
                        current_model_dir = globals().get('model_dir') if 'model_dir' in globals() else None
                        log_artifact_to_model_directory(output_filename, current_model_dir, "predictions/comparisons")
                        log_artifact_to_model_directory(input_only_filename, current_model_dir, "predictions/inputs")
                        log_artifact_to_model_directory(pred_only_filename, current_model_dir, "predictions/outputs")
                        
                        # Create and log prediction metadata
                        prediction_metadata = {
                            'input_file': os.path.basename(input_file),
                            'prediction_timestamp': time.time(),
                            'device_used': str(device),
                            'model_type': model_type,
                            'crop_size': crop_size,
                            'input_shape': img.shape,
                            'output_shape': pred.shape,
                            'prediction_threshold': threshold,
                            'files_generated': {
                                'comparison': os.path.basename(output_filename),
                                'input_only': os.path.basename(input_only_filename),
                                'prediction_only': os.path.basename(pred_only_filename)
                            }
                        }
                        
                        # Save metadata file
                        metadata_filename = os.path.join(output_dir, f"metadata_{os.path.splitext(os.path.basename(input_file))[0]}.json")
                        with open(metadata_filename, 'w') as f:
                            json.dump(prediction_metadata, f, indent=2, default=str)
                        
                        log_artifact_to_model_directory(metadata_filename, current_model_dir, "predictions/metadata")
                        
                        logger.info(f"[MLFLOW] Logged prediction artifacts for {os.path.basename(input_file)}")
                        
                except Exception as e:
                    logger.warning(f"Failed to log prediction to MLflow: {e}")
                    
            except Exception as e:
                logger.error(f"Failed to process {input_file}: {e}")
                continue
    
    # Return inference results summary
    results = {
        'processed_files': len(input_files),
        'output_dir': output_dir,
        'generated_files': []
    }
    
    # Collect information about generated files
    if os.path.exists(output_dir):
        for input_file in input_files:
            basename = os.path.basename(input_file)
            file_info = {
                'input_file': basename,
                'comparison_file': f"pred_{basename}",
                'input_only_file': f"input_{basename}",
                'prediction_only_file': f"pred_only_{basename}",
                'metadata_file': f"metadata_{os.path.splitext(basename)[0]}.json"
            }
            results['generated_files'].append(file_info)
    
    logger.info(f"Inference completed. Generated {len(results['generated_files'])} sets of output files.")
    return results

def inference_mode(args):
    """Run the model in inference mode with enhanced MLflow logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    
    inference_start_time = time.time()
    
    if args.mlflow_run_id:
        mlflow.start_run(run_id=args.mlflow_run_id)
        
        # Log inference parameters
        mlflow.log_param("inference_mode", True)
        mlflow.log_param("input_path", args.input_path)
        mlflow.log_param("output_dir", args.output_dir)
        mlflow.log_param("device", args.device)
        mlflow.log_param("model_type", getattr(args, 'model_type', 'unet'))
        mlflow.log_param("resolution", getattr(args, 'resolution', 'original'))
        
        # Set inference tags
        mlflow.set_tag("task", "inference")
        mlflow.set_tag("mode", "prediction")
        
    try:
        run_inference(
            model_path=args.model_path,
            input_path=args.input_path,
            output_dir=args.output_dir,
            device=args.device,
            weights_path=getattr(args, 'weights_path', None),
            model_type=getattr(args, 'model_type', 'unet'),
            crop_size=getattr(args, 'crop_size', 128),
            threshold=getattr(args, 'threshold', 0.5)
        )
        
        # Log inference summary if MLflow run is active
        if args.mlflow_run_id and mlflow.active_run():
            inference_duration = time.time() - inference_start_time
            
            # Count input files
            if os.path.isdir(args.input_path):
                input_count = len([f for f in os.listdir(args.input_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))])
            else:
                input_count = 1
            
            # Count output files
            output_count = len([f for f in os.listdir(args.output_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(args.output_dir) else 0
            
            mlflow.log_metric("inference_duration_seconds", inference_duration)
            mlflow.log_metric("input_files_processed", input_count)
            mlflow.log_metric("output_files_generated", output_count)
            mlflow.log_metric("processing_rate_files_per_second", input_count / inference_duration if inference_duration > 0 else 0)
            
            # Create and log inference summary
            inference_summary = {
                'inference_completed': True,
                'duration_seconds': inference_duration,
                'input_files_processed': input_count,
                'output_files_generated': output_count,
                'processing_rate': input_count / inference_duration if inference_duration > 0 else 0,
                'device_used': args.device,
                'model_type': getattr(args, 'model_type', 'unet'),
                'crop_size': getattr(args, 'crop_size', 128),
                'threshold': getattr(args, 'threshold', 0.5)
            }
            
            summary_file = os.path.join(args.output_dir, 'inference_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(inference_summary, f, indent=2, default=str)
            
            if args.mlflow_run_id:
                # Get model_dir from context if available for inference
                current_model_dir = globals().get('model_dir') if 'model_dir' in globals() else None
                log_artifact_to_model_directory(summary_file, current_model_dir, "inference/summary")
            
    finally:
        if args.mlflow_run_id:
            if mlflow.active_run():
                try:
                    # Mark inference as completed
                    mlflow.set_tag("inference_status", "completed")
                    mlflow.set_tag("completion_time", datetime.now().isoformat())
                    logger.info("[MLFLOW] Inference marked as completed")
                except Exception as tag_error:
                    logger.warning(f"[MLFLOW] Error setting completion tags: {tag_error}")
                    
                mlflow.end_run()
                logger.info("[MLFLOW] ✅ MLflow inference run properly ended")
            else:
                logger.warning("[MLFLOW] No active inference run to end")

def main():
    """Main entry point"""
    # Setup signal handlers (must be in main thread)
    setup_signal_handlers()
    
    args = parse_args()
    
    if args.mode == 'train':
        train_model(args)
    else:  # predict mode
        inference_mode(args)

# --- Exception handling for main ---
def cleanup_dataloader_workers():
    """Clean up DataLoader workers to prevent zombie processes"""
    try:
        import torch.multiprocessing as mp
        import torch
        
        # Force cleanup of PyTorch DataLoader workers
        if hasattr(torch.utils.data, '_utils'):
            if hasattr(torch.utils.data._utils, 'worker'):
                # Force cleanup worker processes
                try:
                    torch.utils.data._utils.worker._cleanup_workers()
                except:
                    pass
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logging.info("[CLEANUP] DataLoader workers cleanup completed")
    except Exception as e:
        logging.warning(f"[CLEANUP] Error during DataLoader cleanup: {e}")

def cleanup_multiprocessing_workers():
    """Clean up any remaining multiprocessing workers to prevent zombie processes"""
    try:
        import psutil
        import signal
        current_pid = os.getpid()
        current_process = psutil.Process(current_pid)
        
        # Find all child processes
        children = current_process.children(recursive=True)
        if children:
            logging.info(f"[CLEANUP] Found {len(children)} child processes to clean up")
            
            # First try graceful termination
            for child in children:
                try:
                    if child.is_running():
                        child.terminate()
                        logging.info(f"[CLEANUP] Terminated child process {child.pid}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Wait a bit for graceful termination
            psutil.wait_procs(children, timeout=3)
            
            # Force kill any remaining processes
            for child in children:
                try:
                    if child.is_running():
                        child.kill()
                        logging.info(f"[CLEANUP] Force killed child process {child.pid}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        
        # Clean up multiprocessing resources
        try:
            mp.get_context().shutdown()
        except:
            pass
            
        logging.info("[CLEANUP] Multiprocessing cleanup completed")
    except Exception as e:
        logging.warning(f"[CLEANUP] Error during multiprocessing cleanup: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.exception("Training failed with exception")
        print(f"[Train.py] Exception: {e}")
        raise
    finally:
        cleanup_dataloader_workers()
        cleanup_multiprocessing_workers()
