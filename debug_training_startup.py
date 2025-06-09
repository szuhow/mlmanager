#!/usr/bin/env python3
"""
Debug script to identify and fix the training startup issue
"""

import os
import sys
import django
import logging
import traceback
from pathlib import Path

# Setup Django environment
project_root = Path(__file__).parent
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
django.setup()

from ml_manager.models import MLModel
from ml_manager.forms import TrainingForm
from ml_manager.views import StartTrainingView
from django.test import RequestFactory
from django.contrib.auth.models import User
from django.contrib.messages.storage.fallback import FallbackStorage
from django.contrib.sessions.backends.db import SessionStore

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_training_startup():
    """Debug the training startup issue step by step"""
    
    print("🔍 DEBUGGING TRAINING STARTUP ISSUE")
    print("=" * 60)
    
    try:
        # Step 1: Test form validation
        print("\n📝 Step 1: Testing form validation...")
        
        form_data = {
            'name': 'Debug Test Model',
            'description': 'Testing training startup issue',
            'model_type': 'unet',
            'data_path': 'shared/datasets/data',
            'batch_size': 2,
            'epochs': 1,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'resolution': '256',
            'device': 'auto',
            'use_random_flip': True,
            'use_random_rotate': True,
            'use_random_scale': True,
            'use_random_intensity': True,
            'crop_size': 64,
            'num_workers': 0,
        }
        
        form = TrainingForm(data=form_data)
        
        if form.is_valid():
            print("✅ Form validation PASSED")
            print(f"📊 Cleaned data: {form.cleaned_data}")
        else:
            print("❌ Form validation FAILED")
            print(f"🔍 Form errors: {form.errors}")
            return False
        
        # Step 2: Test view logic
        print("\n🎯 Step 2: Testing view logic...")
        
        # Create a mock request
        factory = RequestFactory()
        request = factory.post('/ml/start-training/', data=form_data)
        
        # Create a test user and add to request
        user, created = User.objects.get_or_create(username='test_user', defaults={'email': 'test@example.com'})
        request.user = user
        
        # Add session and messages
        request.session = SessionStore()
        request._messages = FallbackStorage(request)
        
        # Create view instance
        view = StartTrainingView()
        view.setup(request)
        
        # Test the form_valid method directly
        print("🚀 Calling form_valid method...")
        
        try:
            # This should trigger the actual training startup logic
            response = view.form_valid(form)
            
            # Check if a model was created
            latest_model = MLModel.objects.filter(name='Debug Test Model').last()
            
            if latest_model:
                print(f"✅ Model created successfully:")
                print(f"   - ID: {latest_model.id}")
                print(f"   - Name: {latest_model.name}")
                print(f"   - Status: {latest_model.status}")
                print(f"   - MLflow Run ID: {latest_model.mlflow_run_id}")
                
                # Check if model status changes (indicating subprocess started)
                import time
                print(f"\n⏱️  Monitoring status changes for 10 seconds...")
                
                initial_status = latest_model.status
                for i in range(10):
                    time.sleep(1)
                    latest_model.refresh_from_db()
                    current_status = latest_model.status
                    if current_status != initial_status:
                        print(f"✅ Status changed: {initial_status} → {current_status}")
                        break
                    print(f"   {i+1}s: Status = {current_status}")
                else:
                    print(f"⚠️  Status remained: {latest_model.status}")
                
                # Cleanup
                latest_model.delete()
                print("🧹 Cleaned up test model")
                
                return True
            else:
                print("❌ No model was created")
                return False
                
        except Exception as e:
            print(f"❌ Error in form_valid: {e}")
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        traceback.print_exc()
        return False

def check_subprocess_environment():
    """Check if the subprocess environment is set up correctly"""
    
    print("\n🔧 CHECKING SUBPROCESS ENVIRONMENT")
    print("=" * 50)
    
    # Check if train.py exists
    train_script = project_root / 'shared' / 'train.py'
    print(f"📁 Train script path: {train_script}")
    print(f"📁 Train script exists: {train_script.exists()}")
    
    if train_script.exists():
        print("✅ Train script found")
    else:
        print("❌ Train script NOT found")
        return False
    
    # Check if data directory exists
    data_dir = project_root / 'shared' / 'datasets' / 'data'
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Data directory exists: {data_dir.exists()}")
    
    if data_dir.exists():
        print("✅ Data directory found")
        # List some contents
        try:
            contents = list(data_dir.iterdir())[:5]  # First 5 items
            print(f"📋 Sample contents: {[item.name for item in contents]}")
        except:
            pass
    else:
        print("❌ Data directory NOT found")
    
    # Check Python path and imports
    print(f"\n🐍 Python executable: {sys.executable}")
    print(f"🐍 Python path: {sys.path[:3]}...")  # First 3 paths
    
    try:
        # Test if we can import required modules
        from shared.train import main as train_main
        print("✅ Can import shared.train")
        
        from ml_manager.mlflow_utils import setup_mlflow
        print("✅ Can import mlflow_utils")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_direct_subprocess_call():
    """Test calling the training subprocess directly"""
    
    print("\n🚀 TESTING DIRECT SUBPROCESS CALL")
    print("=" * 50)
    
    import subprocess
    
    # Create a minimal test model for subprocess testing
    from ml_manager.mlflow_utils import setup_mlflow, create_new_run
    
    try:
        setup_mlflow()
        run_id = create_new_run(params={'test': 'direct_subprocess'})
        
        model = MLModel.objects.create(
            name='Direct Subprocess Test',
            description='Testing direct subprocess call',
            status='pending',
            current_epoch=0,
            total_epochs=1,
            mlflow_run_id=run_id,
            training_data_info={'test': True}
        )
        
        print(f"✅ Created test model: {model.id}")
        
        # Create the exact command from StartTrainingView
        command = [
            sys.executable, 
            str(project_root / 'shared' / 'train.py'),
            '--mode=train',
            f'--model-id={model.id}',
            f'--mlflow-run-id={run_id}',
            '--model-type=unet', 
            '--data-path=shared/datasets/data',
            '--batch-size=2',
            '--epochs=1',
            '--learning-rate=0.001',
            '--validation-split=0.2',
            '--resolution=256',
            '--device=auto',
            '--crop-size=64',
            '--num-workers=0',
            '--random-flip',
            '--random-rotate'
        ]
        
        print(f"🎯 Command: {' '.join(command)}")
        
        # Setup environment like StartTrainingView
        current_env = os.environ.copy()
        ml_manager_path = str(project_root / "ml_manager")
        shared_path = str(project_root / "shared")
        
        existing_python_path = current_env.get("PYTHONPATH")
        new_paths = [ml_manager_path, shared_path]
        if existing_python_path:
            current_env["PYTHONPATH"] = ":".join(new_paths) + ":" + existing_python_path
        else:
            current_env["PYTHONPATH"] = ":".join(new_paths)
            
        current_env["DJANGO_SETTINGS_MODULE"] = "coronary_experiments.settings"
        
        print(f"🌍 PYTHONPATH: {current_env['PYTHONPATH']}")
        print(f"🌍 Working dir: {project_root}")
        
        # Start the subprocess
        print("🚀 Starting subprocess...")
        process = subprocess.Popen(
            command,
            shell=False, 
            env=current_env,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"✅ Process started with PID: {process.pid}")
        
        # Wait a bit and check status
        import time
        time.sleep(5)
        
        # Check if process is still running
        poll_result = process.poll()
        if poll_result is None:
            print("✅ Process is still running")
            
            # Check model status
            model.refresh_from_db()
            print(f"📊 Model status: {model.status}")
            
            # Terminate process
            process.terminate()
            process.wait()
            print("🛑 Process terminated")
            
        else:
            print(f"❌ Process exited with code: {poll_result}")
            
            # Get output
            stdout, stderr = process.communicate()
            if stdout:
                print(f"📋 STDOUT:\n{stdout}")
            if stderr:
                print(f"📋 STDERR:\n{stderr}")
        
        # Cleanup
        model.delete()
        print("🧹 Cleaned up test model")
        
        return poll_result is None
        
    except Exception as e:
        print(f"❌ Direct subprocess test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 COMPREHENSIVE TRAINING STARTUP DEBUG")
    print("=" * 60)
    
    # Run all debug tests
    env_ok = check_subprocess_environment()
    form_ok = debug_training_startup()
    subprocess_ok = test_direct_subprocess_call()
    
    print("\n" + "=" * 60)
    print("📊 DEBUG RESULTS:")
    print(f"   Environment setup: {'✅ OK' if env_ok else '❌ FAILED'}")
    print(f"   Form validation:   {'✅ OK' if form_ok else '❌ FAILED'}")
    print(f"   Subprocess call:   {'✅ OK' if subprocess_ok else '❌ FAILED'}")
    
    if all([env_ok, form_ok, subprocess_ok]):
        print("\n🎉 ALL TESTS PASSED - Training startup should work!")
    else:
        print("\n🔧 ISSUES FOUND - Training startup needs fixes")
    
    print("=" * 60)
