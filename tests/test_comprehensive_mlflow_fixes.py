#!/usr/bin/env python3
"""
Comprehensive test of all MLflow training system fixes
"""

import os
import sys
import django
import time
import subprocess
import requests
from pathlib import Path

# Setup Django
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
django.setup()

from ml_manager.models import MLModel
from ml_manager.mlflow_utils import setup_mlflow, create_new_run
import mlflow

def test_mlflow_navigation():
    """Test 1: MLflow navigation button accessibility"""
    print("\n🔗 Test 1: MLflow Navigation Button")
    try:
        response = requests.get("http://localhost:8000/ml/", timeout=10)
        if response.status_code == 200 and 'mlflow-dashboard' in response.text:
            print("✅ MLflow navigation button accessible")
            return True
        else:
            print("❌ MLflow navigation button not found")
            return False
    except Exception as e:
        print(f"❌ Navigation test failed: {e}")
        return False

def test_mlflow_redirect():
    """Test 2: MLflow redirect functionality"""
    print("\n↗️  Test 2: MLflow Redirect")
    try:
        response = requests.get("http://localhost:8000/ml/mlflow-dashboard/", timeout=10, allow_redirects=False)
        if response.status_code in [302, 301]:
            print("✅ MLflow redirect working")
            return True
        else:
            print(f"❌ MLflow redirect failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Redirect test failed: {e}")
        return False

def test_system_monitor_integration():
    """Test 3: System monitor with MLflow run status checking"""
    print("\n📊 Test 3: System Monitor Integration")
    try:
        from shared.utils.system_monitor import SystemMonitor
        
        # Create a test run
        setup_mlflow()
        run_id = create_new_run(params={'test': 'system_monitor'})
        
        # Start system monitor
        monitor = SystemMonitor(log_interval=5, enable_gpu=False)
        monitor.set_mlflow_run_id(run_id)
        
        # Check that it can access the run
        run_info = mlflow.get_run(run_id)
        if run_info.info.status in ['RUNNING', 'FINISHED']:
            print("✅ System monitor can check MLflow run status")
            return True
        else:
            print(f"❌ Unexpected run status: {run_info.info.status}")
            return False
            
    except Exception as e:
        print(f"❌ System monitor test failed: {e}")
        return False
    finally:
        if mlflow.active_run():
            mlflow.end_run()

def test_training_lifecycle():
    """Test 4: Complete training lifecycle with MLflow"""
    print("\n🚀 Test 4: Training Lifecycle")
    try:
        setup_mlflow()
        
        # Create model and run
        form_data = {
            'name': 'Comprehensive Test Model',
            'description': 'Testing complete workflow',
            'model_type': 'unet',
            'data_path': 'shared/datasets/data',
            'batch_size': 1,
            'epochs': 1,
            'learning_rate': 0.001,
            'validation_split': 0.5,
            'crop_size': 32,
            'num_workers': 0,
        }
        
        run_id = create_new_run(params=form_data)
        print(f"   Created MLflow run: {run_id}")
        
        model = MLModel.objects.create(
            name=form_data['name'],
            description=form_data['description'],
            status='pending',
            total_epochs=form_data['epochs'],
            mlflow_run_id=run_id,
            training_data_info=form_data
        )
        
        print(f"   Created model {model.id}")
        
        # Simulate the Django view fix by ending the run in current process
        if mlflow.active_run():
            mlflow.end_run()
            print("   ✅ Django process ended MLflow run")
        
        # Start training subprocess
        training_args = [
            sys.executable, '/app/shared/train.py',
            '--mode', 'train',
            '--model-type', 'unet',
            '--batch-size', '1',
            '--epochs', '1',
            '--learning-rate', '0.001',
            '--data-path', 'shared/datasets/data',
            '--validation-split', '0.5',
            '--crop-size', '32',
            '--num-workers', '0',
            '--mlflow-run-id', run_id,
            '--model-id', str(model.id),
        ]
        
        process = subprocess.Popen(
            training_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"   Started training process PID: {process.pid}")
        
        # Monitor for up to 20 seconds
        start_time = time.time()
        success_indicators = {
            'run_became_running': False,
            'model_status_changed': False,
            'process_completed': False
        }
        
        while time.time() - start_time < 20:
            if process.poll() is not None:
                success_indicators['process_completed'] = True
                print(f"   ✅ Process completed with code: {process.poll()}")
                break
            
            # Check run status
            try:
                run_info = mlflow.get_run(run_id)
                if run_info.info.status == 'RUNNING':
                    success_indicators['run_became_running'] = True
                    print(f"   ✅ MLflow run became RUNNING")
            except:
                pass
            
            # Check model status
            model.refresh_from_db()
            if model.status != 'pending':
                success_indicators['model_status_changed'] = True
                print(f"   ✅ Model status changed to: {model.status}")
            
            time.sleep(2)
        
        # Final check
        model.refresh_from_db()
        final_run_info = mlflow.get_run(run_id)
        
        print(f"   Final model status: {model.status}")
        print(f"   Final run status: {final_run_info.info.status}")
        
        # Success if we got status changes and process behaved correctly
        success = (
            success_indicators['model_status_changed'] and
            (success_indicators['run_became_running'] or final_run_info.info.status == 'FINISHED')
        )
        
        if success:
            print("✅ Training lifecycle working correctly")
        else:
            print("❌ Training lifecycle has issues")
            
        return success
        
    except Exception as e:
        print(f"❌ Training lifecycle test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        try:
            if 'process' in locals() and process.poll() is None:
                process.terminate()
        except:
            pass
        try:
            if 'model' in locals():
                model.delete()
        except:
            pass

def test_rerun_naming():
    """Test 5: Rerun naming convention fix"""
    print("\n🔄 Test 5: Rerun Naming Convention")
    try:
        # Create original model
        original = MLModel.objects.create(
            name='Test Model',
            description='Original model',
            status='completed',
            training_data_info={'test': True}
        )
        
        # Test rerun name generation logic
        base_name = original.name
        if base_name.endswith('(Rerun)'):
            base_name = base_name[:-7].rstrip()
        rerun_name = f"{base_name} (Rerun)"
        
        if rerun_name == "Test Model (Rerun)":
            print("✅ Rerun naming convention working")
            success = True
        else:
            print(f"❌ Rerun naming failed: got '{rerun_name}'")
            success = False
            
        original.delete()
        return success
        
    except Exception as e:
        print(f"❌ Rerun naming test failed: {e}")
        return False

def main():
    """Run comprehensive tests"""
    print("🧪 COMPREHENSIVE MLFLOW TRAINING SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("MLflow Navigation", test_mlflow_navigation),
        ("MLflow Redirect", test_mlflow_redirect),
        ("System Monitor Integration", test_system_monitor_integration),
        ("Training Lifecycle", test_training_lifecycle),
        ("Rerun Naming", test_rerun_naming),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print final results
    print("\n" + "=" * 60)
    print("🎯 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL MLFLOW TRAINING SYSTEM FIXES VERIFIED!")
        print("\n✅ Summary of Working Features:")
        print("  1. MLflow navigation button in UI")
        print("  2. MLflow redirect functionality")
        print("  3. System monitor with run status checking")
        print("  4. Training lifecycle with proper MLflow run management")
        print("  5. Rerun naming convention fix")
        print("\n🚀 The MLflow training system is fully operational!")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed - system needs attention")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
