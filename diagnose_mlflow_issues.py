#!/usr/bin/env python3
"""
Diagnose MLflow training issues
"""

import os
import sys
import tempfile
import json
import time

# Add the project root to the path
sys.path.insert(0, '/app' if os.path.exists('/app') else '.')

import mlflow

def check_mlflow_runs():
    """Check current MLflow runs and their status"""
    print("🔍 Checking MLflow Runs")
    print("=" * 50)
    
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        
        # Get all experiments
        experiments = mlflow.search_experiments()
        print(f"📊 Found {len(experiments)} experiments")
        
        for exp in experiments:
            print(f"🧪 Experiment: {exp.name} (ID: {exp.experiment_id})")
            
            # Get runs for this experiment
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=10)
            print(f"   📝 Runs: {len(runs)}")
            
            for i, run in runs.iterrows():
                run_id = run['run_id']
                run_name = run.get('tags.mlflow.runName', 'Unnamed')
                status = run['status']
                start_time = run.get('start_time', 0)
                end_time = run.get('end_time', 0)
                
                # Convert timestamps
                if start_time:
                    start_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time/1000))
                else:
                    start_str = "Unknown"
                
                if end_time:
                    end_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time/1000))
                    duration = (end_time - start_time) / 1000 if start_time else 0
                else:
                    end_str = "Still running" if status == "RUNNING" else "Unknown"
                    duration = (time.time() * 1000 - start_time) / 1000 if start_time else 0
                
                print(f"   🏃 {run_id[:8]}... - {run_name}")
                print(f"      Status: {status}")
                print(f"      Started: {start_str}")
                print(f"      Ended: {end_str}")
                print(f"      Duration: {duration:.1f}s")
                
                # Check for training-related tags
                training_tags = [col for col in run.index if col.startswith('tags.') and any(keyword in col.lower() for keyword in ['train', 'model', 'epoch'])]
                if training_tags:
                    print(f"      Training tags: {training_tags}")
                
                # Check for metrics
                metric_cols = [col for col in run.index if col.startswith('metrics.')]
                if metric_cols:
                    print(f"      Metrics: {len(metric_cols)} ({metric_cols[:3]}...)")
                
                print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking MLflow runs: {e}")
        return False

def check_for_running_training():
    """Check for signs of running training processes"""
    print("🔍 Checking for Running Training Processes")
    print("=" * 50)
    
    try:
        # Check for training logs
        log_locations = [
            '/app/models/artifacts/training.log',
            '/app/models/artifacts/',
            '/app/models/',
            '/app/mlruns/',
            '/mlflow/artifacts/'
        ]
        
        for location in log_locations:
            try:
                if os.path.exists(location):
                    if os.path.isfile(location):
                        # Check file modification time
                        mtime = os.path.getmtime(location)
                        age = time.time() - mtime
                        print(f"📄 {location} - Modified {age:.1f}s ago")
                        
                        # Check last few lines
                        try:
                            with open(location, 'r') as f:
                                lines = f.readlines()
                                if lines:
                                    print(f"   Last line: {lines[-1].strip()}")
                        except:
                            pass
                    else:
                        # Directory - check contents
                        try:
                            contents = os.listdir(location)
                            print(f"📁 {location} - Contains: {len(contents)} items")
                            if contents:
                                print(f"   Items: {contents[:5]}")
                        except:
                            pass
                else:
                    print(f"❌ {location} - Not found")
            except Exception as e:
                print(f"❌ Error checking {location}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking training processes: {e}")
        return False

def check_image_generation_issue():
    """Check why training images are not being generated"""
    print("🔍 Checking Image Generation Issues")
    print("=" * 50)
    
    try:
        # Test image generation capabilities
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        
        print("✅ Matplotlib available with Agg backend")
        
        # Test basic plot generation
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title('Test Plot')
        
        # Save to temporary location
        temp_dir = tempfile.mkdtemp()
        test_image = os.path.join(temp_dir, 'test_plot.png')
        plt.savefig(test_image, dpi=150, bbox_inches='tight')
        plt.close()
        
        if os.path.exists(test_image):
            size = os.path.getsize(test_image)
            print(f"✅ Test image generated: {test_image} ({size} bytes)")
        else:
            print("❌ Test image generation failed")
        
        # Check training script image generation functions
        try:
            from shared.train import save_sample_predictions, save_training_curves
            print("✅ Training image functions imported successfully")
        except Exception as e:
            print(f"❌ Error importing training image functions: {e}")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking image generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_active_mlflow_runs():
    """Check for any active MLflow runs"""
    print("🔍 Checking for Active MLflow Runs")
    print("=" * 50)
    
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        
        # Try to get the active run
        try:
            active_run = mlflow.active_run()
            if active_run:
                print(f"🟢 Active run found: {active_run.info.run_id}")
                print(f"   Run name: {active_run.data.tags.get('mlflow.runName', 'Unnamed')}")
                print(f"   Status: {active_run.info.status}")
                print(f"   Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(active_run.info.start_time/1000))}")
                return True
            else:
                print("🔴 No active MLflow run found")
        except Exception as e:
            print(f"🔴 No active run: {e}")
        
        # Search for RUNNING status runs
        all_runs = mlflow.search_runs(experiment_ids=['0'], max_results=20)
        running_runs = all_runs[all_runs['status'] == 'RUNNING'] if len(all_runs) > 0 else []
        
        if len(running_runs) > 0:
            print(f"🟢 Found {len(running_runs)} RUNNING runs:")
            for i, run in running_runs.iterrows():
                print(f"   🏃 {run['run_id'][:8]}... - {run.get('tags.mlflow.runName', 'Unnamed')}")
        else:
            print("🔴 No RUNNING status runs found")
        
        return len(running_runs) > 0
        
    except Exception as e:
        print(f"❌ Error checking active runs: {e}")
        return False

def main():
    """Run all diagnostic checks"""
    print("🚀 MLflow Training Issues Diagnostic")
    print("=" * 60)
    
    checks = [
        check_mlflow_runs,
        check_active_mlflow_runs,
        check_for_running_training,
        check_image_generation_issue
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"❌ Check failed: {e}")
            results.append(False)
        print()
    
    print("=" * 60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if all(results):
        print("✅ All diagnostic checks passed")
    else:
        print("⚠️  Some issues found:")
        if not results[0]:
            print("   ❌ MLflow runs check failed")
        if not results[1]:
            print("   ❌ No active MLflow runs found")
        if not results[2]:
            print("   ❌ Training process check failed")
        if not results[3]:
            print("   ❌ Image generation check failed")
    
    print("\n🔧 RECOMMENDATIONS:")
    if not results[1]:
        print("   • Start a new training run with proper MLflow tracking")
        print("   • Ensure mlflow.start_run() is called before training")
    if not results[3]:
        print("   • Check matplotlib configuration for image generation")
        print("   • Verify training script image logging functions")
    
    return 0 if all(results) else 1

if __name__ == "__main__":
    exit(main())
