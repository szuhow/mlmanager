#!/usr/bin/env python3
"""
Simple MLflow Artifact Manager Test
Test the enhanced artifact manager with actual MLflow logging
"""

import os
import sys
import tempfile
import json

# Add the project root to the path
sys.path.insert(0, '/app' if os.path.exists('/app') else '.')

import mlflow
from ml.utils.utils.mlflow_artifact_manager import log_epoch_artifacts, log_final_model

def simple_test():
    """Simple test of artifact logging"""
    print("🚀 Simple MLflow Artifact Manager Test")
    print("=" * 50)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    print(f"🔗 MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    
    # Create a simple test file
    temp_dir = tempfile.mkdtemp(prefix="simple_test_")
    test_file = os.path.join(temp_dir, "test_artifact.txt")
    with open(test_file, 'w') as f:
        f.write("This is a test artifact created by the enhanced MLflow artifact manager.\n")
        f.write("Timestamp: 2025-06-10\n")
        f.write("Test: Container Integration\n")
    
    try:
        # Start MLflow run
        with mlflow.start_run(run_name="Simple_Enhanced_Artifact_Test") as run:
            print(f"📝 Started MLflow run: {run.info.run_id}")
            
            # Test metrics
            metrics = {
                'test_metric_1': 0.85,
                'test_metric_2': 0.92,
                'loss': 0.15
            }
            
            # Test artifacts
            artifacts = {
                'logs': test_file
            }
            
            # Test metadata
            metadata = {
                'test_type': 'container_integration',
                'environment': 'docker',
                'enhanced_manager_version': '1.0'
            }
            
            # Log using enhanced artifact manager
            logged_paths = log_epoch_artifacts(
                epoch=1,
                model_state=None,
                metrics=metrics,
                artifacts=artifacts,
                metadata=metadata
            )
            
            print(f"✅ Successfully logged {len(logged_paths)} artifact types:")
            for category, path in logged_paths.items():
                print(f"   📁 {category}: {path}")
            
            print(f"🌐 View in MLflow UI: http://localhost:5000/#/experiments/0/runs/{run.info.run_id}")
            
        print("✅ Simple test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        print("🧹 Cleaned up test files")

def list_recent_runs():
    """List recent MLflow runs to verify logging"""
    print("\n📊 Recent MLflow Runs:")
    print("=" * 30)
    
    try:
        runs = mlflow.search_runs(experiment_ids=['0'], max_results=3)
        if len(runs) > 0:
            for i, run in runs.iterrows():
                print(f"🏃 Run {i+1}: {run['run_id'][:8]}... - {run.get('tags.mlflow.runName', 'Unnamed')}")
                print(f"   📅 Started: {run['start_time']}")
                if 'metrics.test_metric_1' in run:
                    print(f"   📈 Test Metric 1: {run['metrics.test_metric_1']}")
        else:
            print("No runs found")
            
    except Exception as e:
        print(f"❌ Could not list runs: {e}")

if __name__ == "__main__":
    success = simple_test()
    list_recent_runs()
    
    if success:
        print("\n🎉 CONTAINER INTEGRATION TEST PASSED!")
        print("The Enhanced MLflow Artifact Manager works correctly in Docker container!")
    else:
        print("\n❌ CONTAINER INTEGRATION TEST FAILED!")
        exit(1)
