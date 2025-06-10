#!/usr/bin/env python3
"""
Minimal Container MLflow Test
"""

import os
import sys
import tempfile

# Add the project root to the path
sys.path.insert(0, '/app' if os.path.exists('/app') else '.')

import mlflow
from shared.utils.mlflow_artifact_manager import log_epoch_artifacts

def minimal_test():
    """Minimal test of artifact logging in container"""
    print("🚀 Minimal Container MLflow Test")
    
    # Use internal container networking
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # Create a simple test file
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "test.txt")
    with open(test_file, 'w') as f:
        f.write("Container test artifact")
    
    try:
        with mlflow.start_run(run_name="Container_Test") as run:
            print(f"Started run: {run.info.run_id}")
            
            logged_paths = log_epoch_artifacts(
                epoch=1,
                metrics={'test': 0.95},
                artifacts={'logs': test_file}
            )
            
            print(f"Logged {len(logged_paths)} artifacts")
            for k, v in logged_paths.items():
                print(f"  {k}: {v}")
            
        print("✅ Container test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    success = minimal_test()
    exit(0 if success else 1)
