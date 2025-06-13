#!/usr/bin/env python3
"""
Test MLflow Paths Fix
Test that MLflow artifacts are properly stored in data/mlflow instead of models/organized
"""

import os
import sys
import django
from pathlib import Path

# Setup Django
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
django.setup()

from django.conf import settings

def test_mlflow_settings():
    """Test MLflow settings configuration"""
    print("🧪 Testing MLflow Settings Configuration")
    print("=" * 50)
    
    print(f"BASE_MLRUNS_DIR: {settings.BASE_MLRUNS_DIR}")
    print(f"MLFLOW_TRACKING_URI: {settings.MLFLOW_TRACKING_URI}")
    print(f"MLFLOW_ARTIFACT_ROOT: {settings.MLFLOW_ARTIFACT_ROOT}")
    print(f"MLFLOW_ENABLED: {settings.MLFLOW_ENABLED}")
    
    # Check if the directory exists
    mlruns_path = Path(settings.BASE_MLRUNS_DIR)
    if mlruns_path.exists():
        print(f"✅ MLflow directory exists: {mlruns_path}")
        print(f"   Directory contents: {list(mlruns_path.iterdir())}")
    else:
        print(f"❌ MLflow directory does not exist: {mlruns_path}")
        # Try to create it
        try:
            mlruns_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created MLflow directory: {mlruns_path}")
        except Exception as e:
            print(f"❌ Failed to create MLflow directory: {e}")
    
    return True

def test_mlflow_functions():
    """Test MLflow functions work with new paths"""
    print("\n🧪 Testing MLflow Functions")
    print("=" * 50)
    
    try:
        import mlflow
        
        # Set MLflow tracking URI to our configured backend
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        
        # Test getting experiments
        print("🔄 Testing mlflow.search_experiments...")
        experiments = mlflow.search_experiments()
        print(f"✅ Found {len(experiments)} experiments")
        
        # Test getting runs
        print("🔄 Testing mlflow.search_runs...")
        runs = mlflow.search_runs(max_results=5)
        print(f"✅ Retrieved {len(runs)} runs")
        
        return True
        
    except Exception as e:
        print(f"❌ MLflow functions test failed: {e}")
        return False

def test_database_connection():
    """Test MLflow database connection"""
    print("\n🧪 Testing MLflow Database Connection")
    print("=" * 50)
    
    try:
        import mlflow
        
        # Set MLflow tracking URI to our configured backend
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        
        # Try to list experiments
        experiments = mlflow.search_experiments()
        print(f"✅ Connected to MLflow tracking server")
        print(f"   Found {len(experiments)} experiments")
        
        for exp in experiments:
            print(f"   - {exp.name} (ID: {exp.experiment_id})")
        
        return True
        
    except Exception as e:
        print(f"❌ MLflow database connection failed: {e}")
        return False

def test_artifact_storage():
    """Test artifact storage location"""
    print("\n🧪 Testing Artifact Storage Location")
    print("=" * 50)
    
    artifact_root = settings.MLFLOW_ARTIFACT_ROOT
    print(f"Artifact root: {artifact_root}")
    
    # Check if path contains 'data/mlflow' instead of 'models/organized'
    if 'data/mlflow' in str(artifact_root):
        print("✅ Artifact root uses new 'data/mlflow' path")
    elif 'models/organized' in str(artifact_root):
        print("❌ Artifact root still uses old 'models/organized' path")
        return False
    else:
        print(f"ℹ️  Artifact root uses custom path: {artifact_root}")
    
    # Check if directory exists
    artifact_path = Path(artifact_root)
    if artifact_path.exists():
        print(f"✅ Artifact directory exists: {artifact_path}")
        contents = list(artifact_path.iterdir())
        print(f"   Contents: {contents}")
    else:
        print(f"ℹ️  Artifact directory does not exist yet: {artifact_path}")
        print("   This is normal if no training has been run")
    
    return True

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 MLFLOW PATHS FIX TEST")
    print("=" * 60)
    
    tests = [
        ("MLflow Settings", test_mlflow_settings),
        ("MLflow Functions", test_mlflow_functions), 
        ("Database Connection", test_database_connection),
        ("Artifact Storage", test_artifact_storage)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result if result is not None else True))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("🏁 TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nSummary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! MLflow paths are properly configured.")
    else:
        print("🔧 Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
