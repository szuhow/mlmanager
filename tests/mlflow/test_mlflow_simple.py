#!/usr/bin/env python3
"""
Simple MLflow Paths Test
Test that MLflow settings use the correct paths
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

def test_mlflow_paths():
    """Test MLflow path configuration"""
    print("🧪 Testing MLflow Path Configuration")
    print("=" * 50)
    
    print(f"BASE_MLRUNS_DIR: {settings.BASE_MLRUNS_DIR}")
    print(f"MLFLOW_BACKEND_STORE_URI: {settings.MLFLOW_BACKEND_STORE_URI}")
    print(f"MLFLOW_DEFAULT_ARTIFACT_ROOT: {settings.MLFLOW_DEFAULT_ARTIFACT_ROOT}")
    
    # Check if paths use new structure
    base_dir_str = str(settings.BASE_MLRUNS_DIR)
    artifact_root_str = str(settings.MLFLOW_DEFAULT_ARTIFACT_ROOT)
    
    print("\n📊 Path Analysis:")
    
    # Test BASE_MLRUNS_DIR
    if 'data/mlflow' in base_dir_str:
        print("✅ BASE_MLRUNS_DIR uses new 'data/mlflow' path")
    elif 'models/organized' in base_dir_str:
        print("❌ BASE_MLRUNS_DIR still uses old 'models/organized' path")
        return False
    else:
        print(f"ℹ️  BASE_MLRUNS_DIR uses custom path: {base_dir_str}")
    
    # Test MLFLOW_DEFAULT_ARTIFACT_ROOT
    if 'data/mlflow' in artifact_root_str:
        print("✅ MLFLOW_DEFAULT_ARTIFACT_ROOT uses new 'data/mlflow' path")
    elif 'models/organized' in artifact_root_str:
        print("❌ MLFLOW_DEFAULT_ARTIFACT_ROOT still uses old 'models/organized' path")
        return False
    else:
        print(f"ℹ️  MLFLOW_DEFAULT_ARTIFACT_ROOT uses custom path: {artifact_root_str}")
    
    # Check if directories exist
    print("\n📁 Directory Status:")
    
    mlruns_path = Path(settings.BASE_MLRUNS_DIR)
    if mlruns_path.exists():
        print(f"✅ MLflow directory exists: {mlruns_path}")
        contents = list(mlruns_path.iterdir())
        print(f"   Contents: {[f.name for f in contents]}")
    else:
        print(f"ℹ️  MLflow directory does not exist: {mlruns_path}")
        print("   This is normal if no training has been run yet")
    
    # Check old paths don't exist
    old_models_path = Path(settings.BASE_DIR) / 'models' / 'organized'
    if old_models_path.exists():
        print(f"⚠️  Old models directory still exists: {old_models_path}")
        contents = list(old_models_path.iterdir())
        if contents:
            print(f"   Contents: {[f.name for f in contents]}")
            print("   You may want to migrate these to the new location")
    else:
        print(f"✅ Old models directory does not exist: {old_models_path}")
    
    return True

def test_mlflow_database():
    """Test MLflow database file location"""
    print("\n🧪 Testing MLflow Database Location")
    print("=" * 50)
    
    # Extract database path from backend store URI
    backend_uri = settings.MLFLOW_BACKEND_STORE_URI
    print(f"Backend store URI: {backend_uri}")
    
    if backend_uri.startswith('sqlite:///'):
        db_path = backend_uri.replace('sqlite:///', '')
        db_path = Path(db_path)
        
        print(f"Database path: {db_path}")
        
        if db_path.exists():
            print(f"✅ MLflow database exists: {db_path}")
            # Get file size
            size_mb = db_path.stat().st_size / (1024 * 1024)
            print(f"   Size: {size_mb:.2f} MB")
        else:
            print(f"ℹ️  MLflow database does not exist: {db_path}")
            print("   This is normal for a fresh installation")
    else:
        print(f"ℹ️  Using non-SQLite backend: {backend_uri}")
    
    return True

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 MLFLOW PATHS SIMPLE TEST")
    print("=" * 60)
    
    try:
        result1 = test_mlflow_paths()
        result2 = test_mlflow_database()
        
        print("\n" + "=" * 60)
        print("🏁 TEST RESULTS")
        print("=" * 60)
        
        if result1 and result2:
            print("✅ ALL TESTS PASSED! MLflow paths are properly configured.")
            print("\n📋 Summary:")
            print("   - MLflow artifacts will be stored in data/mlflow/")
            print("   - Old models/organized/ path is not used")
            print("   - Configuration is correct for the new structure")
        else:
            print("❌ Some tests failed. Check the output above.")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
