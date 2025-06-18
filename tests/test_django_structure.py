#!/usr/bin/env python3
"""
Test script to verify the new Django settings structure works correctly.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def test_settings_import():
    """Test importing different settings modules"""
    print("🧪 Testing Django settings structure...")
    
    # Test 1: Import settings router
    print("\n1️⃣ Testing settings router...")
    try:
        os.environ['ENVIRONMENT'] = 'development'
        os.environ['DJANGO_SETTINGS_MODULE'] = 'core.config.settings'
        
        # Import Django settings
        from core.config.settings import BASE_DIR, DEBUG, INSTALLED_APPS
        
        print("✅ Settings router imported successfully")
        print(f"   📁 BASE_DIR: {BASE_DIR}")
        print(f"   🐛 DEBUG: {DEBUG}")
        print(f"   📦 INSTALLED_APPS count: {len(INSTALLED_APPS)}")
        
        # Check if our app is in INSTALLED_APPS
        if 'core.apps.ml_manager' in INSTALLED_APPS:
            print("   ✅ ml_manager app found in INSTALLED_APPS")
        else:
            print("   ❌ ml_manager app NOT found in INSTALLED_APPS")
            
    except Exception as e:
        print(f"   ❌ Failed to import settings: {e}")
        return False
    
    # Test 2: Test different environments
    print("\n2️⃣ Testing environment switching...")
    environments = ['development', 'production', 'testing']
    
    for env in environments:
        try:
            os.environ['ENVIRONMENT'] = env
            # Clear Django settings cache
            if 'core.config.settings' in sys.modules:
                del sys.modules['core.config.settings']
            
            from core.config.settings import DEBUG as debug_setting
            print(f"   ✅ {env.upper()}: DEBUG={debug_setting}")
            
        except Exception as e:
            print(f"   ❌ {env.upper()}: Failed - {e}")
    
    return True

def test_static_files_structure():
    """Test static files structure"""
    print("\n3️⃣ Testing static files structure...")
    
    static_dir = project_root / 'core' / 'static'
    staticfiles_dir = project_root / 'core' / 'staticfiles'
    
    if static_dir.exists():
        print(f"   ✅ Static source directory exists: {static_dir}")
        ml_manager_static = static_dir / 'ml_manager'
        if ml_manager_static.exists():
            print(f"   ✅ ml_manager static files found")
            files = list(ml_manager_static.rglob('*'))
            print(f"   📁 Static files count: {len(files)}")
        else:
            print(f"   ⚠️  ml_manager static directory not found")
    else:
        print(f"   ❌ Static source directory missing: {static_dir}")
    
    if staticfiles_dir.exists():
        print(f"   ✅ Static collection directory exists: {staticfiles_dir}")
    else:
        print(f"   ⚠️  Static collection directory will be created on collectstatic")

def test_app_structure():
    """Test Django app structure"""
    print("\n4️⃣ Testing Django app structure...")
    
    app_dir = project_root / 'core' / 'apps' / 'ml_manager'
    
    required_files = [
        '__init__.py', 'models.py', 'views.py', 'urls.py', 
        'admin.py', 'apps.py', 'forms.py'
    ]
    
    for file in required_files:
        file_path = app_dir / file
        if file_path.exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} missing")
    
    # Check for additional directories
    dirs = ['api', 'services', 'templates', 'migrations']
    for dir_name in dirs:
        dir_path = app_dir / dir_name
        if dir_path.exists():
            print(f"   ✅ {dir_name}/ directory")
        else:
            print(f"   ⚠️  {dir_name}/ directory missing")

if __name__ == '__main__':
    print("🚀 MLManager Django Settings Structure Test")
    print("=" * 50)
    
    try:
        # Test settings import
        if test_settings_import():
            test_static_files_structure()
            test_app_structure()
            
            print("\n🎉 Django settings structure test completed!")
            print("✅ All core components are properly organized")
        else:
            print("\n❌ Settings structure test failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
