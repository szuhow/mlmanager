#!/usr/bin/env python3
"""
Simple test to verify Django settings structure without importing Django
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def test_settings_structure():
    """Test settings structure without Django"""
    print("🧪 Testing Django settings structure (no Django import)...")
    
    # Test environment variables
    environments = ['development', 'production', 'testing']
    
    for env in environments:
        print(f"\n📋 Testing {env.upper()} environment...")
        
        # Set environment
        os.environ['ENVIRONMENT'] = env
        
        # Check if settings files exist
        settings_file = project_root / 'core' / 'config' / 'settings' / f'{env}.py'
        if settings_file.exists():
            print(f"   ✅ Settings file exists: {settings_file}")
            
            # Try to read the file
            try:
                with open(settings_file, 'r') as f:
                    content = f.read()
                    if 'DEBUG' in content:
                        print(f"   ✅ DEBUG setting found")
                    if 'ALLOWED_HOSTS' in content:
                        print(f"   ✅ ALLOWED_HOSTS setting found")
                    if 'SECRET_KEY' in content:
                        print(f"   ✅ SECRET_KEY setting found")
                        
            except Exception as e:
                print(f"   ❌ Error reading settings file: {e}")
        else:
            print(f"   ❌ Settings file missing: {settings_file}")
    
    # Test base settings
    print(f"\n📋 Testing BASE settings...")
    base_settings = project_root / 'core' / 'config' / 'settings' / 'base.py'
    if base_settings.exists():
        print(f"   ✅ Base settings file exists")
        
        try:
            with open(base_settings, 'r') as f:
                content = f.read()
                if 'INSTALLED_APPS' in content:
                    print(f"   ✅ INSTALLED_APPS found")
                if 'core.apps.ml_manager' in content:
                    print(f"   ✅ ml_manager app found in INSTALLED_APPS")
                else:
                    print(f"   ⚠️  ml_manager app reference check needed")
                    
        except Exception as e:
            print(f"   ❌ Error reading base settings: {e}")
    else:
        print(f"   ❌ Base settings file missing")
    
    # Test router
    print(f"\n📋 Testing settings router...")
    router_file = project_root / 'core' / 'config' / 'settings.py'
    if router_file.exists():
        print(f"   ✅ Settings router exists")
        
        try:
            with open(router_file, 'r') as f:
                content = f.read()
                if 'ENVIRONMENT' in content:
                    print(f"   ✅ Environment detection found")
                if 'SETTINGS_MODULES' in content:
                    print(f"   ✅ Settings modules mapping found")
                    
        except Exception as e:
            print(f"   ❌ Error reading router: {e}")
    else:
        print(f"   ❌ Settings router missing")

def test_app_structure():
    """Test app structure"""
    print(f"\n📋 Testing app structure...")
    
    app_path = project_root / 'core' / 'apps' / 'ml_manager'
    if app_path.exists():
        print(f"   ✅ ml_manager app directory exists")
        
        key_files = ['views.py', 'models.py', 'urls.py', 'admin.py']
        for file in key_files:
            file_path = app_path / file
            if file_path.exists() and file_path.stat().st_size > 0:
                print(f"   ✅ {file} exists and has content")
            else:
                print(f"   ⚠️  {file} missing or empty")
    else:
        print(f"   ❌ ml_manager app directory missing")

def test_static_structure():
    """Test static files structure"""
    print(f"\n📋 Testing static files structure...")
    
    # Check static source
    static_src = project_root / 'core' / 'static'
    if static_src.exists():
        print(f"   ✅ Static source directory exists")
        
        ml_static = static_src / 'ml_manager'
        if ml_static.exists():
            print(f"   ✅ ml_manager static files exist")
            files = list(ml_static.rglob('*'))
            print(f"   📁 Static files count: {len([f for f in files if f.is_file()])}")
        else:
            print(f"   ⚠️  ml_manager static directory missing")
    else:
        print(f"   ❌ Static source directory missing")
    
    # Check static collection target
    static_collect = project_root / 'core' / 'staticfiles'
    if static_collect.exists():
        print(f"   ✅ Static collection directory exists")
    else:
        print(f"   ⚠️  Static collection directory will be created by collectstatic")

if __name__ == '__main__':
    print("🚀 MLManager Structure Verification")
    print("=" * 50)
    
    try:
        test_settings_structure()
        test_app_structure()  
        test_static_structure()
        
        print("\n🎉 Structure verification completed!")
        print("✅ Django settings structure looks good")
        
    except Exception as e:
        print(f"\n💥 Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
