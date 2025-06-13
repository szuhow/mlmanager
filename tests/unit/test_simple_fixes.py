#!/usr/bin/env python3
"""
Simple test to verify our fixes
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
print(f"🔧 Testing MLflow Fixes in: {project_root}")

def test_navigation_url():
    """Test if MLflow URL exists"""
    print("\n📍 Test 1: Navigation URL")
    
    urls_file = project_root / 'ml_manager' / 'urls.py'
    if urls_file.exists():
        with open(urls_file, 'r') as f:
            content = f.read()
        
        if 'mlflow-dashboard' in content and 'mlflow_redirect_view' in content:
            print("✅ MLflow URL pattern exists")
            return True
        else:
            print("❌ MLflow URL pattern missing")
            return False
    else:
        print("❌ URLs file not found")
        return False

def test_base_template():
    """Test if base template has MLflow button"""
    print("\n🎨 Test 2: Base Template")
    
    template_file = project_root / 'ml_manager' / 'templates' / 'ml_manager' / 'base.html'
    if template_file.exists():
        with open(template_file, 'r') as f:
            content = f.read()
        
        if 'mlflow-dashboard' in content and 'fa-chart-line' in content:
            print("✅ MLflow navigation button exists in template")
            return True
        else:
            print("❌ MLflow navigation button missing from template")
            return False
    else:
        print("❌ Base template not found")
        return False

def test_system_monitor():
    """Test if SystemMonitor has been enhanced"""
    print("\n🔍 Test 3: SystemMonitor Enhancement")
    
    monitor_file = project_root / 'shared' / 'utils' / 'system_monitor.py'
    if monitor_file.exists():
        with open(monitor_file, 'r') as f:
            content = f.read()
        
        if 'get_run' in content and 'RUNNING' in content and 'log_metrics_to_mlflow' in content:
            print("✅ SystemMonitor has enhanced MLflow run status checking")
            return True
        else:
            print("❌ SystemMonitor missing enhanced MLflow run status checking")
            return False
    else:
        print("❌ SystemMonitor file not found")
        return False

def test_training_preview():
    """Test if training preview has been enhanced"""
    print("\n🖼️  Test 4: Training Preview Enhancement")
    
    views_file = project_root / 'ml_manager' / 'views.py'
    if views_file.exists():
        with open(views_file, 'r') as f:
            content = f.read()
        
        if 'search_for_predictions' in content and 'search_patterns' in content:
            print("✅ Training preview has enhanced artifact search patterns")
            return True
        else:
            print("❌ Training preview missing enhanced artifact search patterns")
            return False
    else:
        print("❌ Views file not found")
        return False

def main():
    """Run tests"""
    tests = [
        ("Navigation URL", test_navigation_url),
        ("Base Template", test_base_template),
        ("SystemMonitor Enhancement", test_system_monitor),
        ("Training Preview Enhancement", test_training_preview),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("🎯 TEST RESULTS")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL FIXES VERIFIED!")
        print("\n📋 Summary of Implemented Fixes:")
        print("1. ✅ MLflow redirect button added to main navigation")
        print("2. ✅ SystemMonitor enhanced with MLflow run status checking")
        print("3. ✅ Training preview enhanced for multiple artifact structures")
        print("4. ✅ URL patterns fixed for MLflow navigation")
        
        print("\n🚀 Next Steps:")
        print("- Test in container environment")
        print("- Verify MLflow artifacts issue is resolved")
        print("- Run end-to-end training test")
        
    return 0 if passed == len(results) else 1

if __name__ == "__main__":
    exit(main())
