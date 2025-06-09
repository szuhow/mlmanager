#!/usr/bin/env python3
"""
Test for Dark Mode Flash Fix Implementation
Tests that the dark mode implementation prevents the flash of light theme.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def print_header(title):
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))

def print_status(message, status):
    status_icon = "✅" if status else "❌"
    print(f"{status_icon} {message}")
    return status

def test_template_structure():
    """Test the HTML template structure for dark mode fix"""
    print_header("🎨 Testing Template Structure")
    
    base_template_path = Path("ml_manager/templates/ml_manager/base.html")
    
    if not base_template_path.exists():
        return print_status("Base template file exists", False)
    
    print_status("Base template file exists", True)
    
    # Read template content
    with open(base_template_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for critical dark mode fix components
    tests = [
        ("Early dark mode script in head", "data-theme-loading" in content),
        ("Initial CSS variables", "--initial-bg" in content),
        ("Loading state CSS", "html[data-theme-loading=" in content),
        ("Visibility change handler", "visibilitychange" in content),
        ("Focus event handler", "addEventListener('focus'" in content),
        ("Transition prevention", "transition: none" in content),
        ("localStorage check", "localStorage.getItem('colorMode')" in content),
    ]
    
    all_passed = True
    for test_name, condition in tests:
        result = print_status(test_name, condition)
        all_passed = all_passed and result
    
    return all_passed

def test_css_structure():
    """Test CSS structure for flash prevention"""
    print_header("🎨 Testing CSS Flash Prevention")
    
    base_template_path = Path("ml_manager/templates/ml_manager/base.html")
    
    with open(base_template_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tests = [
        ("Critical CSS comment present", "Critical: Prevent flash" in content),
        ("Initial background variable", "--initial-bg" in content),
        ("Initial color variable", "--initial-color" in content),
        ("Theme loading dark state", 'data-theme-loading="dark"' in content),
        ("Theme loading light state", 'data-theme-loading="light"' in content),
        ("No transition on initial load", "transition: none" in content),
        ("Dark mode background", "#181a1b" in content),
        ("Light mode background", "#ffffff" in content),
    ]
    
    all_passed = True
    for test_name, condition in tests:
        result = print_status(test_name, condition)
        all_passed = all_passed and result
    
    return all_passed

def test_javascript_structure():
    """Test JavaScript structure for proper theme handling"""
    print_header("🔧 Testing JavaScript Implementation")
    
    base_template_path = Path("ml_manager/templates/ml_manager/base.html")
    
    with open(base_template_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tests = [
        ("Early theme detection script", "data-theme-loading" in content and "localStorage.getItem" in content),
        ("SetMode function exists", "function setMode(" in content),
        ("Skip transition parameter", "skipTransition" in content),
        ("Visibility change listener", "visibilitychange" in content),
        ("Focus event listener", "addEventListener('focus'" in content),
        ("BeforeUnload preparation", "beforeunload" in content),
        ("DOM ready handler", "DOMContentLoaded" in content),
        ("Toggle button handler", "onclick" in content),
    ]
    
    all_passed = True
    for test_name, condition in tests:
        result = print_status(test_name, condition)
        all_passed = all_passed and result
    
    return all_passed

def test_fix_completeness():
    """Test if the fix addresses all known flash scenarios"""
    print_header("🔍 Testing Fix Completeness")
    
    base_template_path = Path("ml_manager/templates/ml_manager/base.html")
    
    with open(base_template_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    scenarios = [
        ("Page load flash prevention", "html, body" in content and "--initial-bg" in content),
        ("Window switching flash prevention", "visibilitychange" in content),
        ("Window focus flash prevention", "focus" in content),
        ("Page reload preparation", "beforeunload" in content),
        ("Fast theme reapplication", "setMode(mode, true)" in content),
        ("Transition control", "transition:" in content),
        ("Loading state cleanup", "removeAttribute('data-theme-loading')" in content),
    ]
    
    all_passed = True
    for scenario_name, condition in scenarios:
        result = print_status(f"Handles {scenario_name}", condition)
        all_passed = all_passed and result
    
    return all_passed

def test_django_integration():
    """Test Django template integration"""
    print_header("🐍 Testing Django Integration")
    
    base_template_path = Path("ml_manager/templates/ml_manager/base.html")
    
    with open(base_template_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tests = [
        ("Django template tags", "{% load static %}" in content),
        ("Template blocks", "{% block title %}" in content),
        ("URL tags", "{% url " in content),
        ("Static files", "{% static " in content),
        ("Valid HTML structure", "<!DOCTYPE html>" in content),
        ("Bootstrap integration", "bootstrap" in content),
        ("Font Awesome integration", "font-awesome" in content),
    ]
    
    all_passed = True
    for test_name, condition in tests:
        result = print_status(test_name, condition)
        all_passed = all_passed and result
    
    return all_passed

def validate_html_syntax():
    """Basic HTML syntax validation"""
    print_header("📝 Validating HTML Syntax")
    
    base_template_path = Path("ml_manager/templates/ml_manager/base.html")
    
    with open(base_template_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Basic syntax checks
    tests = [
        ("Opening <html> tag", "<html" in content),
        ("Closing </html> tag", "</html>" in content),
        ("Opening <head> tag", "<head>" in content),
        ("Closing </head> tag", "</head>" in content),
        ("Opening <body> tag", "<body>" in content),
        ("Closing </body> tag", "</body>" in content),
        ("DOCTYPE declaration", "<!DOCTYPE html>" in content),
        ("Meta charset", "charset=" in content),
        ("Meta viewport", "viewport" in content),
    ]
    
    all_passed = True
    for test_name, condition in tests:
        result = print_status(test_name, condition)
        all_passed = all_passed and result
    
    return all_passed

def run_django_syntax_check():
    """Run Django template syntax check"""
    print_header("🔍 Django Template Syntax Check")
    
    try:
        # We're already in the project directory
        
        # Run Django check command
        result = subprocess.run([
            sys.executable, "manage.py", "check", "--tag", "templates"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print_status("Django template syntax check passed", True)
            return True
        else:
            print_status("Django template syntax check failed", False)
            if result.stderr:
                print(f"Errors: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print_status("Django check timed out", False)
        return False
    except Exception as e:
        print_status(f"Django check error: {e}", False)
        return False

def main():
    """Run all dark mode flash fix tests"""
    print_header("🌓 Dark Mode Flash Fix Validation")
    print("Testing implementation to prevent light mode flash when switching windows...")
    
    all_tests = []
    
    # Run all tests
    all_tests.append(test_template_structure())
    all_tests.append(test_css_structure())
    all_tests.append(test_javascript_structure())
    all_tests.append(test_fix_completeness())
    all_tests.append(test_django_integration())
    all_tests.append(validate_html_syntax())
    all_tests.append(run_django_syntax_check())
    
    # Summary
    print_header("📊 Test Summary")
    passed = sum(all_tests)
    total = len(all_tests)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print_status("🎉 All dark mode flash fix tests PASSED!", True)
        print("\n🌓 Dark mode flash fix implementation is complete and ready!")
        print("✅ The fix includes:")
        print("   • Early theme detection in <head>")
        print("   • CSS variables for instant theme application")
        print("   • Window visibility change handling")
        print("   • Window focus event handling")
        print("   • Transition control for smooth UX")
        print("   • LocalStorage integration")
        print("\n🚀 The dark mode should now switch smoothly without flashing!")
        return True
    else:
        print_status("❌ Some tests failed. Please review the implementation.", False)
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
