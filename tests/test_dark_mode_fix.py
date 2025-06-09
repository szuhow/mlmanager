#!/usr/bin/env python3
"""
Test script to verify the dark mode blinking fix implementation.
This script checks that the necessary components are in place to prevent
the light mode flash when switching between windows in dark mode.
"""

import os
import re

def test_dark_mode_fix():
    """Test that all dark mode fix components are properly implemented."""
    
    base_template_path = "/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments/ml_manager/templates/ml_manager/base.html"
    
    print("🔍 Testing Dark Mode Flash Fix Implementation")
    print("=" * 50)
    
    # Read the base template
    try:
        with open(base_template_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ Base template not found!")
        return False
    
    # Test 1: Check for critical early script in head
    print("\n1. Checking for early theme detection script...")
    if "Critical: Apply dark mode immediately to prevent flash" in content:
        print("✅ Early theme detection script found")
        
        # Check for localStorage access
        if "localStorage.getItem('colorMode')" in content:
            print("✅ localStorage theme retrieval implemented")
        else:
            print("❌ localStorage theme retrieval missing")
            return False
            
        # Check for CSS custom properties
        if "--initial-bg" in content and "--initial-color" in content:
            print("✅ CSS custom properties for initial theme found")
        else:
            print("❌ CSS custom properties for initial theme missing")
            return False
            
    else:
        print("❌ Early theme detection script missing")
        return False
    
    # Test 2: Check for flash prevention CSS
    print("\n2. Checking for flash prevention CSS...")
    if "Prevent flash of wrong theme on page load" in content:
        print("✅ Flash prevention CSS found")
        
        # Check for data-theme-loading selectors
        if 'html[data-theme-loading="dark"]' in content and 'html[data-theme-loading="light"]' in content:
            print("✅ Theme loading state selectors found")
        else:
            print("❌ Theme loading state selectors missing")
            return False
            
    else:
        print("❌ Flash prevention CSS missing")
        return False
    
    # Test 3: Check for enhanced JavaScript functionality
    print("\n3. Checking for enhanced JavaScript functionality...")
    if "Handle page visibility changes" in content:
        print("✅ Page visibility change handler found")
    else:
        print("❌ Page visibility change handler missing")
        return False
        
    if "Handle focus events" in content:
        print("✅ Window focus handler found")
    else:
        print("❌ Window focus handler missing")
        return False
        
    if "removeAttribute('data-theme-loading')" in content:
        print("✅ Loading state cleanup found")
    else:
        print("❌ Loading state cleanup missing")
        return False
    
    # Test 4: Check for proper meta charset
    print("\n4. Checking for proper HTML structure...")
    if '<meta charset="UTF-8">' in content:
        print("✅ Proper meta charset found")
    else:
        print("❌ Meta charset issue detected")
        return False
    
    # Test 5: Check for transition handling
    print("\n5. Checking for smooth transition handling...")
    if "transition: none !important" in content:
        print("✅ Transition prevention during load found")
    else:
        print("❌ Transition prevention missing")
        return False
        
    if "skipTransition" in content:
        print("✅ Transition control parameter found")
    else:
        print("❌ Transition control missing")
        return False
    
    print("\n" + "=" * 50)
    print("✅ All dark mode flash fix components are properly implemented!")
    print("\nThe fix includes:")
    print("• Early theme detection in <head> section")
    print("• CSS custom properties for immediate theme application")
    print("• Flash prevention CSS with loading states")
    print("• Enhanced JavaScript with visibility and focus handlers")
    print("• Proper transition management")
    print("• Loading state cleanup")
    
    return True

def test_css_structure():
    """Test that CSS structure is correct for the theme system."""
    
    base_template_path = "/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments/ml_manager/templates/ml_manager/base.html"
    
    print("\n🎨 Testing CSS Structure")
    print("=" * 30)
    
    try:
        with open(base_template_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ Base template not found!")
        return False
    
    # Check for proper dark mode selectors
    dark_selectors = [
        'body[data-color-mode="dark"]',
        'html[data-theme-loading="dark"]',
    ]
    
    for selector in dark_selectors:
        if selector in content:
            print(f"✅ Found selector: {selector}")
        else:
            print(f"❌ Missing selector: {selector}")
            return False
    
    # Check for important CSS rules
    important_rules = [
        'background-color: var(--initial-bg, #ffffff) !important;',
        'color: var(--initial-color, #000000) !important;',
        'transition: none !important;'
    ]
    
    for rule in important_rules:
        if rule in content:
            print(f"✅ Found CSS rule: {rule[:40]}...")
        else:
            print(f"❌ Missing CSS rule: {rule[:40]}...")
            return False
    
    print("✅ CSS structure is correct!")
    return True

if __name__ == "__main__":
    success = test_dark_mode_fix()
    css_success = test_css_structure()
    
    if success and css_success:
        print("\n🎉 Dark mode flash fix is ready!")
        print("\nTo test the fix:")
        print("1. Set dark mode in your browser")
        print("2. Navigate between ML Manager pages")
        print("3. Switch between browser windows/tabs")
        print("4. Refresh pages")
        print("5. Verify no light mode flash occurs")
    else:
        print("\n❌ Some issues were found. Please review the implementation.")
        exit(1)
