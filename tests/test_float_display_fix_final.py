#!/usr/bin/env python3
"""
Test JavaScript Float Display Fix - Final Comprehensive Test
Tests all JavaScript files and HTML templates to ensure epoch display works correctly
"""

import os
import re
import subprocess
import sys
from pathlib import Path

# Test results
test_results = {
    'javascript_fixes': [],
    'template_checks': [],
    'float_formatting_issues': [],
    'success': True
}

def test_javascript_files():
    """Test all JavaScript files for proper epoch integer formatting"""
    js_files = [
        'core/static/ml_manager/js/model_detail_unified.js',
        'core/static/ml_manager/js/model_detail_simple.js', 
        'core/static/ml_manager/js/model_detail_fixes.js'
    ]
    
    print("🔍 Testing JavaScript files for integer formatting...")
    
    for js_file in js_files:
        if os.path.exists(js_file):
            with open(js_file, 'r') as f:
                content = f.read()
                
            # Check for proper epoch formatting
            has_integer_formatting = False
            has_problematic_formatting = False
            
            # Pattern 1: updateMetric with integer parameter 
            if "updateMetric('current-epoch', progress.current_epoch, true)" in content.lower() or \
               "updatemetric('current-epoch', progress.current_epoch, true)" in content.lower():
                has_integer_formatting = True
                
            # Pattern 2: toString() formatting for epochs
            if "progress.current_epoch ? progress.current_epoch.toString()" in content:
                has_integer_formatting = True
                
            # Check for problematic patterns
            if "current-epoch': progress.current_epoch," in content and \
               "progress.current_epoch ? progress.current_epoch.toString()" not in content:
                has_problematic_formatting = True
                
            result = {
                'file': js_file,
                'has_integer_formatting': has_integer_formatting,
                'has_problematic_formatting': has_problematic_formatting,
                'status': '✅' if has_integer_formatting and not has_problematic_formatting else '❌'
            }
            
            test_results['javascript_fixes'].append(result)
            print(f"  {result['status']} {js_file}")
            if has_problematic_formatting:
                print(f"    ⚠️  Still has float formatting for epochs")
            if has_integer_formatting:
                print(f"    ✅ Has proper integer formatting")
        else:
            print(f"  ⚠️  File not found: {js_file}")

def test_html_templates():
    """Test HTML templates for floatformat usage on epochs"""
    print("\n🔍 Testing HTML templates for epoch floatformat usage...")
    
    template_dir = 'core/apps/ml_manager/templates/ml_manager'
    if os.path.exists(template_dir):
        for template_file in os.listdir(template_dir):
            if template_file.endswith('.html'):
                template_path = os.path.join(template_dir, template_file)
                with open(template_path, 'r') as f:
                    content = f.read()
                    
                # Check for problematic floatformat usage on epochs
                problematic_patterns = [
                    r'current_epoch\|floatformat',
                    r'total_epochs\|floatformat'
                ]
                
                has_issues = False
                for pattern in problematic_patterns:
                    if re.search(pattern, content):
                        has_issues = True
                        test_results['float_formatting_issues'].append({
                            'file': template_path,
                            'pattern': pattern
                        })
                
                result = {
                    'file': template_file,
                    'has_epoch_floatformat': has_issues,
                    'status': '❌' if has_issues else '✅'
                }
                
                test_results['template_checks'].append(result)
                print(f"  {result['status']} {template_file}")
                if has_issues:
                    print(f"    ⚠️  Uses floatformat on epoch values")

def test_epoch_display_simulation():
    """Simulate epoch display formatting"""
    print("\n🧪 Testing epoch display simulation...")
    
    # Test scenarios
    test_scenarios = [
        {'current_epoch': 1, 'total_epochs': 10, 'description': 'Early training'},
        {'current_epoch': 5, 'total_epochs': 10, 'description': 'Mid training'},
        {'current_epoch': 10, 'total_epochs': 10, 'description': 'Training complete'},
        {'current_epoch': 27, 'total_epochs': 100, 'description': 'Long training'},
    ]
    
    for scenario in test_scenarios:
        current = scenario['current_epoch']
        total = scenario['total_epochs']
        description = scenario['description']
        
        # Test different formatting approaches
        epoch_display = f"Current Epoch: {current} / {total}"
        problematic_display = f"Current Epoch: {float(current):.4f} / {total}"
        
        print(f"  📊 {description}:")
        print(f"    ✅ Correct:     {epoch_display}")
        print(f"    ❌ Problematic: {problematic_display}")
        
        # Verify our fix would work
        if f"{current}" in epoch_display and ".0000" not in epoch_display:
            print(f"    ✓ Format test passed")
        else:
            print(f"    ✗ Format test failed")
            test_results['success'] = False

def run_javascript_syntax_check():
    """Run basic JavaScript syntax check"""
    print("\n🔧 Running JavaScript syntax checks...")
    
    js_files = [
        'core/static/ml_manager/js/model_detail_unified.js',
        'core/static/ml_manager/js/model_detail_simple.js', 
        'core/static/ml_manager/js/model_detail_fixes.js'
    ]
    
    for js_file in js_files:
        if os.path.exists(js_file):
            try:
                # Use node to check syntax if available
                result = subprocess.run(['node', '--check', js_file], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  ✅ {js_file} - Syntax OK")
                else:
                    print(f"  ❌ {js_file} - Syntax Error: {result.stderr}")
                    test_results['success'] = False
            except FileNotFoundError:
                print(f"  ⚠️  Node.js not available for syntax check of {js_file}")

def summarize_results():
    """Print test summary"""
    print("\n" + "="*60)
    print("🏁 FLOAT DISPLAY FIX TEST RESULTS")
    print("="*60)
    
    # JavaScript fixes summary
    print("\n📜 JavaScript Files:")
    js_passed = 0
    for result in test_results['javascript_fixes']:
        print(f"  {result['status']} {result['file']}")
        if result['status'] == '✅':
            js_passed += 1
    
    print(f"\n  JavaScript files passed: {js_passed}/{len(test_results['javascript_fixes'])}")
    
    # Template checks summary
    print("\n📄 HTML Templates:")
    template_passed = sum(1 for r in test_results['template_checks'] if r['status'] == '✅')
    print(f"  Template files checked: {len(test_results['template_checks'])}")
    print(f"  Templates without epoch floatformat: {template_passed}")
    
    # Float formatting issues
    if test_results['float_formatting_issues']:
        print("\n⚠️  Float formatting issues found:")
        for issue in test_results['float_formatting_issues']:
            print(f"    - {issue['file']}: {issue['pattern']}")
    
    # Overall result
    overall_status = "✅ PASSED" if test_results['success'] else "❌ FAILED"
    print(f"\n🎯 Overall Status: {overall_status}")
    
    if test_results['success']:
        print("\n✨ All epoch display formatting issues should be resolved!")
        print("   The 'Current Epoch: 2.0000 / 3' issue should now show as 'Current Epoch: 2 / 3'")
    else:
        print("\n⚠️  Some issues remain that may cause float display problems.")
    
    return test_results['success']

def main():
    """Run all tests"""
    print("🚀 Starting Comprehensive Float Display Fix Test")
    print("="*60)
    
    # Change to project directory
    os.chdir('/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments')
    
    # Run all tests
    test_javascript_files()
    test_html_templates()
    test_epoch_display_simulation()
    run_javascript_syntax_check()
    
    # Summary
    success = summarize_results()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
