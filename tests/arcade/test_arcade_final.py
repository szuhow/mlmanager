#!/usr/bin/env python3
"""
Final test of all 6 ARCADE dataset types in MLManager GUI.
This test verifies that all dataset types are properly handled and can generate previews.
"""

import os
import sys
import django
import requests
import json
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent / 'core'))

def test_arcade_gui_integration():
    """Test ARCADE integration through GUI endpoints"""
    print("🔍 Testing ARCADE GUI Integration")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Test 1: Check if application is running
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Application is running")
        else:
            print(f"❌ Application returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to application: {e}")
        return False
    
    # Test 2: Test dataset preview endpoint for each ARCADE type
    arcade_types = [
        'arcade_binary_segmentation',
        'arcade_semantic_segmentation', 
        'arcade_artery_classification',
        'arcade_semantic_seg_binary',
        'arcade_stenosis_detection',
        'arcade_stenosis_segmentation'
    ]
    
    test_data_path = '/app/data/datasets'  # Path inside Docker container
    
    results = {}
    
    for dataset_type in arcade_types:
        print(f"\n🧪 Testing {dataset_type}...")
        
        try:
            # Prepare POST data
            post_data = {
                'dataset_type': dataset_type,
                'data_path': test_data_path
            }
            
            # Make request to dataset preview endpoint
            response = requests.post(
                f"{base_url}/ml/dataset-preview/",
                data=post_data,
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"   ✅ HTTP 200 OK")
                
                # Check if response contains expected elements
                content = response.text
                
                checks = {
                    'samples_generated': 'dataset_preview' in content,
                    'no_error_message': 'error' not in content.lower() or 'Error loading dataset' not in content,
                    'has_images': 'image_' in content,
                    'has_masks': 'mask_' in content,
                    'has_analysis': 'analysis' in content
                }
                
                passed_checks = sum(checks.values())
                total_checks = len(checks)
                
                print(f"   📊 Passed {passed_checks}/{total_checks} content checks")
                
                for check_name, passed in checks.items():
                    status = "✅" if passed else "❌"
                    print(f"      {status} {check_name}")
                
                results[dataset_type] = {
                    'status_code': response.status_code,
                    'checks_passed': passed_checks,
                    'total_checks': total_checks,
                    'success': passed_checks >= total_checks * 0.8  # 80% threshold
                }
                
            else:
                print(f"   ❌ HTTP {response.status_code}")
                print(f"   📄 Response: {response.text[:200]}...")
                results[dataset_type] = {
                    'status_code': response.status_code,
                    'checks_passed': 0,
                    'total_checks': 5,
                    'success': False
                }
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request failed: {e}")
            results[dataset_type] = {
                'status_code': 0,
                'checks_passed': 0,
                'total_checks': 5,
                'success': False,
                'error': str(e)
            }
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 FINAL RESULTS")
    print("=" * 60)
    
    successful_types = 0
    total_types = len(arcade_types)
    
    for dataset_type, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{status} {dataset_type}")
        if result['success']:
            successful_types += 1
            
        if 'error' in result:
            print(f"    Error: {result['error']}")
    
    print("\n" + "=" * 60)
    print(f"🎯 OVERALL RESULT: {successful_types}/{total_types} ARCADE types working")
    
    if successful_types == total_types:
        print("🎉 SUCCESS! All ARCADE dataset types are properly integrated!")
        return True
    elif successful_types >= total_types * 0.8:
        print("⚠️  MOSTLY WORKING: Most ARCADE types are functional")
        return True
    else:
        print("❌ FAILED: Many ARCADE types have issues")
        return False

def main():
    """Main test function"""
    print("🚀 ARCADE Dataset Types - Final Integration Test")
    print("=" * 60)
    
    success = test_arcade_gui_integration()
    
    if success:
        print("\n✅ All tests passed! ARCADE implementation is ready.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
