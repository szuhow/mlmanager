#!/usr/bin/env python3
"""
Test final ARCADE mask generation in GUI.
This script tests all 6 ARCADE dataset types and checks if masks are properly generated with visible content.
"""

import requests
import json
import time
import os

def test_arcade_dataset_preview(base_url="http://localhost:8000"):
    """Test ARCADE dataset preview generation"""
    
    arcade_types = [
        ('arcade_binary_segmentation', 'ARCADE Binary Segmentation'),
        ('arcade_semantic_segmentation', 'ARCADE Semantic Segmentation'),
        ('arcade_artery_classification', 'ARCADE Artery Classification'),
        ('arcade_semantic_seg_binary', 'ARCADE Semantic Segmentation Binary'),
        ('arcade_stenosis_detection', 'ARCADE Stenosis Detection'),
        ('arcade_stenosis_segmentation', 'ARCADE Stenosis Segmentation'),
    ]
    
    results = {}
    
    for dataset_type, dataset_name in arcade_types:
        print(f"\n=== Testing {dataset_name} ===")
        
        try:
            # Make POST request to generate preview
            response = requests.post(f"{base_url}/ml/dataset-preview/", data={
                'data_path': '/app/data/datasets',
                'dataset_type': dataset_type
            }, timeout=60)
            
            if response.status_code == 200:
                print(f"✅ {dataset_name}: Preview generated successfully")
                
                # Check if response contains sample data
                if 'samples' in response.text and 'mask_url' in response.text:
                    print(f"✅ {dataset_name}: Contains sample data with masks")
                    
                    # Check for error messages in template
                    if 'Failed lookup for key [mask_coverage]' in response.text:
                        print(f"❌ {dataset_name}: Template error - missing mask_coverage field")
                        results[dataset_type] = "Template Error"
                    else:
                        print(f"✅ {dataset_name}: No template errors detected")
                        results[dataset_type] = "Success"
                else:
                    print(f"⚠️ {dataset_name}: No samples generated")
                    results[dataset_type] = "No Samples"
            else:
                print(f"❌ {dataset_name}: HTTP {response.status_code}")
                results[dataset_type] = f"HTTP {response.status_code}"
                
        except Exception as e:
            print(f"❌ {dataset_name}: Exception - {e}")
            results[dataset_type] = f"Exception: {e}"
        
        # Small delay between requests
        time.sleep(2)
    
    # Summary
    print("\n" + "="*60)
    print("ARCADE DATASET PREVIEW TEST SUMMARY")
    print("="*60)
    
    success_count = 0
    for dataset_type, result in results.items():
        status = "✅" if result == "Success" else "❌"
        print(f"{status} {dataset_type}: {result}")
        if result == "Success":
            success_count += 1
    
    print(f"\nResults: {success_count}/{len(arcade_types)} datasets working correctly")
    
    if success_count == len(arcade_types):
        print("🎉 ALL ARCADE DATASET TYPES ARE WORKING!")
        return True
    else:
        print("⚠️ Some ARCADE dataset types need fixes")
        return False

def test_mask_visibility():
    """Test if generated masks are visible (not black/empty)"""
    import docker
    
    print("\n=== Testing Mask Visibility ===")
    
    try:
        client = docker.from_env()
        container = client.containers.get('web')
        
        # List mask files
        exec_result = container.exec_run("ls -la /app/data/media/temp/dataset_preview/mask_*.png 2>/dev/null | head -5")
        
        if exec_result.exit_code == 0:
            mask_files = exec_result.output.decode().strip().split('\n')
            print(f"Found {len(mask_files)} mask files")
            
            # Check file sizes (should be > 300 bytes for visible content)
            large_files = 0
            for line in mask_files:
                if line and 'mask_' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        size = int(parts[4])
                        filename = parts[-1]
                        if size > 300:
                            large_files += 1
                            print(f"✅ {filename}: {size} bytes (visible content)")
                        else:
                            print(f"⚠️ {filename}: {size} bytes (possibly empty)")
            
            visibility_ratio = large_files / len(mask_files) if mask_files else 0
            print(f"\nMask visibility: {large_files}/{len(mask_files)} files have visible content ({visibility_ratio*100:.1f}%)")
            
            return visibility_ratio > 0.7  # At least 70% should have visible content
        else:
            print("No mask files found")
            return False
            
    except Exception as e:
        print(f"Error checking mask visibility: {e}")
        return False

if __name__ == "__main__":
    print("ARCADE Mask Generation Test")
    print("="*40)
    
    # Test dataset preview generation
    preview_success = test_arcade_dataset_preview()
    
    # Test mask visibility
    mask_visibility = test_mask_visibility()
    
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"Dataset Preview Generation: {'✅ PASS' if preview_success else '❌ FAIL'}")
    print(f"Mask Visibility: {'✅ PASS' if mask_visibility else '❌ FAIL'}")
    
    if preview_success and mask_visibility:
        print("\n🎉 ALL TESTS PASSED! ARCADE mask generation is working correctly!")
        exit(0)
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        exit(1)
