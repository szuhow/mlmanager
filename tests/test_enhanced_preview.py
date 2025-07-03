#!/usr/bin/env python3

"""
Test enhanced dataset preview for ARCADE stenosis detection and artery classification
"""

import os
import sys
import django

# Add the project root to Python path
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/core')

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

import requests
import json
from ml.datasets.arcade_loader import get_arcade_dataset_root, get_arcade_task_paths
from ml.datasets.torch_arcade_loader import get_arcade_dataset_info

def test_enhanced_preview():
    """Test enhanced dataset preview for stenosis detection and artery classification"""
    
    print("🚀 Testing Enhanced Dataset Preview for ARCADE")
    print("=" * 60)
    
    # Base URL for Django API
    base_url = "http://localhost:8000"
    
    try:
        # Test 1: Get ARCADE dataset info
        print("\n1. Testing ARCADE Dataset Detection...")
        
        arcade_root = get_arcade_dataset_root()
        print(f"   ARCADE root: {arcade_root}")
        
        if not arcade_root or not arcade_root.exists():
            print("   ❌ ARCADE dataset not found!")
            return False
        
        # Test segmentation paths
        segmentation_paths = get_arcade_task_paths(arcade_root, 'segmentation')
        print(f"   Segmentation paths found: {bool(segmentation_paths)}")
        
        if segmentation_paths:
            print(f"   Train images: {segmentation_paths.get('train_images')}")
            print(f"   Val images: {segmentation_paths.get('val_images')}")
            print(f"   Annotations: {segmentation_paths.get('train_annotations')}")
        
        # Test 2: Dataset Preview API for Binary Segmentation
        print("\n2. Testing Binary Segmentation Preview...")
        
        if segmentation_paths and segmentation_paths.get('train_images'):
            preview_data = {
                'data_path': str(segmentation_paths['train_images'].parent),
                'dataset_type': 'binary_segmentation'
            }
            
            response = requests.post(f"{base_url}/ml/dataset-preview/", data=preview_data)
            
            if response.status_code == 200:
                print("   ✅ Binary segmentation preview successful")
                # Check if response contains samples
                if 'samples' in response.text and 'Binary Segmentation' in response.text:
                    print("   ✅ Binary segmentation properly detected and displayed")
                else:
                    print("   ⚠️ Binary segmentation response missing expected content")
            else:
                print(f"   ❌ Binary segmentation preview failed: {response.status_code}")
        
        # Test 3: Stenosis Detection Preview
        print("\n3. Testing Stenosis Detection Preview...")
        
        try:
            # Force stenosis detection type
            preview_data = {
                'data_path': str(arcade_root / 'training'),
                'dataset_type': 'stenosis_detection'
            }
            
            response = requests.post(f"{base_url}/ml/dataset-preview/", data=preview_data)
            
            if response.status_code == 200:
                print("   ✅ Stenosis detection preview successful")
                
                # Check for stenosis-specific content
                if 'stenosis' in response.text.lower() or 'bounding' in response.text.lower():
                    print("   ✅ Stenosis detection properly detected")
                else:
                    print("   ⚠️ Stenosis detection content not found in response")
                    
            else:
                print(f"   ❌ Stenosis detection preview failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error testing stenosis detection: {e}")
        
        # Test 4: Artery Classification Preview
        print("\n4. Testing Artery Classification Preview...")
        
        try:
            # Force artery classification type
            preview_data = {
                'data_path': str(arcade_root / 'training'),
                'dataset_type': 'artery_classification'
            }
            
            response = requests.post(f"{base_url}/ml/dataset-preview/", data=preview_data)
            
            if response.status_code == 200:
                print("   ✅ Artery classification preview successful")
                
                # Check for classification-specific content
                if 'artery' in response.text.lower() or 'classification' in response.text.lower():
                    print("   ✅ Artery classification properly detected")
                else:
                    print("   ⚠️ Artery classification content not found in response")
                    
            else:
                print(f"   ❌ Artery classification preview failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error testing artery classification: {e}")
        
        # Test 5: Auto-detection of Dataset Types
        print("\n5. Testing Auto-detection...")
        
        try:
            # Test auto-detection without specifying type
            preview_data = {
                'data_path': str(arcade_root / 'training'),
                'dataset_type': 'auto'
            }
            
            response = requests.post(f"{base_url}/ml/dataset-preview/", data=preview_data)
            
            if response.status_code == 200:
                print("   ✅ Auto-detection successful")
                
                # Try to determine what was detected
                response_text = response.text.lower()
                if 'binary segmentation' in response_text:
                    print("   📋 Auto-detected: Binary Segmentation")
                elif 'semantic segmentation' in response_text:
                    print("   📋 Auto-detected: Semantic Segmentation")
                elif 'stenosis detection' in response_text:
                    print("   📋 Auto-detected: Stenosis Detection")
                elif 'artery classification' in response_text:
                    print("   📋 Auto-detected: Artery Classification")
                else:
                    print("   📋 Auto-detected: Unknown/Other")
                    
            else:
                print(f"   ❌ Auto-detection failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error testing auto-detection: {e}")
        
        # Test 6: Direct torch-arcade loader test
        print("\n6. Testing torch-arcade Loaders...")
        
        try:
            from ml.datasets.torch_arcade_loader import ARCADEStenosisDetection, ARCADEArteryClassification
            
            # Test stenosis detection loader
            try:
                stenosis_dataset = ARCADEStenosisDetection(
                    root=arcade_root,
                    image_set='train',
                    download=False
                )
                print(f"   ✅ Stenosis detection loader: {len(stenosis_dataset)} samples")
                
                # Try to get a sample
                if len(stenosis_dataset) > 0:
                    sample = stenosis_dataset[0]
                    print(f"   📊 Sample keys: {list(sample.keys()) if isinstance(sample, dict) else 'tuple/list'}")
                    
            except Exception as e:
                print(f"   ❌ Stenosis detection loader error: {e}")
            
            # Test artery classification loader
            try:
                classification_dataset = ARCADEArteryClassification(
                    root=arcade_root,
                    image_set='train',
                    download=False
                )
                print(f"   ✅ Artery classification loader: {len(classification_dataset)} samples")
                
                # Try to get a sample
                if len(classification_dataset) > 0:
                    sample = classification_dataset[0]
                    print(f"   📊 Sample keys: {list(sample.keys()) if isinstance(sample, dict) else 'tuple/list'}")
                    
            except Exception as e:
                print(f"   ❌ Artery classification loader error: {e}")
                
        except ImportError as e:
            print(f"   ❌ torch-arcade import error: {e}")
        except Exception as e:
            print(f"   ❌ torch-arcade test error: {e}")
        
        print("\n" + "=" * 60)
        print("🎉 Enhanced Preview Test Completed!")
        print("✨ Check the web interface at: http://localhost:8000/ml/dataset-preview/")
        print("🔍 Try different dataset types: binary_segmentation, stenosis_detection, artery_classification")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_preview()
    sys.exit(0 if success else 1)
