#!/usr/bin/env python3
"""
Test enhanced model details view functionality
"""
import time
import requests
import json
from pathlib import Path

def test_model_details_enhanced():
    """Test enhanced model details functionality"""
    
    base_url = "http://localhost:8000"
    
    print("🚀 Testing Enhanced Model Details View")
    print("=" * 50)
    
    # Test 1: Check if Django is running
    try:
        response = requests.get(f"{base_url}/ml/")
        print(f"✅ Django server accessible (status: {response.status_code})")
    except Exception as e:
        print(f"❌ Django server not accessible: {e}")
        return False
    
    # Test 2: Get model list and find a model
    try:
        response = requests.get(f"{base_url}/ml/")
        if response.status_code == 200:
            print("✅ ML Manager accessible")
            
            # Try to find existing models
            if "model_" in response.text or "Model #" in response.text:
                print("✅ Found existing models")
                
                # Extract model ID from HTML (basic approach)
                content = response.text
                import re
                model_ids = re.findall(r'/ml/model/(\d+)/', content)
                
                if model_ids:
                    model_id = model_ids[0]
                    print(f"✅ Found model ID: {model_id}")
                    
                    # Test 3: Check model detail page
                    detail_response = requests.get(f"{base_url}/ml/model/{model_id}/")
                    if detail_response.status_code == 200:
                        print("✅ Model detail page accessible")
                        
                        # Check for enhanced sections
                        content = detail_response.text
                        checks = [
                            ("Data Augmentation", "Data Augmentation section"),
                            ("Medical Preprocessing", "Medical Preprocessing section"),
                            ("Optimizer", "Optimizer & Early Stopping section"),
                            ("toggleImageZoom", "Image zoom functionality"),
                            ("ModelDetailManager", "Auto-refresh manager"),
                            ("openTrainingImageModal", "Training image modal"),
                        ]
                        
                        for check_text, description in checks:
                            if check_text in content:
                                print(f"✅ {description} present")
                            else:
                                print(f"⚠️  {description} missing")
                        
                        # Test 4: Check progress endpoint
                        progress_response = requests.get(f"{base_url}/ml/model/{model_id}/progress/")
                        if progress_response.status_code == 200:
                            print("✅ Progress endpoint accessible")
                            try:
                                progress_data = progress_response.json()
                                print(f"✅ Progress data: {progress_data.get('status', 'unknown')}")
                            except:
                                print("⚠️  Progress data not JSON")
                        else:
                            print(f"❌ Progress endpoint failed: {progress_response.status_code}")
                    else:
                        print(f"❌ Model detail page failed: {detail_response.status_code}")
                else:
                    print("⚠️  No model IDs found in HTML")
            else:
                print("⚠️  No models found - create a model first")
        else:
            print(f"❌ ML Manager failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing models: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Enhanced Model Details Test Complete!")
    print("\nTo manually test:")
    print(f"1. Visit: {base_url}/ml/")
    print("2. Click on any model to view details")
    print("3. Check for augmentation/preprocessing parameters")
    print("4. Click on training images to zoom")
    print("5. Start training and watch auto-refresh")
    return True

def test_image_zoom_js():
    """Test that JavaScript functions are properly defined"""
    print("\n🔍 Testing JavaScript Function Definitions")
    print("=" * 50)
    
    template_path = Path("core/apps/ml_manager/templates/ml_manager/model_detail.html")
    
    if not template_path.exists():
        print("❌ Template file not found")
        return False
    
    content = template_path.read_text()
    
    js_functions = [
        "toggleImageZoom",
        "downloadImage", 
        "initializeTrainingImages",
        "openTrainingImageModal",
        "navigateTrainingImage",
        "showImageInModal",
        "openLogsModal"
    ]
    
    for func in js_functions:
        if f"function {func}" in content:
            print(f"✅ {func}() defined")
        else:
            print(f"❌ {func}() missing")
    
    # Check for modal elements
    modal_elements = [
        'id="imageModal"',
        'id="modalImage"', 
        'id="prevImageBtn"',
        'id="nextImageBtn"',
        'onclick="toggleImageZoom()"'
    ]
    
    for element in modal_elements:
        if element in content:
            print(f"✅ Modal element: {element}")
        else:
            print(f"❌ Missing modal element: {element}")
    
    return True

def test_training_parameters_display():
    """Test if training parameters sections are present"""
    print("\n📊 Testing Training Parameters Display")
    print("=" * 50)
    
    template_path = Path("core/apps/ml_manager/templates/ml_manager/model_detail.html")
    content = template_path.read_text()
    
    parameter_sections = [
        ("Data Augmentation", [
            "training_details.augmentation.random_flip",
            "training_details.augmentation.random_rotate", 
            "training_details.augmentation.random_scale",
            "training_details.augmentation.random_intensity"
        ]),
        ("Medical Preprocessing", [
            "training_details.preprocessing.use_medical_preprocessing",
            "training_details.preprocessing.use_clahe",
            "training_details.preprocessing.use_frangi_filter"
        ]),
        ("Optimizer & Early Stopping", [
            "training_details.optimizer.optimizer",
            "training_details.optimizer.learning_rate",
            "training_details.optimizer.lr_scheduler",
            "training_details.optimizer.early_stopping"
        ])
    ]
    
    for section_name, parameters in parameter_sections:
        print(f"\n{section_name}:")
        section_found = section_name in content
        print(f"  Section present: {'✅' if section_found else '❌'}")
        
        for param in parameters:
            param_found = param in content
            print(f"  {param}: {'✅' if param_found else '❌'}")
    
    return True

if __name__ == "__main__":
    print("🔧 Enhanced Model Details Test Suite")
    print("=" * 60)
    
    test_image_zoom_js()
    test_training_parameters_display()
    test_model_details_enhanced()
    
    print("\n🎉 Test suite completed!")
    print("\nNext steps:")
    print("1. Start Django server: make enhanced-start")
    print("2. Visit http://localhost:8000/ml/")
    print("3. Create or view an existing model")
    print("4. Test enhanced features manually")
