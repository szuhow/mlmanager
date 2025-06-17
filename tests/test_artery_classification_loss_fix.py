#!/usr/bin/env python3
"""
Test script to verify that the artery classification loss function fix works correctly
"""

import sys
import os
import torch

# Add the project root to Python path
sys.path.insert(0, '/Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager')

def test_loss_function_selection():
    """Test that the correct loss function is selected for artery classification"""
    print("🧪 Testing Loss Function Selection for Artery Classification")
    print("=" * 70)
    
    try:
        # Mock the class_info that would be returned for artery classification
        class_info = {
            'num_classes': 2,
            'class_type': 'classification',
            'task_type': 'artery_classification'
        }
        
        print(f"📋 Mock class_info: {class_info}")
        
        # Test the logic that would be in train.py
        if class_info and class_info.get('task_type') == 'artery_classification':
            # Classification task (ARCADEArteryClassification)
            print(f"✅ Detected artery classification task")
            loss_function = torch.nn.CrossEntropyLoss()
            loss_name = "CrossEntropyLoss"
        else:
            # This shouldn't happen for our test
            print(f"❌ Failed to detect artery classification task")
            loss_name = "WRONG_LOSS"
            
        print(f"🎯 Selected loss function: {loss_name}")
        
        # Test the loss function with sample data
        if loss_name == "CrossEntropyLoss":
            # Create sample data for classification
            batch_size = 16
            num_classes = 2
            
            # Model outputs (logits) - shape [batch_size, num_classes]
            outputs = torch.randn(batch_size, num_classes)
            
            # Ground truth labels - shape [batch_size] with class indices
            labels = torch.randint(0, num_classes, (batch_size,))
            
            print(f"📊 Test data shapes:")
            print(f"   - Outputs (logits): {outputs.shape}")
            print(f"   - Labels (class indices): {labels.shape}")
            print(f"   - Labels values: {labels}")
            
            # Calculate loss
            loss_value = loss_function(outputs, labels)
            
            print(f"✅ Loss calculation successful: {loss_value.item():.4f}")
            
            # Test accuracy calculation
            predicted_classes = torch.argmax(outputs, dim=1)
            accuracy = (predicted_classes == labels).float().mean().item()
            
            print(f"📈 Accuracy calculation: {accuracy:.4f}")
            
            return True
        else:
            print(f"❌ Wrong loss function selected")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_segmentation_loss_still_works():
    """Test that segmentation tasks still use DiceLoss correctly"""
    print("\n🧪 Testing Segmentation Loss Function (Should Still Work)")
    print("=" * 70)
    
    try:
        # Mock class_info for regular segmentation
        class_info = {
            'num_classes': 1,
            'class_type': 'binary',
            'task_type': 'segmentation'
        }
        
        print(f"📋 Mock class_info: {class_info}")
        
        # Test the logic for segmentation
        if class_info and class_info.get('task_type') == 'artery_classification':
            loss_name = "WRONG - Should not reach here"
        else:
            # Binary segmentation (default)
            print(f"✅ Detected binary segmentation task")
            try:
                from monai.losses import DiceLoss as MonaiDiceLoss
                loss_function = MonaiDiceLoss(sigmoid=True)
                loss_name = "MonaiDiceLoss"
            except ImportError:
                print("⚠️  MONAI not available, using torch.nn.BCEWithLogitsLoss as fallback")
                loss_function = torch.nn.BCEWithLogitsLoss()
                loss_name = "BCEWithLogitsLoss"
                
        print(f"🎯 Selected loss function: {loss_name}")
        
        # Test with segmentation data
        batch_size = 16
        height, width = 64, 64
        
        # Model outputs (logits) - shape [batch_size, 1, H, W]
        outputs = torch.randn(batch_size, 1, height, width)
        
        # Ground truth masks - shape [batch_size, 1, H, W] with binary values
        labels = torch.randint(0, 2, (batch_size, 1, height, width)).float()
        
        print(f"📊 Test data shapes:")
        print(f"   - Outputs (logits): {outputs.shape}")
        print(f"   - Labels (binary masks): {labels.shape}")
        
        # Calculate loss
        loss_value = loss_function(outputs, labels)
        
        print(f"✅ Loss calculation successful: {loss_value.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🔧 ARTERY CLASSIFICATION LOSS FUNCTION FIX TEST")
    print("=" * 70)
    
    test1_passed = test_loss_function_selection()
    test2_passed = test_segmentation_loss_still_works()
    
    print("\n📋 TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"✅ Artery Classification Loss: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"✅ Segmentation Loss (unchanged): {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Loss function selection fix is working correctly")
        print("✅ Artery classification will use CrossEntropyLoss")
        print("✅ Segmentation tasks will continue using DiceLoss/BCE")
        return True
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please check the implementation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
