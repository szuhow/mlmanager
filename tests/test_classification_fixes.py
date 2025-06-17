#!/usr/bin/env python3
"""
Test poprawek dla modeli klasyfikacji:
1. MLflow step-based logging (step=epoch+1)
2. Classification accuracy metrics instead of dice
3. Enhanced model loading with error handling
"""

import os
import sys
import tempfile
import torch
import numpy as np
from pathlib import Path

# Add paths for imports
sys.path.insert(0, '/app')

def test_mlflow_step_logging():
    """Test 1: MLflow step-based logging fix"""
    print("🧪 Test 1: MLflow step-based logging")
    
    # Check train.py content for proper step usage
    train_file = '/app/ml/training/train.py'
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Find all MLflow metric logging lines
    step_issues = []
    step_fixes = []
    
    for i, line in enumerate(content.split('\n'), 1):
        if 'mlflow.log_metric' in line and 'step=' in line:
            if 'step=epoch+1' in line:
                step_fixes.append((i, line.strip()))
            elif 'step=epoch' in line and 'step=epoch+1' not in line:
                step_issues.append((i, line.strip()))
    
    print(f"✅ Fixed step logging lines: {len(step_fixes)}")
    print(f"❌ Problematic step logging lines: {len(step_issues)}")
    
    for line_num, line in step_fixes:
        print(f"  ✅ Line {line_num}: step=epoch+1 used correctly")
    
    if step_issues:
        print("⚠️ Found issues with step logging:")
        for line_num, line in step_issues:
            print(f"  ❌ Line {line_num}: {line}")
    
    return len(step_issues) == 0

def test_classification_metrics():
    """Test 2: Classification metrics using accuracy instead of dice"""
    print("\n🧪 Test 2: Classification metrics")
    
    train_file = '/app/ml/training/train.py'
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Check if classification uses accuracy metrics
    has_classification_accuracy = (
        'train_accuracy' in content and 
        'val_accuracy' in content and
        'artery_classification' in content
    )
    
    # Check if segmentation still uses dice
    has_segmentation_dice = (
        'train_dice' in content and 
        'val_dice' in content
    )
    
    print(f"✅ Classification uses accuracy metrics: {has_classification_accuracy}")
    print(f"✅ Segmentation still uses dice metrics: {has_segmentation_dice}")
    
    # Look for the specific metrics dictionary creation
    lines = content.split('\n')
    classification_section_found = False
    segmentation_section_found = False
    
    for i, line in enumerate(lines):
        if 'artery_classification' in line and any('train_accuracy' in lines[j] for j in range(i+1, min(i+6, len(lines)))):
            classification_section_found = True
            print(f"  ✅ Found classification metrics section at line {i+1}")
            break
    
    for i, line in enumerate(lines):
        if 'else:' in line and any('train_dice' in lines[j] for j in range(i+1, min(i+5, len(lines)))):
            segmentation_section_found = True
            print(f"  ✅ Found segmentation metrics section at line {i+1}")
            break
    
    return has_classification_accuracy and has_segmentation_dice and classification_section_found

def test_enhanced_model_loading():
    """Test 3: Enhanced model loading with comprehensive error handling"""
    print("\n🧪 Test 3: Enhanced model loading")
    
    predict_file = '/app/ml/inference/predict.py'
    with open(predict_file, 'r') as f:
        content = f.read()
    
    # Check for enhanced error handling features
    features = {
        'strict=True first attempt': 'strict=True' in content,
        'strict=False fallback': 'strict=False' in content,
        'Missing keys detection': 'missing_keys' in content and 'unexpected_keys' in content,
        'Alternative model creation': 'model_alt' in content,
        'Channel mismatch detection': 'actual_in_channels' in content,
        'Parameter naming conventions': 'n_channels' in content and 'input_channels' in content,
        'Comprehensive error logging': 'print(f"⚠️' in content or 'print(f"❌' in content,
    }
    
    all_features_present = True
    for feature, present in features.items():
        status = "✅" if present else "❌"
        print(f"  {status} {feature}: {'Present' if present else 'Missing'}")
        if not present:
            all_features_present = False
    
    # Test the actual error handling structure
    has_proper_structure = (
        'try:' in content and
        'model.load_state_dict(state_dict, strict=True)' in content and
        'except RuntimeError as e:' in content and
        'Missing key(s)' in content
    )
    
    print(f"  ✅ Proper try-catch structure: {has_proper_structure}")
    
    return all_features_present and has_proper_structure

def create_test_model_checkpoint():
    """Create a test model checkpoint to test loading"""
    print("\n🧪 Creating test model checkpoint")
    
    try:
        from ml.training.models.classification_models import UNetClassifier
        
        # Create a test model
        model = UNetClassifier(input_channels=3, num_classes=2)
        
        # Create test checkpoint with metadata
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_metadata': {
                'model_type': 'classification',
                'input_channels': 3,
                'num_classes': 2,
                'task_type': 'artery_classification',
                'epoch': 5,
                'validation_metric': 0.85
            }
        }
        
        # Save to temporary file
        temp_path = tempfile.mktemp(suffix='.pth')
        torch.save(checkpoint, temp_path)
        
        print(f"✅ Test checkpoint created: {temp_path}")
        return temp_path
        
    except Exception as e:
        print(f"❌ Failed to create test checkpoint: {e}")
        return None

def test_model_loading_functionality(checkpoint_path):
    """Test actual model loading functionality"""
    print("\n🧪 Test 4: Model loading functionality")
    
    if not checkpoint_path:
        print("❌ No checkpoint available for testing")
        return False
    
    try:
        from ml.inference.predict import load_model
        
        # Test loading the checkpoint
        model, detected_type = load_model(checkpoint_path, model_type='classification')
        
        print(f"✅ Model loaded successfully")
        print(f"✅ Detected type: {detected_type}")
        print(f"✅ Model class: {type(model).__name__}")
        
        # Test inference with dummy data
        dummy_input = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Inference test passed. Output shape: {output.shape}")
        
        # Clean up
        os.unlink(checkpoint_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up
        if os.path.exists(checkpoint_path):
            os.unlink(checkpoint_path)
        
        return False

def test_early_stopping_fix():
    """Test 5: EarlyStopping AttributeError fix"""
    print("🧪 Test 5: EarlyStopping AttributeError fix")
    
    try:
        # Import EarlyStopping class
        from ml.utils.early_stopping import EarlyStopping
        
        # Test initialization
        early_stopping = EarlyStopping(
            patience=5,
            min_epochs=10,
            min_delta=1e-4,
            monitor_metric='val_dice',
            mode='max',
            verbose=False
        )
        
        # Check that epochs_without_improvement exists (not 'wait')
        assert hasattr(early_stopping, 'epochs_without_improvement'), "EarlyStopping should have 'epochs_without_improvement' attribute"
        assert not hasattr(early_stopping, 'wait'), "EarlyStopping should NOT have 'wait' attribute"
        
        # Test the __call__ method returns a dictionary
        metrics = {'val_dice': 0.85, 'val_loss': 0.15}
        result = early_stopping(epoch=5, metrics=metrics, model=None)
        
        assert isinstance(result, dict), "EarlyStopping should return a dictionary"
        assert 'should_stop' in result, "Result should contain 'should_stop' key"
        assert 'epochs_without_improvement' in result, "Result should contain 'epochs_without_improvement' key"
        
        # Test attributes are accessible
        patience = early_stopping.patience
        best_metric = early_stopping.best_metric
        best_epoch = early_stopping.best_epoch
        epochs_no_improvement = early_stopping.epochs_without_improvement
        
        print("   ✅ EarlyStopping class correctly implemented")
        print(f"   ✅ Has 'epochs_without_improvement' attribute: {epochs_no_improvement}")
        print(f"   ✅ Does NOT have 'wait' attribute")
        print(f"   ✅ __call__ returns dictionary: {result}")
        
        # Now check train.py doesn't reference 'wait' attribute
        train_file = '/app/ml/training/train.py'  # Use container path
        with open(train_file, 'r') as f:
            train_content = f.read()
        
        # Check that old 'wait' reference is fixed
        assert 'early_stopping.wait' not in train_content, "train.py should not reference 'early_stopping.wait'"
        assert 'early_stopping.epochs_without_improvement' in train_content, "train.py should reference 'early_stopping.epochs_without_improvement'"
        
        print("   ✅ train.py correctly references 'epochs_without_improvement'")
        print("   ✅ No remaining 'wait' attribute references found")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in EarlyStopping fix test: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 TESTING CLASSIFICATION MODEL FIXES")
    print("=" * 60)
    
    results = []
    
    # Test 1: MLflow step logging
    results.append(("MLflow step logging", test_mlflow_step_logging()))
    
    # Test 2: Classification metrics
    results.append(("Classification metrics", test_classification_metrics()))
    
    # Test 3: Enhanced model loading
    results.append(("Enhanced model loading", test_enhanced_model_loading()))
    
    # Test 4: Model loading functionality
    checkpoint_path = create_test_model_checkpoint()
    results.append(("Model loading functionality", test_model_loading_functionality(checkpoint_path)))
    
    # Test 5: EarlyStopping fix
    results.append(("EarlyStopping fix", test_early_stopping_fix()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status:<12} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All classification fixes are working correctly!")
    else:
        print("⚠️ Some fixes need attention.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
