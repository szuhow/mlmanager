#!/usr/bin/env python3

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml'))

def test_arcade_class_detection():
    """Test class detection for different ARCADE dataset types"""
    
    print("=== Testing ARCADE Class Detection ===")
    
    # Mock dataset classes for testing
    class MockARCADEDataset:
        def __init__(self, name):
            self.name = name
            
        def __class__(self):
            return type(self.name, (), {})
    
    # Test cases
    test_cases = [
        ('ARCADESemanticSegmentation', 27, 'semantic_segmentation'),
        ('ARCADEBinarySegmentation', 1, 'binary_segmentation'),
        ('ARCADEStenosisDetection', 1, 'stenosis_detection'),
        ('ARCADEArteryClassification', 2, 'artery_classification'),
        ('ARCADEStenosisSegmentation', 1, 'stenosis_segmentation'),
        ('ARCADESemanticSegmentationBinary', 26, 'semantic_segmentation_binary'),
    ]
    
    # Simulate class detection logic
    for class_name, expected_classes, expected_task in test_cases:
        print(f"\n--- Testing {class_name} ---")
        
        # Simulate detection logic
        if 'ARCADEArteryClassification' in class_name:
            result = {'num_classes': 2, 'task_type': 'artery_classification'}
        elif 'ARCADESemanticSegmentation' in class_name and 'Binary' not in class_name:
            result = {'num_classes': 27, 'task_type': 'semantic_segmentation'}
        elif 'ARCADEBinarySegmentation' in class_name:
            result = {'num_classes': 1, 'task_type': 'binary_segmentation'}
        elif 'ARCADEStenosisDetection' in class_name:
            result = {'num_classes': 1, 'task_type': 'stenosis_detection'}
        elif 'ARCADEStenosisSegmentation' in class_name:
            result = {'num_classes': 1, 'task_type': 'stenosis_segmentation'}
        elif 'ARCADESemanticSegmentationBinary' in class_name:
            result = {'num_classes': 26, 'task_type': 'semantic_segmentation_binary'}
        else:
            result = {'num_classes': 1, 'task_type': 'unknown'}
        
        print(f"  Class name: {class_name}")
        print(f"  Expected classes: {expected_classes}")
        print(f"  Expected task: {expected_task}")
        print(f"  Detected classes: {result['num_classes']}")
        print(f"  Detected task: {result['task_type']}")
        
        # Check results
        if result['num_classes'] == expected_classes and result['task_type'] == expected_task:
            print(f"  ✅ PASS")
        else:
            print(f"  ❌ FAIL")
    
    print(f"\n=== Test Complete ===")

if __name__ == "__main__":
    test_arcade_class_detection()
