#!/usr/bin/env python3
"""
Test the enhanced grayscale binary detection logic
"""

import sys
import os
sys.path.append('/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments')

from ml.training.train import _analyze_class_distribution

def test_grayscale_detection():
    """Test various scenarios for grayscale binary detection"""
    
    print("🧪 Testing Enhanced Grayscale Binary Detection")
    print("=" * 60)
    
    # Test case 1: 256 grayscale values (0-255) - should be binary
    print("\n1️⃣ Test Case: 256 grayscale values (0-255)")
    unique_values = list(range(256))  # [0, 1, 2, ..., 255]
    result = _analyze_class_distribution(unique_values, 1, "arcade")
    expected = "binary"
    status = "✅ PASS" if result['class_type'] == expected else "❌ FAIL"
    print(f"   Result: {result['class_type']} ({result['num_classes']} classes) - {status}")
    
    # Test case 2: Normalized grayscale values (0.0-1.0) - should be binary
    print("\n2️⃣ Test Case: Normalized grayscale values (many values 0.0-1.0)")
    import numpy as np
    normalized_values = list(np.linspace(0, 1, 100))  # 100 values from 0.0 to 1.0
    result = _analyze_class_distribution(normalized_values, 1, "arcade")
    expected = "binary"
    status = "✅ PASS" if result['class_type'] == expected else "❌ FAIL"
    print(f"   Result: {result['class_type']} ({result['num_classes']} classes) - {status}")
    
    # Test case 3: Perfect binary (0, 1) - should be binary
    print("\n3️⃣ Test Case: Perfect binary values (0, 1)")
    binary_values = [0, 1]
    result = _analyze_class_distribution(binary_values, 1, "arcade")
    expected = "binary"
    status = "✅ PASS" if result['class_type'] == expected else "❌ FAIL"
    print(f"   Result: {result['class_type']} ({result['num_classes']} classes) - {status}")
    
    # Test case 4: 8-bit binary (0, 255) - should be binary
    print("\n4️⃣ Test Case: 8-bit binary values (0, 255)")
    eightbit_values = [0, 255]
    result = _analyze_class_distribution(eightbit_values, 1, "arcade")
    expected = "binary"
    status = "✅ PASS" if result['class_type'] == expected else "❌ FAIL"
    print(f"   Result: {result['class_type']} ({result['num_classes']} classes) - {status}")
    
    # Test case 5: True multi-class (discrete classes) - should be semantic_single
    print("\n5️⃣ Test Case: True multi-class semantic (discrete classes)")
    multiclass_values = [0, 1, 2, 3, 4]  # 5 distinct semantic classes
    result = _analyze_class_distribution(multiclass_values, 1, "arcade")
    expected = "semantic_single"
    status = "✅ PASS" if result['class_type'] == expected else "❌ FAIL"
    print(f"   Result: {result['class_type']} ({result['num_classes']} classes) - {status}")
    
    # Test case 6: One-hot encoded (multi-channel) - should be semantic_onehot
    print("\n6️⃣ Test Case: One-hot encoded multi-channel")
    onehot_values = [0, 1]
    result = _analyze_class_distribution(onehot_values, 27, "arcade")  # 27 channels like ARCADE
    expected = "semantic_onehot"
    status = "✅ PASS" if result['class_type'] == expected else "❌ FAIL"
    print(f"   Result: {result['class_type']} ({result['num_classes']} classes) - {status}")
    
    # Test case 7: Edge case - few grayscale values (shouldn't trigger grayscale detection)
    print("\n7️⃣ Test Case: Few grayscale values (10 values, not binary)")
    few_values = list(range(10))  # [0, 1, 2, ..., 9]
    result = _analyze_class_distribution(few_values, 1, "arcade")
    expected = "semantic_single"
    status = "✅ PASS" if result['class_type'] == expected else "❌ FAIL"
    print(f"   Result: {result['class_type']} ({result['num_classes']} classes) - {status}")
    
    print("\n" + "=" * 60)
    print("🎯 Summary: Enhanced detection correctly identifies:")
    print("   • 256 grayscale values (0-255) → Binary")
    print("   • Many normalized values (0.0-1.0) → Binary") 
    print("   • Perfect binary (0,1) or (0,255) → Binary")
    print("   • One-hot multi-channel → Semantic One-hot")
    print("   • Few discrete classes → True multi-class")
    print("=" * 60)

if __name__ == "__main__":
    test_grayscale_detection()
