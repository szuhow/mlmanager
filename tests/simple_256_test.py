#!/usr/bin/env python3
"""
Simple test of the 256-channel fix
"""

import sys
import os
sys.path.append('/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments')

def simple_test():
    try:
        from ml.training.train import _analyze_class_distribution
        
        print("🧪 Testing 256-channel fix...")
        
        # Test the exact scenario from the error
        unique_values = list(range(256))  # [0, 1, 2, ..., 255]
        max_channels = 1
        dataset_type = "arcade"
        
        print(f"Input: {len(unique_values)} unique values (0-255)")
        
        result = _analyze_class_distribution(unique_values, max_channels, dataset_type)
        
        print(f"Output: {result}")
        
        if result['class_type'] == 'binary' and result['num_classes'] == 1:
            print("✅ SUCCESS: 256 grayscale values correctly detected as binary!")
            return True
        else:
            print(f"❌ FAILED: Expected binary/1 class, got {result['class_type']}/{result['num_classes']}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_test()
    print(f"Test {'PASSED' if success else 'FAILED'}")
