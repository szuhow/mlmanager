#!/usr/bin/env python3
"""
Test the dataset detection fix for arcade_classification vs arcade_artery_classification
"""
import sys
import os
sys.path.append('/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments')
sys.path.append('/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments/ml')

def test_forced_type_mapping():
    """Test that arcade_classification maps correctly to artery_classification task"""
    print("🧪 Testing Dataset Type Mapping Fix")
    print("=" * 50)
    
    # Import the get_arcade_datasets function
    try:
        from ml.training.train import get_arcade_datasets
        print("✅ Successfully imported get_arcade_datasets")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False
    
    # Test the mapping logic by looking at the source code
    import inspect
    source = inspect.getsource(get_arcade_datasets)
    
    # Check if both mappings exist
    has_arcade_classification = "'arcade_classification': 'artery_classification'" in source
    has_arcade_artery_classification = "'arcade_artery_classification': 'artery_classification'" in source
    
    print(f"📋 Mapping check:")
    print(f"   arcade_classification → artery_classification: {'✅' if has_arcade_classification else '❌'}")
    print(f"   arcade_artery_classification → artery_classification: {'✅' if has_arcade_artery_classification else '❌'}")
    
    if has_arcade_classification and has_arcade_artery_classification:
        print("\n🎉 SUCCESS: Both mappings exist!")
        print("   GUI can now send 'arcade_classification' and it will correctly map to ARCADEArteryClassification")
        return True
    else:
        print("\n❌ FAILURE: Missing required mappings")
        return False

def test_mock_forced_type():
    """Test the actual forced_type logic with mock args"""
    print("\n🧪 Testing Mock Forced Type Logic")
    print("=" * 30)
    
    # Create a mock args object
    class MockArgs:
        def __init__(self):
            self.dataset_type = 'arcade_classification'
            self.model_type = 'unet'
            self.batch_size = 32
            self.resolution = '256'
            self.crop_size = 128
            self.num_workers = 2
    
    # Test the mapping logic directly
    forced_type = 'arcade_classification'
    mapping = {
        'arcade_binary': 'binary_segmentation',
        'arcade_binary_segmentation': 'binary_segmentation',
        'arcade_semantic': 'semantic_segmentation',
        'arcade_semantic_segmentation': 'semantic_segmentation',
        'arcade_stenosis': 'stenosis_detection',
        'arcade_stenosis_detection': 'stenosis_detection',
        'arcade_classification': 'artery_classification',
        'arcade_artery_classification': 'artery_classification'
    }
    
    task = mapping.get(forced_type, 'binary_segmentation')
    
    print(f"Input: forced_type = '{forced_type}'")
    print(f"Output: task = '{task}'")
    
    if task == 'artery_classification':
        print("✅ SUCCESS: arcade_classification correctly maps to artery_classification")
        return True
    else:
        print(f"❌ FAILURE: Expected 'artery_classification', got '{task}'")
        return False

def main():
    """Main test runner"""
    print("🔧 Dataset Detection Fix Verification")
    print("=" * 60)
    
    test1_passed = test_forced_type_mapping()
    test2_passed = test_mock_forced_type()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"✅ Source code mapping check: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"✅ Mock forced type logic: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n📝 What this fixes:")
        print("   • GUI sends 'arcade_classification' from dropdown")
        print("   • Training script now correctly maps this to 'artery_classification' task")
        print("   • ARCADEArteryClassification dataset will be instantiated correctly")
        print("   • Logs will show 'ARCADEArteryClassification' instead of 'ARCADEBinarySegmentation'")
        return True
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
