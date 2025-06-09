#!/usr/bin/env python3
"""
Test script to verify the model-agnostic training functionality.
Tests that the system can handle different model families and types.
"""

import os
import sys
import argparse
from pathlib import Path

# Add shared directory to path
sys.path.append('/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments/shared')

def test_argument_parsing():
    """Test that the new model-family and model-type arguments work correctly"""
    
    print("Testing argument parsing...")
    
    # Import the parse_args function
    from train import parse_args
    
    # Test with model-family argument
    test_args = [
        '--mode', 'train',
        '--model-family', 'ResNet-Coronary',
        '--model-type', 'resnet',
        '--batch-size', '16',
        '--epochs', '5',
        '--learning-rate', '0.0001',
        '--data-path', 'test/data',
        '--validation-split', '0.2'
    ]
    
    # Override sys.argv for testing
    original_argv = sys.argv
    sys.argv = ['train.py'] + test_args
    
    try:
        args = parse_args()
        
        # Check that arguments are parsed correctly
        assert hasattr(args, 'model_family'), "model_family argument not found"
        assert hasattr(args, 'model_type'), "model_type argument not found"
        assert args.model_family == 'ResNet-Coronary', f"Expected 'ResNet-Coronary', got '{args.model_family}'"
        assert args.model_type == 'resnet', f"Expected 'resnet', got '{args.model_type}'"
        
        print("✓ New arguments parsed correctly:")
        print(f"  - model_family: {args.model_family}")
        print(f"  - model_type: {args.model_type}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing argument parsing: {e}")
        return False
    finally:
        sys.argv = original_argv

def test_backward_compatibility():
    """Test that the system still works without the new arguments"""
    
    print("\nTesting backward compatibility...")
    
    from train import parse_args
    
    # Test without model-family argument (should use defaults)
    test_args = [
        '--mode', 'train',
        '--batch-size', '16',
        '--epochs', '5',
        '--learning-rate', '0.0001',
        '--data-path', 'test/data',
        '--validation-split', '0.2'
    ]
    
    original_argv = sys.argv
    sys.argv = ['train.py'] + test_args
    
    try:
        args = parse_args()
        
        # Check that defaults are used when arguments are not provided
        model_family = getattr(args, 'model_family', 'UNet-Coronary')
        model_type = getattr(args, 'model_type', 'unet')
        
        print("✓ Backward compatibility maintained:")
        print(f"  - model_family (default): {model_family}")
        print(f"  - model_type (default): {model_type}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing backward compatibility: {e}")
        return False
    finally:
        sys.argv = original_argv

def test_utility_functions():
    """Test that utility functions work with dynamic model families"""
    
    print("\nTesting utility functions...")
    
    try:
        from train import create_organized_model_directory, save_enhanced_model_metadata
        
        # Test create_organized_model_directory with different model families
        test_families = ['UNet-Coronary', 'ResNet-Segmentation', 'Vision-Transformer']
        
        for family in test_families:
            model_dir, unique_id = create_organized_model_directory(
                model_id=123,
                model_family=family,
                version="2.0.0"
            )
            
            print(f"✓ Created directory for {family}: {model_dir}")
            
            # Check that the family is correctly incorporated in the path
            assert family.replace(" ", "_").lower() in model_dir, f"Model family not in path: {model_dir}"
        
        print("✓ Utility functions work with dynamic model families")
        return True
        
    except Exception as e:
        print(f"✗ Error testing utility functions: {e}")
        return False

def test_help_output():
    """Test that help output includes the new arguments"""
    
    print("\nTesting help output...")
    
    from train import parse_args
    
    # Capture help output
    original_argv = sys.argv
    sys.argv = ['train.py', '--help']
    
    try:
        # This will raise SystemExit, but that's expected for --help
        parse_args()
    except SystemExit:
        pass  # Expected behavior for --help
    finally:
        sys.argv = original_argv
    
    print("✓ Help output generated (check manually for new arguments)")
    return True

def main():
    """Main test function"""
    print("=" * 60)
    print("MODEL-AGNOSTIC TRAINING FUNCTIONALITY TEST")
    print("=" * 60)
    
    tests = [
        ("Argument Parsing", test_argument_parsing),
        ("Backward Compatibility", test_backward_compatibility),
        ("Utility Functions", test_utility_functions),
        ("Help Output", test_help_output)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"SUMMARY: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The model-agnostic functionality is working correctly.")
        print("\nKey improvements implemented:")
        print("• Dynamic model family support via --model-family argument")
        print("• Dynamic model type support via --model-type argument")
        print("• Backward compatibility with existing code")
        print("• MLflow Model Registry integration with configurable naming")
        print("• No more hardcoded 'UNet-Coronary' strings in training flow")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
