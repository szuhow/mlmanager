"""
Test script to verify the error handling in enhanced_inference.py

This script tests:
1. Loading different checkpoint formats
2. Handling GPU errors and falling back to CPU
3. Proper error messaging
"""

import os
import sys
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from project
from ml.utils.enhanced_inference import run_enhanced_inference

def test_inference_error_handling():
    """Test the error handling in enhanced_inference.py"""
    
    # Test image (use a sample image from the repo)
    input_image = "test_image.jpg"
    if not os.path.exists(input_image):
        logger.error(f"Test image not found: {input_image}")
        return
    
    # Output directory
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test cases
    test_cases = [
        {
            'name': "Nested checkpoint with GPU request on CPU-only system",
            'model_path': "data/models/sample_nested_checkpoint.pth",
            'device': 'cuda:0',
            'expect_success': True,  # Should work by falling back to CPU
            'config': {'model_type': 'unet', 'threshold': 0.5}
        },
        {
            'name': "Invalid checkpoint path",
            'model_path': "nonexistent_checkpoint.pth",
            'device': 'cpu',
            'expect_success': False,
            'config': {'model_type': 'unet', 'threshold': 0.5}
        },
        # Add more test cases here
    ]
    
    # Run tests
    for tc in test_cases:
        logger.info(f"\n===== Testing: {tc['name']} =====")
        
        try:
            # Check if the model file exists
            if not os.path.exists(tc['model_path']) and 'nonexistent' not in tc['model_path']:
                logger.warning(f"Model path doesn't exist: {tc['model_path']}, skipping test")
                continue
                
            # Run inference
            result = run_enhanced_inference(
                model_path=tc['model_path'],
                input_image_path=input_image,
                output_dir=output_dir,
                config=tc['config'],
                device=tc['device']
            )
            
            # Check result
            if result['success'] == tc['expect_success']:
                logger.info(f"TEST PASSED: {tc['name']}")
                logger.info(f"Result: {result['status']}")
                if not result['success']:
                    logger.info(f"Error message: {result['error_message']}")
                    if 'technical_error' in result:
                        logger.debug(f"Technical error: {result['technical_error']}")
            else:
                logger.error(f"TEST FAILED: {tc['name']}")
                logger.error(f"Expected success: {tc['expect_success']}, got: {result['success']}")
                logger.error(f"Result: {result}")
                
        except Exception as e:
            logger.error(f"Test exception: {e}")
            logger.error(f"TEST FAILED: {tc['name']}")
    
    logger.info("\n===== All tests completed =====")

if __name__ == "__main__":
    test_inference_error_handling()
