#!/usr/bin/env python3
"""
Test script to verify that training logs are no longer stored in database
and only file paths/status messages are saved.
"""

import os
import sys
import django

# Add the core directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from apps.ml_manager.models import MLModel
from apps.ml_manager.tasks.tasks import read_process_output
import tempfile

def test_read_process_output():
    """Test that read_process_output returns file path, not content"""
    print("🧪 Testing read_process_output function...")
    
    # Create a temporary log file with some content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        f.write("This is a test log line 1\n")
        f.write("This is a test log line 2\n") 
        f.write("This is a test log line 3\n")
        log_file_path = f.name
    
    try:
        # Test the function
        success, message, result = read_process_output(log_file_path, max_size_mb=1)
        
        print(f"✅ Success: {success}")
        print(f"✅ Message: {message}")
        print(f"✅ Result type: {type(result)}")
        print(f"✅ Result content: {result}")
        
        # Verify that result is a file path, not the content
        if isinstance(result, str) and result.endswith('.log'):
            print("✅ PASS: Function returns file path instead of content")
        else:
            print("❌ FAIL: Function does not return file path")
            
        return success
        
    finally:
        # Clean up
        if os.path.exists(log_file_path):
            os.unlink(log_file_path)

def test_database_training_logs():
    """Test that database training_logs contain only status messages, not full content"""
    print("\n🧪 Testing database training_logs content...")
    
    # Get some models and check their training_logs
    models = MLModel.objects.all()[:5]  # Check first 5 models
    
    for model in models:
        if model.training_logs:
            print(f"\n📋 Model {model.id} ({model.name}):")
            print(f"   Training logs length: {len(model.training_logs)} chars")
            print(f"   Content preview: {model.training_logs[:200]}...")
            
            # Check if it looks like a status message (short) or full logs (long)
            if len(model.training_logs) < 500:  # Status messages should be short
                print("   ✅ PASS: Looks like status message (short)")
            else:
                print("   ❌ FAIL: Looks like full log content (long)")
        else:
            print(f"📋 Model {model.id} ({model.name}): No training logs")

def main():
    print("🔍 Testing log storage changes...\n")
    
    # Test 1: read_process_output function
    test_read_process_output()
    
    # Test 2: Database content
    test_database_training_logs()
    
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    main()
