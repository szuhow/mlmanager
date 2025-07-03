#!/usr/bin/env python3
"""
Test the redirect fix for the 'Start training & monitor' button
"""
import os
import sys
import django

# Setup Django
sys.path.append('core')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
django.setup()

from django.test import Client
from django.urls import reverse

def test_redirect_logic():
    """Test the redirect logic with proper form data"""
    print("🧪 Testing redirect logic fix...")
    
    client = Client()
    login_success = client.login(username='admin', password='admin123')
    
    if not login_success:
        print("❌ Login failed!")
        return
    
    # Test data for creating a training - mimicking the real training that was submitted
    form_data = {
        'name': 'Test Redirect Logic',
        'model_type': 'deep_resunet_attention',
        'epochs': '10',
        'batch_size': '32',
        'learning_rate': '0.001',
        'data_path': '/app/data/datasets/',
        'dataset_type': 'arcade_binary',
        'validation_split': '0.2',
        'resolution': '256',
        'device': 'auto',
        'optimizer': 'adam',
        'lr_scheduler': 'plateau',
        'num_workers': '0',
        'loss_function': 'dice',
        'checkpoint_strategy': 'best_val_dice',
        'max_checkpoints': '3',
        'monitor_metric': 'val_dice',
        'redirect_to_list': 'false'  # This should redirect to model detail
    }
    
    print("📋 Submitting training form with redirect_to_list='false'...")
    response = client.post(reverse('ml_manager:start-training'), form_data, follow=False)
    
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 302:
        redirect_url = response.get('Location', '')
        print(f"Redirect URL: {redirect_url}")
        
        if '/model/' in redirect_url and not redirect_url.endswith('/ml/'):
            print("✅ SUCCESS: Correctly redirects to model detail page")
            return True
        elif redirect_url.endswith('/ml/'):
            print("❌ ISSUE: Still redirects to model list instead of model detail")
            return False
        else:
            print(f"⚠️  Unexpected redirect: {redirect_url}")
            return False
    else:
        print(f"❌ Not a redirect response: {response.status_code}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Redirect Fix")
    print("=" * 40)
    success = test_redirect_logic()
    print("=" * 40)
    if success:
        print("🎉 Redirect fix is working!")
    else:
        print("💥 Redirect fix needs more work")
