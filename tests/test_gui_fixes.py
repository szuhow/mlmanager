#!/usr/bin/env python3
"""
Test script for GUI fixes validation
Tests both Safari dropdown fixes and auto-refresh improvements
"""

import os
import sys
import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.safari.options import Options as SafariOptions

def test_safari_dropdown_fixes():
    """Test Safari dropdown positioning and z-index fixes"""
    print("🧪 Testing Safari dropdown fixes...")
    
    try:
        # Configure Safari for testing
        safari_options = SafariOptions()
        driver = webdriver.Safari(options=safari_options)
        driver.set_window_size(1200, 800)
        
        # Navigate to model list page
        driver.get("http://localhost:8000/ml/models/")
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "custom-dropdown"))
        )
        
        print("✅ Page loaded successfully")
        
        # Find first dropdown
        dropdown = driver.find_element(By.CLASS_NAME, "custom-dropdown")
        toggle = dropdown.find_element(By.CLASS_NAME, "custom-dropdown-toggle")
        menu = dropdown.find_element(By.CLASS_NAME, "custom-dropdown-menu")
        
        # Test dropdown opening
        print("🔧 Testing dropdown opening...")
        toggle.click()
        
        # Wait for dropdown to be visible
        WebDriverWait(driver, 5).until(
            EC.visibility_of(menu)
        )
        
        # Check if menu is properly positioned
        menu_rect = driver.execute_script("return arguments[0].getBoundingClientRect();", menu)
        toggle_rect = driver.execute_script("return arguments[0].getBoundingClientRect();", toggle)
        
        print(f"Toggle position: {toggle_rect}")
        print(f"Menu position: {menu_rect}")
        
        # Verify menu is visible and positioned correctly
        assert menu_rect['top'] >= toggle_rect['bottom'] - 5, "Menu should be below toggle"
        assert menu_rect['left'] >= toggle_rect['left'] - 100, "Menu should be near toggle horizontally"
        
        # Check z-index
        menu_z_index = driver.execute_script("return window.getComputedStyle(arguments[0]).zIndex;", menu)
        print(f"Menu z-index: {menu_z_index}")
        assert int(menu_z_index) > 1000, "Menu should have high z-index"
        
        print("✅ Safari dropdown positioning test passed")
        
        # Test dropdown closing
        driver.execute_script("document.body.click();")
        time.sleep(0.5)
        
        assert not menu.is_displayed(), "Menu should close when clicking outside"
        print("✅ Safari dropdown closing test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Safari dropdown test failed: {e}")
        return False
    finally:
        if 'driver' in locals():
            driver.quit()

def test_auto_refresh_pending_models():
    """Test auto-refresh functionality for pending models"""
    print("🧪 Testing auto-refresh for pending models...")
    
    try:
        # Use Chrome for this test (more stable for automation)
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        chrome_options = ChromeOptions()
        chrome_options.add_argument('--headless')
        driver = webdriver.Chrome(options=chrome_options)
        
        # Navigate to model list page
        driver.get("http://localhost:8000/ml/models/")
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "tbody"))
        )
        
        print("✅ Page loaded successfully")
        
        # Check if auto-refresh starts for pending models
        pending_models = driver.find_elements(By.CSS_SELECTOR, 'tr[data-model-status="pending"]')
        training_models = driver.find_elements(By.CSS_SELECTOR, 'tr[data-model-status="training"]')
        
        print(f"Found {len(pending_models)} pending models and {len(training_models)} training models")
        
        if len(pending_models) > 0 or len(training_models) > 0:
            # Check if auto-refresh indicator appears
            try:
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.ID, "auto-refresh-indicator"))
                )
                indicator = driver.find_element(By.ID, "auto-refresh-indicator")
                indicator_text = indicator.text
                
                print(f"✅ Auto-refresh indicator found: {indicator_text}")
                
                # Check if fast mode is active for pending models
                if len(pending_models) > 0:
                    assert "fast mode" in indicator_text.lower(), "Fast mode should be active for pending models"
                    print("✅ Fast mode active for pending models")
                
                return True
                
            except Exception as e:
                print(f"⚠️  Auto-refresh indicator not found: {e}")
                return False
        else:
            print("ℹ️  No active models found - auto-refresh test skipped")
            return True
            
    except Exception as e:
        print(f"❌ Auto-refresh test failed: {e}")
        return False
    finally:
        if 'driver' in locals():
            driver.quit()

def test_status_change_notifications():
    """Test status change notifications"""
    print("🧪 Testing status change notifications...")
    
    # This is a simulated test since we can't easily trigger real status changes
    try:
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        chrome_options = ChromeOptions()
        chrome_options.add_argument('--headless')
        driver = webdriver.Chrome(options=chrome_options)
        
        driver.get("http://localhost:8000/ml/models/")
        
        # Inject test notification
        driver.execute_script("""
            window.showStatusChangeNotification('Test Notification', 'This is a test message', 'success');
        """)
        
        # Wait for notification to appear
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".alert-success"))
        )
        
        notification = driver.find_element(By.CSS_SELECTOR, ".alert-success")
        assert "Test Notification" in notification.text
        
        print("✅ Status change notification test passed")
        return True
        
    except Exception as e:
        print(f"❌ Status change notification test failed: {e}")
        return False
    finally:
        if 'driver' in locals():
            driver.quit()

def main():
    """Run all GUI tests"""
    print("🚀 Starting GUI fixes validation tests...")
    print("=" * 50)
    
    results = {
        'safari_dropdown': False,
        'auto_refresh': False,
        'notifications': False
    }
    
    # Test Safari dropdown fixes
    results['safari_dropdown'] = test_safari_dropdown_fixes()
    print()
    
    # Test auto-refresh improvements
    results['auto_refresh'] = test_auto_refresh_pending_models()
    print()
    
    # Test status change notifications
    results['notifications'] = test_status_change_notifications()
    print()
    
    # Summary
    print("=" * 50)
    print("📊 Test Results Summary:")
    print(f"Safari Dropdown Fixes: {'✅ PASSED' if results['safari_dropdown'] else '❌ FAILED'}")
    print(f"Auto-refresh for Pending: {'✅ PASSED' if results['auto_refresh'] else '❌ FAILED'}")
    print(f"Status Notifications: {'✅ PASSED' if results['notifications'] else '❌ FAILED'}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n🎯 Overall: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("🎉 All GUI fixes are working correctly!")
        return 0
    else:
        print("⚠️  Some issues detected - please review the failed tests")
        return 1

if __name__ == "__main__":
    sys.exit(main())
