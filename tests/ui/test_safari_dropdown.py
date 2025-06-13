#!/usr/bin/env python3
"""
Test script to verify Safari dropdown functionality
"""

import os
import sys
import django
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.safari.service import Service as SafariService

# Setup Django
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings')
django.setup()

def test_safari_dropdown():
    """Test dropdown functionality in Safari"""
    print("🧪 Testing Safari Dropdown Functionality")
    print("=" * 50)
    
    try:
        # Setup Safari WebDriver
        print("🚀 Starting Safari WebDriver...")
        options = webdriver.SafariOptions()
        driver = webdriver.Safari(options=options)
        
        # Navigate to the application
        app_url = "http://localhost:8000/ml/"
        print(f"📖 Opening {app_url}")
        driver.get(app_url)
        
        # Wait for page to load
        wait = WebDriverWait(driver, 10)
        
        # Check if login is required
        if "login" in driver.current_url.lower():
            print("🔐 Login required - please log in manually")
            input("Press Enter after logging in...")
        
        # Wait for model list to load
        print("⏳ Waiting for model list to load...")
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "custom-dropdown")))
        
        # Find dropdown elements
        dropdowns = driver.find_elements(By.CLASS_NAME, "custom-dropdown")
        print(f"✅ Found {len(dropdowns)} dropdown(s)")
        
        if len(dropdowns) > 0:
            # Test first dropdown
            first_dropdown = dropdowns[0]
            toggle = first_dropdown.find_element(By.CLASS_NAME, "custom-dropdown-toggle")
            menu = first_dropdown.find_element(By.CLASS_NAME, "custom-dropdown-menu")
            
            print("🖱️  Testing dropdown click...")
            
            # Check initial state
            is_open_before = "show" in first_dropdown.get_attribute("class")
            print(f"   Initial state: {'open' if is_open_before else 'closed'}")
            
            # Click to open
            toggle.click()
            time.sleep(0.5)  # Wait for animation
            
            # Check state after click
            is_open_after = "show" in first_dropdown.get_attribute("class")
            print(f"   State after click: {'open' if is_open_after else 'closed'}")
            
            # Test menu items
            if is_open_after:
                menu_items = menu.find_elements(By.CLASS_NAME, "custom-dropdown-item")
                print(f"   Menu items found: {len(menu_items)}")
                
                # Test hover on first item
                if len(menu_items) > 0:
                    first_item = menu_items[0]
                    print(f"   Testing hover on: {first_item.text}")
                    
                    # Move to item and check if hover effect works
                    webdriver.ActionChains(driver).move_to_element(first_item).perform()
                    time.sleep(0.2)
                    
                    # Check if hover style is applied
                    bg_color = first_item.value_of_css_property("background-color")
                    print(f"   Hover background-color: {bg_color}")
            
            # Click outside to close
            print("🖱️  Testing close by clicking outside...")
            driver.find_element(By.TAG_NAME, "body").click()
            time.sleep(0.5)
            
            # Check final state
            is_open_final = "show" in first_dropdown.get_attribute("class")
            print(f"   Final state: {'open' if is_open_final else 'closed'}")
            
            # Test result
            if not is_open_before and is_open_after and not is_open_final:
                print("✅ SUCCESS: Dropdown works correctly in Safari!")
                return True
            else:
                print("❌ ISSUE: Dropdown behavior not as expected")
                return False
        else:
            print("❌ ERROR: No dropdowns found on page")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False
    finally:
        if 'driver' in locals():
            print("🧹 Closing Safari...")
            driver.quit()

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 SAFARI DROPDOWN COMPATIBILITY TEST")
    print("=" * 60)
    
    success = test_safari_dropdown()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 RESULT: SAFARI DROPDOWN IS WORKING!")
    else:
        print("🔧 RESULT: SAFARI DROPDOWN NEEDS MORE WORK")
        print("\n💡 TIPS FOR MANUAL TESTING:")
        print("   1. Open http://localhost:8000/ml/ in Safari")
        print("   2. Try clicking on 'Actions' dropdown")
        print("   3. Check if menu opens and items are clickable")
        print("   4. Try clicking outside to close")
        print("   5. Test on different Safari versions/devices")
    print("=" * 60)

if __name__ == "__main__":
    main()
