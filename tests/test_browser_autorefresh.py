"""
Test auto-refresh in browser via browser automation
"""
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def test_auto_refresh():
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Remove this to see the browser
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    
    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)
        
        # Login first
        print("Logging in...")
        driver.get("http://localhost:8000/accounts/login/")
        
        username_field = driver.find_element(By.NAME, "username")
        password_field = driver.find_element(By.NAME, "password")
        
        username_field.send_keys("admin")
        password_field.send_keys("admin123")
        
        login_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        login_button.click()
        
        # Wait for login redirect
        WebDriverWait(driver, 10).until(
            lambda d: "login" not in d.current_url
        )
        print(f"Logged in successfully. Current URL: {driver.current_url}")
        
        # Navigate to model detail page
        model_url = "http://localhost:8000/ml/model/92/"
        print(f"Navigating to {model_url}")
        driver.get(model_url)
        
        # Wait for page load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-model-status]"))
        )
        
        # Check console logs
        print("\n=== BROWSER CONSOLE LOGS ===")
        for log in driver.get_log('browser'):
            print(f"{log['level']}: {log['message']}")
        
        # Check if ModelDetailManager is initialized
        try:
            model_status = driver.find_element(By.CSS_SELECTOR, "[data-model-status]")
            status = model_status.get_attribute("data-model-status")
            print(f"\nModel status in DOM: {status}")
            
            # Check if progress bars exist
            progress_bars = driver.find_elements(By.CSS_SELECTOR, ".progress-bar")
            print(f"Found {len(progress_bars)} progress bars")
            
            # Wait and check for updates
            print("\nWaiting 5 seconds for auto-refresh...")
            time.sleep(5)
            
            print("\n=== CONSOLE LOGS AFTER WAITING ===")
            for log in driver.get_log('browser'):
                print(f"{log['level']}: {log['message']}")
                
        except Exception as e:
            print(f"Error checking page elements: {e}")
            
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    test_auto_refresh()
