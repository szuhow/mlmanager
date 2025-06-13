"""
Configuration for end-to-end tests
"""
import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

@pytest.fixture(scope="session")
def browser():
    """Browser fixture for e2e tests"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)
    yield driver
    driver.quit()

@pytest.fixture
def live_server_url():
    """Live server URL for e2e tests"""
    return "http://localhost:8000"

@pytest.fixture
def test_user_credentials():
    """Test user credentials"""
    return {
        'username': 'testuser',
        'password': 'testpass123'
    }
