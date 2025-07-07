#!/usr/bin/env python3
"""
Test CSRF and session authentication for progress API
"""
import requests
import json

# Start session
session = requests.Session()

# Get login page to get CSRF token
login_url = "http://localhost:8000/accounts/login/"
login_page = session.get(login_url)

if login_page.status_code != 200:
    print(f"ERROR: Cannot access login page: {login_page.status_code}")
    exit(1)

# Extract CSRF token from login page
csrf_token = None
for line in login_page.text.split('\n'):
    if 'csrfmiddlewaretoken' in line and 'value=' in line:
        csrf_token = line.split('value="')[1].split('"')[0]
        break

if not csrf_token:
    print("ERROR: Could not find CSRF token in login page")
    exit(1)

print(f"Found CSRF token: {csrf_token[:20]}...")

# Login with admin credentials (adjust as needed)
login_data = {
    'username': 'admin',  # Change this to your username
    'password': 'admin123',  # Change this to your password
    'csrfmiddlewaretoken': csrf_token
}

login_response = session.post(login_url, data=login_data)
print(f"Login response status: {login_response.status_code}")

# Test progress API with authenticated session
progress_url = "http://localhost:8000/ml/model/92/progress/"
progress_response = session.get(progress_url)

print(f"\nProgress API response status: {progress_response.status_code}")

if progress_response.status_code == 200:
    try:
        data = progress_response.json()
        print("Progress API response:")
        print(json.dumps(data, indent=2))
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print("Response text:")
        print(progress_response.text[:500])
else:
    print(f"Progress API failed: {progress_response.status_code}")
    print("Response text:")
    print(progress_response.text[:500])
