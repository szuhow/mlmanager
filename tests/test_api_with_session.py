#!/usr/bin/env python3

import requests
import json

# Create a session
session = requests.Session()

# Get login page to get CSRF token
login_url = 'http://localhost:8000/accounts/login/'
response = session.get(login_url)
csrf_token = session.cookies.get('csrftoken')

print(f"CSRF Token: {csrf_token}")

# Try to access progress API without login (should redirect)
progress_url = 'http://localhost:8000/ml/model/92/progress/'
response = session.get(progress_url)
print(f"Status without login: {response.status_code}")
print(f"Response headers: {dict(response.headers)}")

if response.status_code == 302:
    print("Redirected to login (expected)")
    print(f"Location: {response.headers.get('Location', 'Not provided')}")

# Test if we can get a basic page
home_response = session.get('http://localhost:8000/')
print(f"Home page status: {home_response.status_code}")
