#!/usr/bin/env python3
"""
Test script to check if the logs are preloaded correctly in the MLManager web UI.
"""
import requests
import sys
import time
import json
import os
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_logs_preload(model_id, base_url="http://localhost:8000"):
    """Check if the logs are preloaded correctly for a specific model."""
    model_detail_url = f"{base_url}/ml/model/{model_id}/"
    logs_url = f"{base_url}/ml/model/{model_id}/logs/"
    
    logger.info(f"Testing model: {model_id}")
    logger.info(f"Model detail URL: {model_detail_url}")
    logger.info(f"Logs API URL: {logs_url}")
    
    # Check logs API first
    try:
        logger.info("Checking logs API...")
        response = requests.get(logs_url, timeout=5)
        response.raise_for_status()
        
        try:
            data = response.json()
            log_count = len(data.get('logs', [])) if isinstance(data, dict) else 0
            logger.info(f"✅ Logs API working, found {log_count} log entries")
        except json.JSONDecodeError:
            # Some log endpoints might return plain text, which is fine
            logger.info("✅ Logs API working, returned plain text")
            
    except requests.RequestException as e:
        logger.error(f"❌ Error accessing logs API: {e}")
        return False
    
    # Now check model detail page for preload functionality
    try:
        logger.info("Checking model detail page for preload functionality...")
        response = requests.get(model_detail_url, timeout=5)
        response.raise_for_status()
        
        html = response.text
        
        # Check for key indicators that our changes are present
        indicators = {
            'preloadLogs function': 'preloadLogs',
            'displayLogsData function': 'displayLogsData',
            'Logs modal': 'id="logsModal"',
            'Logs content container': 'id="modal-logs-content"'
        }
        
        all_found = True
        for name, pattern in indicators.items():
            found = pattern in html
            status = "✅ Found" if found else "❌ Missing"
            logger.info(f"{status} {name}")
            all_found = all_found and found
        
        return all_found
    
    except requests.RequestException as e:
        logger.error(f"❌ Error accessing model detail page: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test logs preloading in MLManager UI")
    parser.add_argument("--model-id", "-m", type=int, default=88, 
                        help="ID of the model to test (default: 88)")
    parser.add_argument("--url", "-u", default="http://localhost:8000",
                        help="Base URL of the MLManager application")
    
    args = parser.parse_args()
    
    logger.info("Starting logs preload check...")
    logger.info("Waiting for server to be ready...")
    time.sleep(2)
    
    success = check_logs_preload(args.model_id, args.url)
    
    if success:
        logger.info("\n✅ All logs preload tests passed! Logs should now be displayed correctly by default.")
        return 0
    else:
        logger.error("\n❌ Some logs preload tests failed. See errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
