#!/usr/bin/env python3
"""
Test script to verify the Voice AI Agent web app is working correctly
"""

import requests
import json
import time
import subprocess
import sys
import os

def test_api_endpoints():
    """Test the API endpoints to ensure they work"""
    base_url = "http://localhost:5000"

    print("🧪 Testing Voice AI Agent API endpoints...")

    # Test 1: Health check
    try:
        print("\n1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

    # Test 2: Manual command processing
    try:
        print("\n2. Testing manual command processing...")
        payload = {"command": "open youtube"}
        response = requests.post(f"{base_url}/process_manual",
                               json=payload,
                               headers={'Content-Type': 'application/json'})
        if response.status_code == 200:
            data = response.json()
            print("✅ Manual command processing passed")
            print(f"   Intent: {data.get('intent', 'N/A')}")
        else:
            print(f"❌ Manual command failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Manual command error: {e}")
        return False

    # Test 3: CORS headers
    try:
        print("\n3. Testing CORS headers...")
        response = requests.options(f"{base_url}/process_manual",
                                  headers={'Origin': 'http://localhost:3000',
                                          'Access-Control-Request-Method': 'POST'})
        cors_headers = response.headers.get('Access-Control-Allow-Origin')
        if cors_headers:
            print("✅ CORS headers present")
        else:
            print("⚠️  CORS headers not found (may still work)")
    except Exception as e:
        print(f"⚠️  CORS test error: {e}")

    print("\n🎉 All API tests passed!")
    return True

def start_server():
    """Start the Flask server in the background"""
    print("🚀 Starting Flask server...")
    try:
        # Start server in background
        process = subprocess.Popen([
            sys.executable, "voice_ai_agent/web_app.py"
        ], cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for server to start
        time.sleep(5)

        return process
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None

def main():
    print("🔧 Voice AI Agent Web App Test Suite")
    print("=" * 50)

    # Check if virtual environment is activated
    if 'voice' not in sys.executable:
        print("⚠️  Virtual environment not detected!")
        print("Please run: voice\\Scripts\\activate")
        return

    # Start server
    server_process = start_server()
    if not server_process:
        return

    try:
        # Run tests
        success = test_api_endpoints()

        if success:
            print("\n🎊 All tests passed! Your web app is working correctly.")
            print("\n📋 Next steps:")
            print("1. Open your browser to: http://localhost:5000")
            print("2. Try the manual command input")
            print("3. Test voice recording (may need HTTPS for microphone)")
        else:
            print("\n❌ Some tests failed. Check the error messages above.")

    finally:
        # Clean up server
        print("\n🛑 Stopping server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    main()