"""
Example client for Voice AI Agent API
"""

import requests
import json
import os

class VoiceAIClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"
    
    def process_text(self, command):
        """Process a text command"""
        response = requests.post(
            f"{self.api_url}/process",
            json={"type": "text", "command": command}
        )
        return response.json()
    
    def process_audio(self, audio_file_path):
        """Process an audio file"""
        with open(audio_file_path, 'rb') as f:
            files = {'audio': f}
            response = requests.post(f"{self.api_url}/process", files=files)
        return response.json()
    
    def classify_only(self, command):
        """Classify without execution"""
        response = requests.post(
            f"{self.api_url}/classify",
            json={"command": command}
        )
        return response.json()

# Usage example
if __name__ == "__main__":
    client = VoiceAIClient()
    
    # Text command
    result = client.classify_only("make the volume up")
    print(f"Result: {result}")
    
    # Check what action to take
    if result['command'] == 'open_app':
        print(f"Opening app: {result['app_name']}")
    elif result['command'] == 'settings':
        print(f"Changing setting: {result['settings_action']}")