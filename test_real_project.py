#!/usr/bin/env python3
"""
Test Vertex AI with the correct project ID
"""

import os
import json
import requests
from google.auth import default
from google.auth.transport.requests import Request

def test_with_real_project():
    """Test Vertex AI with your actual project"""
    
    try:
        # Get credentials and project ID
        credentials, detected_project_id = default()
        credentials.refresh(Request())
        access_token = credentials.token
        
        # Use the project ID we know from setup
        project_id = detected_project_id or "the-racer-461804-s1"
        
        print(f"✅ Authentication successful!")
        print(f"📋 Project ID: {project_id}")
        print(f"🔑 Access token: {access_token[:20]}...")
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False
    
    # Test the gemini-2.5-flash-lite-preview-06-17 model specifically
    model = "gemini-2.5-flash-lite-preview-06-17"
    
    base_url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/publishers/google/models"
    
    print(f"\n=== Testing {model} ===")
    url = f"{base_url}/{model}:generateContent"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Hello! What is 2+2?"}]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 100
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            
            # Extract the response text
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if len(parts) > 0 and "text" in parts[0]:
                        print(f"🤖 Response: {parts[0]['text']}")
            
            print(f"📄 Full response: {json.dumps(result, indent=2)}")
            
        elif response.status_code == 403:
            error_data = response.json()
            if "SERVICE_DISABLED" in str(error_data):
                print("⚠️  Need to enable Vertex AI API")
                print(f"🔗 Enable here: https://console.developers.google.com/apis/api/aiplatform.googleapis.com/overview?project={project_id}")
            else:
                print(f"❌ Permission denied: {response.text}")
                
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    print("Testing Vertex AI with your authenticated project...")
    test_with_real_project()
