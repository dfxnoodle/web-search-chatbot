#!/usr/bin/env python3
"""
Quick test script for Vertex AI - run this after enabling billing
"""

import requests
import json
from google.auth import default
from google.auth.transport.requests import Request

def quick_test():
    """Quick test of gemini-2.5-flash-lite-preview-06-17 model"""
    
    # Get authentication
    credentials, project_id = default()
    if not project_id:
        project_id = "the-racer-461804-s1"
    
    credentials.refresh(Request())
    access_token = credentials.token
    
    print(f"🔧 Testing gemini-2.5-flash-lite-preview-06-17 with project: {project_id}")
    
    # Focus on the specific model you want to use
    model = "gemini-2.5-flash-lite-preview-06-17"
    
    print(f"\n📡 Testing {model}...")
    
    url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/publishers/google/models/{model}:generateContent"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Say hello and tell me what 5+3 equals."}]
            }
        ],
        "generationConfig": {
            "temperature": 0.5,
            "maxOutputTokens": 50
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    text = candidate["content"]["parts"][0]["text"]
                    print(f"✅ SUCCESS! Response: {text.strip()}")
                    status = "✅ Working"
                else:
                    print("✅ Response received but no text found")
                    status = "✅ Working (no text)"
            else:
                print("✅ Response received but no candidates")
                status = "✅ Working (no candidates)"
                
        elif response.status_code == 403:
            error_msg = response.json().get('error', {}).get('message', '')
            if 'billing' in error_msg.lower():
                print("❌ Billing not enabled")
                status = "❌ Need billing"
            elif 'api' in error_msg.lower() and 'disabled' in error_msg.lower():
                print("❌ API not enabled")
                status = "❌ Need API enabled"
            else:
                print(f"❌ Permission denied: {error_msg}")
                status = "❌ Permission denied"
                
        else:
            print(f"❌ Error {response.status_code}")
            status = f"❌ Error {response.status_code}"
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        status = f"❌ Request failed: {e}"
    
    # Summary
    print(f"\n{'='*60}")
    print("🔄 FINAL RESULT:")
    print(f"   {model}: {status}")
    
    # Check if working
    if "✅ Working" in status:
        print(f"\n🎉 MODEL WORKING! Your Vertex AI setup is complete!")
        return True
    else:
        print(f"\n⚠️  Issue found. Check the links in VERTEX_AI_NEXT_STEPS.md")
        return False

if __name__ == "__main__":
    print("🚀 Quick Vertex AI Test - gemini-2.5-flash-lite-preview-06-17")
    print("=" * 55)
    success = quick_test()
    
    if not success:
        print(f"\n💡 Next steps:")
        print(f"   1. Check VERTEX_AI_NEXT_STEPS.md")
        print(f"   2. Enable billing and API")
        print(f"   3. Run this test again")
