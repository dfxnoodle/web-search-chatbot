#!/usr/bin/env python3
"""
Focused test for gemini-2.5-flash-lite-preview-06-17 model
"""

import requests
import json
from google.auth import default
from google.auth.transport.requests import Request

def test_gemini_2_5_flash_lite():
    """Test the gemini-2.5-flash-lite-preview-06-17 model specifically"""
    
    # Get authentication
    credentials, project_id = default()
    if not project_id:
        project_id = "the-racer-461804-s1"
    
    credentials.refresh(Request())
    access_token = credentials.token
    
    print(f"🎯 Testing gemini-2.5-flash-lite-preview-06-17")
    print(f"📋 Project: {project_id}")
    print(f"🔑 Token: {access_token[:20]}...")
    
    # Model configuration
    model = "gemini-2.5-flash-lite-preview-06-17"
    url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/publishers/google/models/{model}:generateContent"
    
    print(f"\n🔗 Endpoint: {url}")
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    # Test with a simple prompt
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "Hello! Please respond with a greeting and solve this math problem: What is 7 × 8?"
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 150,
            "topK": 40,
            "topP": 0.95
        }
    }
    
    print(f"\n📤 Request payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        print(f"\n⏳ Sending request...")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS!")
            
            # Extract and display the response
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if len(parts) > 0 and "text" in parts[0]:
                        response_text = parts[0]["text"]
                        print(f"🤖 Model Response:")
                        print(f"   {response_text}")
                        
                        # Check for safety ratings if present
                        if "safetyRatings" in candidate:
                            print(f"\n🛡️  Safety Ratings:")
                            for rating in candidate["safetyRatings"]:
                                category = rating.get("category", "Unknown")
                                probability = rating.get("probability", "Unknown")
                                print(f"   - {category}: {probability}")
                
                # Display usage metadata if present
                if "usageMetadata" in result:
                    usage = result["usageMetadata"]
                    print(f"\n📊 Usage Statistics:")
                    if "promptTokenCount" in usage:
                        print(f"   - Prompt tokens: {usage['promptTokenCount']}")
                    if "candidatesTokenCount" in usage:
                        print(f"   - Response tokens: {usage['candidatesTokenCount']}")
                    if "totalTokenCount" in usage:
                        print(f"   - Total tokens: {usage['totalTokenCount']}")
            
            print(f"\n📄 Full Response Structure:")
            print(json.dumps(result, indent=2))
            return True
            
        else:
            print(f"\n❌ Request failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"🔍 Error details:")
                print(json.dumps(error_data, indent=2))
                
                # Provide specific guidance
                error_msg = error_data.get('error', {}).get('message', '')
                if 'billing' in error_msg.lower():
                    print(f"\n💳 Billing issue detected")
                    print(f"🔗 Enable billing: https://console.developers.google.com/billing/enable?project={project_id}")
                elif 'api' in error_msg.lower() and ('disabled' in error_msg.lower() or 'not been used' in error_msg.lower()):
                    print(f"\n🔧 API not enabled")
                    print(f"🔗 Enable API: https://console.developers.google.com/apis/api/aiplatform.googleapis.com/overview?project={project_id}")
                    
            except:
                print(f"Raw error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"\n⏰ Request timed out")
        return False
    except Exception as e:
        print(f"\n❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Gemini 2.5 Flash Lite Preview Test")
    print("=" * 45)
    
    success = test_gemini_2_5_flash_lite()
    
    if success:
        print(f"\n🎉 gemini-2.5-flash-lite-preview-06-17 is working perfectly!")
        print(f"✅ Ready for integration into your applications")
    else:
        print(f"\n💡 Troubleshooting steps:")
        print(f"   1. Check billing is enabled")
        print(f"   2. Verify Vertex AI API is enabled")
        print(f"   3. Wait a few minutes if recently enabled")
        print(f"   4. Check project permissions")
