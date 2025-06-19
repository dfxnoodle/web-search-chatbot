#!/usr/bin/env python3
"""
Simple test script for the FastAPI web search chatbot
"""

import asyncio
import httpx
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_fastapi_endpoints():
    """Test the FastAPI endpoints"""
    base_url = "http://localhost:5000"
    
    async with httpx.AsyncClient() as client:
        print("🧪 Testing FastAPI Web Search Chatbot endpoints...")
        
        # Test health endpoint
        try:
            print("\n1. Testing health endpoint...")
            response = await client.get(f"{base_url}/api/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Health check passed: {health_data['status']}")
                print(f"   Azure OpenAI: {health_data['azure_openai']['status']}")
                print(f"   Google AI: {health_data['google_ai']['status']}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Health check error: {e}")
        
        # Test memory status endpoint
        try:
            print("\n2. Testing memory status endpoint...")
            response = await client.get(f"{base_url}/api/memory/status")
            if response.status_code == 200:
                memory_data = response.json()
                print(f"✅ Memory status: {memory_data['memory_count']} conversations")
            else:
                print(f"❌ Memory status failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Memory status error: {e}")
        
        # Test chat endpoint with a simple query
        try:
            print("\n3. Testing chat endpoint...")
            chat_request = {
                "message": "What is the weather like today?",
                "ai_provider": "azure"
            }
            response = await client.post(f"{base_url}/api/chat", json=chat_request)
            if response.status_code == 200:
                chat_data = response.json()
                print(f"✅ Chat response received")
                print(f"   Provider: {chat_data['ai_provider']}")
                print(f"   Response length: {len(chat_data['response'])} characters")
                print(f"   Sources found: {len(chat_data['sources'])}")
            else:
                print(f"❌ Chat request failed: {response.status_code}")
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"❌ Chat request error: {e}")

def test_environment():
    """Test if required environment variables are set"""
    print("🔧 Checking environment configuration...")
    
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_API_KEY',
        'AZURE_OPENAI_DEPLOYMENT_NAME',
        'AZURE_OPENAI_API_VERSION'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("   Please check your .env file")
        return False
    else:
        print("✅ All required environment variables are set")
        return True

if __name__ == "__main__":
    print("🚀 FastAPI Web Search Chatbot Test Suite")
    print("=" * 50)
    
    # Test environment
    env_ok = test_environment()
    
    if env_ok:
        print("\n🌐 Starting endpoint tests...")
        print("   Make sure the FastAPI server is running on localhost:5000")
        print("   Run: ./run_fastapi.sh or uv run uvicorn fastapi_app:app --reload")
        
        try:
            asyncio.run(test_fastapi_endpoints())
        except KeyboardInterrupt:
            print("\n🛑 Tests interrupted by user")
        except Exception as e:
            print(f"\n❌ Test suite error: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Test suite completed")
