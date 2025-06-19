#!/usr/bin/env python3
"""
Simple test script to verify the memory functionality of the chatbot.
"""
import requests
import json
import time

def test_memory_functionality():
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Memory Functionality")
    print("=" * 50)
    
    # Test 1: Check initial memory status
    print("\n1. Checking initial memory status...")
    try:
        response = requests.get(f"{base_url}/api/memory/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Initial memory count: {data['memory_count']}/3")
        else:
            print(f"❌ Failed to get memory status: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        print("Make sure the server is running with: python app.py")
        return
    
    # Test 2: Send a few test messages
    test_messages = [
        "What is the capital of France?",
        "Tell me about that city's famous landmarks",
        "What's the weather like there usually?"
    ]
    
    print(f"\n2. Sending {len(test_messages)} test messages...")
    for i, message in enumerate(test_messages, 1):
        print(f"   Message {i}: {message[:30]}...")
        try:
            response = requests.post(f"{base_url}/api/chat", 
                                   json={"message": message, "ai_provider": "google"})
            if response.status_code == 200:
                data = response.json()
                memory_count = data.get('conversation_memory_count', 'unknown')
                print(f"   ✅ Response received. Memory: {memory_count}/3")
            else:
                print(f"   ❌ Failed to send message: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error sending message: {e}")
        
        time.sleep(1)  # Be nice to the APIs
    
    # Test 3: Check memory status after messages
    print(f"\n3. Checking memory status after messages...")
    try:
        response = requests.get(f"{base_url}/api/memory/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Final memory count: {data['memory_count']}/3")
            if data['memory_summary']:
                print("📝 Memory summary:")
                for summary in data['memory_summary']:
                    print(f"   - Dialogue {summary['dialogue_number']}: {summary['user_query_preview']}")
        else:
            print(f"❌ Failed to get memory status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting memory status: {e}")
    
    # Test 4: Clear memory
    print(f"\n4. Testing memory clear functionality...")
    try:
        response = requests.post(f"{base_url}/api/memory/clear")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Memory cleared: {data['message']}")
        else:
            print(f"❌ Failed to clear memory: {response.status_code}")
    except Exception as e:
        print(f"❌ Error clearing memory: {e}")
    
    # Test 5: Verify memory is cleared
    print(f"\n5. Verifying memory is cleared...")
    try:
        response = requests.get(f"{base_url}/api/memory/status")
        if response.status_code == 200:
            data = response.json()
            final_count = data['memory_count']
            if final_count == 0:
                print(f"✅ Memory successfully cleared: {final_count}/3")
            else:
                print(f"❌ Memory not properly cleared: {final_count}/3")
        else:
            print(f"❌ Failed to get memory status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting memory status: {e}")
    
    print("\n🎉 Memory functionality test complete!")

if __name__ == "__main__":
    test_memory_functionality()
