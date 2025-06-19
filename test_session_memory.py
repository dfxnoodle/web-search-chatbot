#!/usr/bin/env python3
"""
Test script to verify that conversation memory is now isolated per session.
"""

import requests
import json

def test_session_isolation():
    """Test that different sessions have isolated conversation memory"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Session Memory Isolation")
    print("="*50)
    
    # Create two different session objects (simulating two different users)
    session1 = requests.Session()
    session2 = requests.Session()
    
    # Test Chat for Session 1
    print("\n👤 Session 1: First message")
    response1 = session1.post(f"{base_url}/api/chat", 
                             json={"message": "My name is Alice", "ai_provider": "google"})
    if response1.status_code == 200:
        print("✅ Session 1 message sent successfully")
        data = response1.json()
        print(f"   Memory count: {data.get('conversation_memory_count', 0)}")
    else:
        print(f"❌ Session 1 failed: {response1.status_code}")
    
    # Test Chat for Session 2  
    print("\n👤 Session 2: First message")
    response2 = session2.post(f"{base_url}/api/chat",
                             json={"message": "My name is Bob", "ai_provider": "google"})
    if response2.status_code == 200:
        print("✅ Session 2 message sent successfully")
        data = response2.json()
        print(f"   Memory count: {data.get('conversation_memory_count', 0)}")
    else:
        print(f"❌ Session 2 failed: {response2.status_code}")
    
    # Check memory status for both sessions
    print("\n📊 Checking memory status for each session:")
    
    # Session 1 memory
    mem1 = session1.get(f"{base_url}/api/memory/status")
    if mem1.status_code == 200:
        data1 = mem1.json()
        print(f"   Session 1 memory count: {data1.get('memory_count', 0)}")
        if data1.get('memory_summary'):
            print(f"   Session 1 last message: {data1['memory_summary'][-1]['user_query_preview']}")
    
    # Session 2 memory  
    mem2 = session2.get(f"{base_url}/api/memory/status")
    if mem2.status_code == 200:
        data2 = mem2.json()
        print(f"   Session 2 memory count: {data2.get('memory_count', 0)}")
        if data2.get('memory_summary'):
            print(f"   Session 2 last message: {data2['memory_summary'][-1]['user_query_preview']}")
    
    # Test follow-up questions to verify context isolation
    print("\n🔄 Testing follow-up questions for context isolation:")
    
    # Session 1 asks about their name
    response1_followup = session1.post(f"{base_url}/api/chat",
                                     json={"message": "What is my name?", "ai_provider": "google"})
    if response1_followup.status_code == 200:
        print("✅ Session 1 follow-up sent")
        data = response1_followup.json()
        print(f"   Session 1 response should mention Alice: {data.get('response', '')[:100]}...")
    
    # Session 2 asks about their name  
    response2_followup = session2.post(f"{base_url}/api/chat",
                                     json={"message": "What is my name?", "ai_provider": "google"})
    if response2_followup.status_code == 200:
        print("✅ Session 2 follow-up sent")
        data = response2_followup.json()
        print(f"   Session 2 response should mention Bob: {data.get('response', '')[:100]}...")
    
    print("\n🎯 Test Summary:")
    print("   - Each session should maintain its own conversation history")
    print("   - Session 1 should remember Alice, Session 2 should remember Bob")
    print("   - Memory counts should be independent between sessions")

if __name__ == "__main__":
    print("⚠️  Make sure the Flask app is running on localhost:5000 before running this test!")
    print("   Run: uv run python app.py")
    print()
    
    try:
        test_session_isolation()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask app. Make sure it's running on localhost:5000")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
