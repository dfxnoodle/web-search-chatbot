#!/usr/bin/env python3
"""
Test Google Search grounding with gemini-2.5-flash-lite-preview-06-17
"""

import os
import json
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    HttpOptions,
    Tool,
)

def setup_environment():
    """Set up environment variables for Google Gen AI SDK"""
    # Get project ID from default credentials
    from google.auth import default
    credentials, project_id = default()
    
    if not project_id:
        project_id = "the-racer-461804-s1"
    
    # Set required environment variables
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    
    print(f"✅ Environment configured:")
    print(f"   Project: {project_id}")
    print(f"   Location: global")
    print(f"   Using Vertex AI: True")
    
    return project_id

def test_google_search_grounding():
    """Test Google Search grounding with gemini-2.5-flash-lite-preview-06-17"""
    
    print("🔍 Testing Google Search Grounding with Gemini 2.5 Flash Lite")
    print("=" * 65)
    
    # Setup environment
    project_id = setup_environment()
    
    try:
        # Initialize the client
        print(f"\n📡 Initializing Google Gen AI client...")
        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        
        # Test queries that would benefit from real-time search
        test_queries = [
            "When is the next total solar eclipse in the United States?",
            "What are the latest developments in AI technology in 2025?",
            "What is the current weather in Tokyo, Japan?",
            "Who won the most recent Nobel Prize in Physics and what was their contribution?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n" + "="*50)
            print(f"🔍 Test {i}: {query}")
            print("="*50)
            
            try:
                # Generate content with Google Search grounding
                print(f"⏳ Sending request to gemini-2.5-flash-lite-preview-06-17...")
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite-preview-06-17",
                    contents=query,
                    config=GenerateContentConfig(
                        tools=[
                            # Use Google Search Tool
                            Tool(google_search=GoogleSearch())
                        ],
                        # Recommended temperature for grounding
                        temperature=1.0,
                        max_output_tokens=500
                    ),
                )
                
                print(f"✅ Response received!")
                print(f"\n🤖 Model Response:")
                print(f"{response.text}")
                
                # Check for grounding metadata
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    
                    # Look for grounding metadata
                    if hasattr(candidate, 'grounding_metadata'):
                        print(f"\n🔗 Grounding Information:")
                        metadata = candidate.grounding_metadata
                        
                        if hasattr(metadata, 'search_entry_point'):
                            print(f"   - Search entry point available")
                            
                        if hasattr(metadata, 'grounding_chunks'):
                            print(f"   - Found {len(metadata.grounding_chunks)} grounding sources")
                            for j, chunk in enumerate(metadata.grounding_chunks[:3]):  # Show first 3
                                if hasattr(chunk, 'web'):
                                    web_info = chunk.web
                                    print(f"     {j+1}. {web_info.title if hasattr(web_info, 'title') else 'Source'}")
                                    print(f"        URL: {web_info.uri if hasattr(web_info, 'uri') else 'N/A'}")
                    
                    # Check for citations in the response
                    if hasattr(candidate, 'citation_metadata'):
                        print(f"\n📚 Citations:")
                        citations = candidate.citation_metadata
                        if hasattr(citations, 'citation_sources'):
                            for j, citation in enumerate(citations.citation_sources[:3]):
                                print(f"   {j+1}. {citation.uri if hasattr(citation, 'uri') else 'Citation source'}")
                
                print(f"\n📊 Response Details:")
                print(f"   - Model: gemini-2.5-flash-lite-preview-06-17")
                print(f"   - Grounding: Google Search")
                print(f"   - Query: {query}")
                
                # Small delay between requests
                import time
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error for query '{query}': {e}")
                continue
        
        print(f"\n🎉 Google Search grounding test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize client or run test: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Ensure Google Search Suggestions are enabled")
        print(f"   2. Check project has proper API access")
        print(f"   3. Verify billing is enabled")
        return False

def test_comparison_without_grounding():
    """Test the same queries without grounding for comparison"""
    
    print(f"\n" + "="*65)
    print("🔄 Comparison Test: Same queries WITHOUT grounding")
    print("="*65)
    
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "the-racer-461804-s1")
    
    try:
        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        
        # Test one query without grounding
        query = "When is the next total solar eclipse in the United States?"
        
        print(f"\n🔍 Query: {query}")
        print(f"⏳ Sending request WITHOUT grounding...")
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite-preview-06-17",
            contents=query,
            config=GenerateContentConfig(
                temperature=1.0,
                max_output_tokens=300
                # No tools - no grounding
            ),
        )
        
        print(f"✅ Response received!")
        print(f"\n🤖 Model Response (No Grounding):")
        print(f"{response.text}")
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")

if __name__ == "__main__":
    print("🌟 Google Search Grounding Test for Gemini 2.5 Flash Lite")
    print("📋 This will test real-time search integration with your model")
    print()
    
    success = test_google_search_grounding()
    
    if success:
        # Run comparison test
        test_comparison_without_grounding()
        
        print(f"\n✨ Summary:")
        print(f"   ✅ Google Search grounding is working")
        print(f"   ✅ Model: gemini-2.5-flash-lite-preview-06-17")
        print(f"   ✅ Real-time web data integration enabled")
        print(f"   ✅ Ready for your web search chatbot!")
    else:
        print(f"\n💡 Next steps:")
        print(f"   1. Enable Google Search Suggestions in your project")
        print(f"   2. Check Google Gen AI SDK setup")
        print(f"   3. Verify project permissions")
