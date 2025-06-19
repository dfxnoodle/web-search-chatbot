from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from openai import AzureOpenAI
from googlesearch import search
import re
import logging
from urllib.parse import urljoin, urlparse
import time
from pydantic import BaseModel
from typing import List
from enum import Enum
import secrets

# Google AI imports
from google import genai
from google.genai.types import GenerateContentConfig, GoogleSearch, Tool, HttpOptions

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure session
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(16))
app.config['SESSION_TYPE'] = 'filesystem'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Google AI environment
os.environ["GOOGLE_CLOUD_PROJECT"] = "the-racer-461804-s1"
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

# Pydantic models for structured outputs
class SourceInfo(BaseModel):
    title: str
    url: str
    snippet: str

class ResponseType(str, Enum):
    INFORMATIONAL = "informational"
    FACTUAL = "factual" 
    OPINION = "opinion"
    ERROR = "error"

class KeywordExtractionResponse(BaseModel):
    keywords: str
    reasoning: str

class ChatbotResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    search_keywords: str
    response_type: ResponseType
    confidence: str  # "high", "medium", "low"

class WebSearchChatbot:
    def __init__(self):
        # Azure OpenAI setup
        self.azure_client = AzureOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION')
        )
        self.deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
        
        # Google AI setup
        try:
            self.google_client = genai.Client(http_options=HttpOptions(api_version="v1"))
            self.google_model = "gemini-2.5-flash-lite-preview-06-17"
            logger.info("Google AI client initialized successfully")
        except Exception as e:
            logger.warning(f"Google AI client initialization failed: {e}")
            self.google_client = None
            
        # Remove global conversation memory - now using session-based storage
        self.max_memory_length = 3
    
    def get_session_memory(self):
        """Get conversation memory for the current session"""
        if 'conversation_memory' not in session:
            session['conversation_memory'] = []
        return session['conversation_memory']
    
    def add_to_memory(self, user_query, assistant_response):
        """Add a dialogue to conversation memory for the current session"""
        dialogue = {
            "user": user_query,
            "assistant": assistant_response,
            "timestamp": time.time()
        }
        
        # Get current session memory
        conversation_memory = self.get_session_memory()
        conversation_memory.append(dialogue)
        
        # Keep only the last 3 dialogues
        if len(conversation_memory) > self.max_memory_length:
            conversation_memory.pop(0)
        
        # Update session
        session['conversation_memory'] = conversation_memory
        session.modified = True
        
        logger.info(f"Added dialogue to session memory. Memory size: {len(conversation_memory)}")
    
    def clear_memory(self):
        """Clear conversation memory for the current session"""
        session['conversation_memory'] = []
        session.modified = True
        logger.info("Cleared session conversation memory")
    
    def get_conversation_context(self):
        """Get formatted conversation context for AI models"""
        conversation_memory = self.get_session_memory()
        if not conversation_memory:
            return ""
        
        context = "\nPrevious conversation context:\n"
        for i, dialogue in enumerate(conversation_memory, 1):
            context += f"\nDialogue {i}:\n"
            context += f"User: {dialogue['user']}\n"
            context += f"Assistant: {dialogue['assistant'][:200]}...\n"  # Truncate long responses
        
        return context
    
    def get_conversation_history_for_google(self):
        """Get conversation history formatted for Google AI"""
        conversation_memory = self.get_session_memory()
        if not conversation_memory:
            return []
        
        history = []
        for dialogue in conversation_memory:
            history.append({"role": "user", "parts": [{"text": dialogue["user"]}]})
            history.append({"role": "model", "parts": [{"text": dialogue["assistant"]}]})
        
        return history
    
    def extract_google_sources(self, response):
        """Extract grounding sources from Google AI response"""
        sources = []
        
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata'):
                    metadata = candidate.grounding_metadata
                    if hasattr(metadata, 'grounding_chunks'):
                        for chunk in metadata.grounding_chunks:
                            if hasattr(chunk, 'web'):
                                sources.append(SourceInfo(
                                    title=getattr(chunk.web, 'title', 'Source'),
                                    url=getattr(chunk.web, 'uri', '#'),
                                    snippet=getattr(chunk, 'content', '')[:200] + "..." if hasattr(chunk, 'content') else ""
                                ))
        except Exception as e:
            logger.error(f"Error extracting Google sources: {e}")
        
        return sources
    
    def process_query_google(self, user_query):
        """Process query using Google AI with search grounding and conversation memory"""
        if not self.google_client:
            return ChatbotResponse(
                answer="Google AI is not available. Please check configuration.",
                sources=[],
                search_keywords=user_query,
                response_type=ResponseType.ERROR,
                confidence="low"
            )
        
        try:
            logger.info(f"Processing query with Google AI: {user_query}")
            
            start_time = time.time()
            
            # Prepare conversation history for context
            conversation_history = self.get_conversation_history_for_google()
            
            # Prepare contents with conversation history and current query
            contents = conversation_history.copy()  # Include previous dialogues
            
            # Add context instruction if there's conversation history
            if conversation_history:
                context_instruction = "Consider the previous conversation context when answering the following question. Provide a response that's aware of our conversation history:"
                contents.append({"role": "user", "parts": [{"text": context_instruction}]})
            
            # Add current user query
            contents.append({"role": "user", "parts": [{"text": user_query}]})
            
            # Generate response with Google Search grounding
            response = self.google_client.models.generate_content(
                model=self.google_model,
                contents=contents,
                config=GenerateContentConfig(
                    tools=[Tool(google_search=GoogleSearch())],
                    temperature=0.7,
                    max_output_tokens=500
                ),
            )
            
            response_time = time.time() - start_time
            logger.info(f"Google AI response received in {response_time:.2f}s")
            
            # Extract sources from grounding metadata
            sources = self.extract_google_sources(response)
            
            # Add to conversation memory
            self.add_to_memory(user_query, response.text)
            
            # Create structured response
            return ChatbotResponse(
                answer=response.text,
                sources=sources,
                search_keywords="Processed with Google Search grounding",
                response_type=ResponseType.INFORMATIONAL,
                confidence="high" if sources else "medium"
            )
            
        except Exception as e:
            logger.error(f"Error with Google AI: {e}")
            return ChatbotResponse(
                answer=f"Error processing with Google AI: {str(e)}",
                sources=[],
                search_keywords=user_query,
                response_type=ResponseType.ERROR,
                confidence="low"
            )
        
    def extract_search_keywords(self, user_query):
        """Use Azure OpenAI to extract optimal search keywords from user query"""
        try:
            response = self.azure_client.beta.chat.completions.parse(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a search keyword extraction expert. Extract the most relevant search keywords from the user's query and provide reasoning for your choice. Return keywords that would be effective for web search."
                    },
                    {
                        "role": "user", 
                        "content": f"Extract search keywords from: {user_query}"
                    }
                ],
                response_format=KeywordExtractionResponse,
                max_tokens=150,
                temperature=0.3
            )
            
            if response.choices[0].message.parsed:
                extraction = response.choices[0].message.parsed
                keywords = extraction.keywords
                logger.info(f"Extracted keywords: {keywords} (Reasoning: {extraction.reasoning})")
                return keywords
            else:
                logger.warning("Keyword extraction returned no parsed result")
                return user_query
                
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            # Fallback to original query
            return user_query

    def search_web(self, keywords, num_results=3):
        """Search the web and return URLs"""
        try:
            urls = []
            logger.info(f"Searching for: {keywords}")
            
            # Use the correct googlesearch-python API
            search_results = search(keywords, num_results=num_results, sleep_interval=1)
            
            for url in search_results:
                urls.append(url)
                logger.info(f"Found URL: {url}")
                # The library handles the pagination and sleep internally
            
            logger.info(f"Found {len(urls)} search results")
            return urls
        except Exception as e:
            logger.error(f"Error searching web: {e}")
            # Fallback URLs for testing if search fails
            if "joke" in keywords.lower():
                logger.info("Using fallback URLs for jokes")
                return [
                    "https://www.rd.com/jokes/",
                    "https://parade.com/1040121/marynliles/funny-jokes/",
                    "https://www.goodhousekeeping.com/life/entertainment/g30724216/best-jokes/"
                ]
            elif "test" in keywords.lower():
                logger.info("Using fallback URLs for testing")
                return [
                    "https://en.wikipedia.org/wiki/Test",
                    "https://www.merriam-webster.com/dictionary/test"
                ]
            else:
                return []

    def scrape_webpage(self, url, max_content_length=5000):
        """Scrape content from a webpage"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit content length
            if len(text) > max_content_length:
                text = text[:max_content_length] + "..."
            
            logger.info(f"Scraped {len(text)} characters from {url}")
            return text
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return ""

    def generate_response(self, user_query, scraped_content, source_urls, search_keywords):
        """Generate response using Azure OpenAI based on scraped content with conversation memory"""
        try:
            # Get conversation context
            conversation_context = self.get_conversation_context()
            
            system_prompt = """You are a helpful AI assistant that provides accurate and informative responses based on web search results. 
            
            IMPORTANT REQUIREMENTS:
            1. Use the provided web content to answer the user's question accurately
            2. ALWAYS include at least one source URL in your response
            3. Provide comprehensive but concise answers
            4. Indicate your confidence level based on the quality and relevance of sources
            5. Classify the response type appropriately
            6. Consider the conversation history when relevant to provide contextual responses
            
            If the content doesn't contain relevant information, say so clearly but still provide the source URLs for transparency."""
            
            user_prompt = f"""User Question: {user_query}
Search Keywords Used: {search_keywords}

{conversation_context}

Web Content from Sources:
{scraped_content}

Source URLs: {', '.join(source_urls)}

Please provide a structured response with your answer, source information, and metadata. Consider the conversation history when answering."""

            response = self.azure_client.beta.chat.completions.parse(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=ChatbotResponse,
                max_tokens=1500,
                temperature=0.7
            )
            
            if response.choices[0].message.parsed:
                parsed_response = response.choices[0].message.parsed
                # Add to conversation memory
                self.add_to_memory(user_query, parsed_response.answer)
                return parsed_response
            else:
                logger.warning("Structured response parsing failed, using fallback")
                # Fallback response with basic structure
                fallback_response = ChatbotResponse(
                    answer="I apologize, but I encountered an issue while processing your request. Please try again.",
                    sources=[SourceInfo(title="Error", url=url, snippet="Processing error") for url in source_urls[:1]],
                    search_keywords=search_keywords,
                    response_type=ResponseType.ERROR,
                    confidence="low"
                )
                self.add_to_memory(user_query, fallback_response.answer)
                return fallback_response
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback response with basic structure
            fallback_response = ChatbotResponse(
                answer="I apologize, but I encountered an error while processing your request. Please try again.",
                sources=[SourceInfo(title="Error", url=source_urls[0] if source_urls else "unknown", snippet="System error")],
                search_keywords=search_keywords,
                response_type=ResponseType.ERROR,
                confidence="low"
            )
            self.add_to_memory(user_query, fallback_response.answer)
            return fallback_response

    def process_query(self, user_query):
        """Main method to process user query"""
        try:
            logger.info(f"Processing query: {user_query}")
            
            # Step 1: Extract search keywords
            keywords = self.extract_search_keywords(user_query)
            logger.info(f"Using keywords for search: {keywords}")
            
            # Step 2: Search the web
            urls = self.search_web(keywords)
            logger.info(f"Search returned {len(urls)} URLs: {urls}")
            
            if not urls:
                return ChatbotResponse(
                    answer="I couldn't find any relevant web results for your query. Please try rephrasing your question.",
                    sources=[],
                    search_keywords=keywords,
                    response_type=ResponseType.ERROR,
                    confidence="low"
                )
            
            # Step 3: Scrape content from URLs
            all_content = []
            source_urls = []
            for url in urls:
                logger.info(f"Scraping content from: {url}")
                content = self.scrape_webpage(url)
                if content:
                    all_content.append(f"From {url}:\n{content}\n\n")
                    source_urls.append(url)
                    logger.info(f"Successfully scraped {len(content)} characters from {url}")
                else:
                    logger.warning(f"No content extracted from {url}")
                    # Still include the URL even if scraping failed
                    source_urls.append(url)
                time.sleep(1)  # Be respectful to websites
            
            if not all_content:
                # Create a fallback response with source URLs even if scraping failed
                fallback_sources = [SourceInfo(title="Search Result", url=url, snippet="Content could not be extracted") for url in urls[:3]]
                return ChatbotResponse(
                    answer="I found some web results but couldn't extract readable content from them. Please try a different query or check the source links directly.",
                    sources=fallback_sources,
                    search_keywords=keywords,
                    response_type=ResponseType.ERROR,
                    confidence="low"
                )
            
            # Combine all content
            combined_content = "\n".join(all_content)
            logger.info(f"Total content length: {len(combined_content)} characters")
            
            # Step 4: Generate response using Azure OpenAI with structured output
            structured_response = self.generate_response(user_query, combined_content, source_urls, keywords)
            
            return structured_response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return ChatbotResponse(
                answer="I encountered an error while processing your request. Please try again.",
                sources=[],
                search_keywords=user_query,
                response_type=ResponseType.ERROR,
                confidence="low"
            )

# Initialize chatbot
chatbot = WebSearchChatbot()

@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory('.', 'index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests with AI provider toggle and conversation memory"""
    try:
        data = request.get_json()
        user_query = data.get('message', '').strip()
        ai_provider = data.get('ai_provider', 'azure').lower()  # 'azure' or 'google'
        
        if not user_query:
            return jsonify({'error': 'Message is required'}), 400
        
        logger.info(f"Processing query with {ai_provider.upper()} AI: {user_query}")
        logger.info(f"Current memory size: {len(chatbot.get_session_memory())}")
        
        # Process query based on selected AI provider
        if ai_provider == 'google':
            # Check if Google AI is configured
            if not chatbot.google_client:
                return jsonify({
                    'error': 'Google AI is not properly configured. Using Azure OpenAI fallback.',
                    'fallback': True
                }), 500
            
            structured_response = chatbot.process_query_google(user_query)
            
        else:  # Default to Azure OpenAI
            # Check if Azure OpenAI is configured
            if not all([
                os.getenv('AZURE_OPENAI_ENDPOINT'),
                os.getenv('AZURE_OPENAI_API_KEY'),
                os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
            ]):
                return jsonify({
                    'error': 'Azure OpenAI is not properly configured. Please check your environment variables.'
                }), 500
            
            structured_response = chatbot.process_query(user_query)
        
        # Convert Pydantic model to dict for JSON serialization
        if isinstance(structured_response, ChatbotResponse):
            response_dict = {
                'response': structured_response.answer,
                'sources': [
                    {
                        'title': source.title,
                        'url': source.url,
                        'snippet': source.snippet
                    }
                    for source in structured_response.sources
                ],
                'search_keywords': structured_response.search_keywords,
                'response_type': structured_response.response_type.value,
                'confidence': structured_response.confidence,
                'ai_provider': ai_provider,
                'conversation_memory_count': len(chatbot.get_session_memory())
            }
        else:
            # Fallback for any non-structured responses
            response_dict = {
                'response': str(structured_response),
                'sources': [],
                'search_keywords': user_query,
                'response_type': 'error',
                'confidence': 'low',
                'ai_provider': ai_provider,
                'conversation_memory_count': len(chatbot.get_session_memory())
            }
        
        return jsonify(response_dict)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint for both AI providers"""
    health_status = {
        'status': 'healthy',
        'azure_openai': {
            'configured': bool(all([
                os.getenv('AZURE_OPENAI_ENDPOINT'),
                os.getenv('AZURE_OPENAI_API_KEY'),
                os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
            ])),
            'status': 'available' if all([
                os.getenv('AZURE_OPENAI_ENDPOINT'),
                os.getenv('AZURE_OPENAI_API_KEY'),
                os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
            ]) else 'not_configured'
        },
        'google_ai': {
            'configured': bool(chatbot.google_client),
            'model': chatbot.google_model if chatbot.google_client else None,
            'project': os.environ.get("GOOGLE_CLOUD_PROJECT"),
            'status': 'available' if chatbot.google_client else 'not_configured'
        }
    }
    
    # Test Google AI connectivity if available
    if chatbot.google_client:
        try:
            test_response = chatbot.google_client.models.generate_content(
                model=chatbot.google_model,
                contents="Test connectivity",
                config=GenerateContentConfig(max_output_tokens=5)
            )
            health_status['google_ai']['test_successful'] = bool(test_response.text)
        except Exception as e:
            health_status['google_ai']['status'] = 'error'
            health_status['google_ai']['error'] = str(e)
    
    return jsonify(health_status)

@app.route('/api/memory/clear', methods=['POST'])
def clear_memory():
    """Clear conversation memory"""
    try:
        chatbot.clear_memory()
        return jsonify({
            'status': 'success',
            'message': 'Conversation memory cleared',
            'memory_count': 0
        })
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        return jsonify({'error': 'Failed to clear memory'}), 500

@app.route('/api/memory/status', methods=['GET'])
def memory_status():
    """Get current memory status"""
    try:
        conversation_memory = chatbot.get_session_memory()
        return jsonify({
            'memory_count': len(conversation_memory),
            'max_memory_length': chatbot.max_memory_length,
            'memory_summary': [
                {
                    'dialogue_number': i + 1,
                    'user_query_preview': dialogue['user'][:50] + '...' if len(dialogue['user']) > 50 else dialogue['user'],
                    'timestamp': dialogue['timestamp']
                }
                for i, dialogue in enumerate(conversation_memory)
            ]
        })
    except Exception as e:
        logger.error(f"Error getting memory status: {e}")
        return jsonify({'error': 'Failed to get memory status'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true')
