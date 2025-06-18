# Web Search Chatbot

A smart chatbot that searches the web, scrapes content, and provides AI-powered responses using Azure OpenAI.

## Features

- 🔍 **Intelligent Web Search**: AI extracts optimal search keywords from user queries
- 🌐 **Web Scraping**: Automatically scrapes and processes content from search results
- 🤖 **AI-Powered Responses**: Uses Azure OpenAI to generate comprehensive answers
- 💻 **Clean Interface**: Simple HTML/CSS/JS frontend without frameworks
- ⚡ **Fast Backend**: Python Flask backend with efficient processing

## Architecture

- **Frontend**: Vanilla HTML, CSS, and JavaScript
- **Backend**: Python Flask with Azure OpenAI integration
- **Web Search**: Google search API integration
- **Web Scraping**: BeautifulSoup for content extraction
- **AI Processing**: Azure OpenAI for intelligent responses

## Setup Instructions

### 1. Clone and Navigate
```bash
cd /home/dinochlai/web-search-chatbot
```

### 2. Set Up Python Environment
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies using uv
uv sync
```

### 3. Configure Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file with your Azure OpenAI credentials
nano .env
```

Required environment variables:
```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
FLASK_DEBUG=True
```

### 4. Run the Application
```bash
uv run python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. Open your web browser and go to `http://localhost:5000`
2. Type your question in the chat input
3. The bot will:
   - Extract search keywords from your query
   - Search the web for relevant content
   - Scrape and analyze the content
   - Generate an AI-powered response

## API Endpoints

- `GET /` - Serves the frontend
- `POST /api/chat` - Main chat endpoint
- `GET /api/health` - Health check endpoint

### Chat API Request Format
```json
{
  "message": "Your question here"
}
```

### Chat API Response Format
```json
{
  "response": "AI-generated response based on web search results"
}
```

## Project Structure
```
web-search-chatbot/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .env                  # Your environment variables (not in git)
├── .gitignore           # Git ignore file
├── index.html           # Frontend HTML
├── static/
│   ├── style.css        # Frontend CSS
│   └── script.js        # Frontend JavaScript
└── README.md            # This file
```

## Dependencies

### Backend (Python)
- Flask - Web framework
- Flask-CORS - Cross-origin resource sharing
- Requests - HTTP library
- BeautifulSoup4 - Web scraping
- OpenAI - Azure OpenAI client
- Python-dotenv - Environment variable management
- googlesearch-python - Google search API

### Frontend
- Vanilla JavaScript (ES6+)
- CSS3 with Flexbox and Grid
- HTML5

## Configuration

### Azure OpenAI Setup
1. Create an Azure OpenAI resource in Azure Portal
2. Deploy a model (GPT-3.5-turbo or GPT-4)
3. Get your endpoint, API key, and deployment name
4. Update the `.env` file with your credentials

### Customization
- Modify search result count in `app.py` (default: 3 results)
- Adjust content length limits for scraping
- Customize AI prompts for different response styles
- Update CSS for different UI themes

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure virtual environment is activated and dependencies are installed
2. **Azure OpenAI Errors**: Verify your credentials and deployment name
3. **Search Errors**: Google search may have rate limits; add delays between requests
4. **Scraping Errors**: Some websites block scraping; the app handles this gracefully

### Debugging
- Set `FLASK_DEBUG=True` in `.env` for detailed error messages
- Check browser console for frontend errors
- Monitor Flask logs for backend issues

## Security Notes

- Keep your `.env` file secure and never commit it to version control
- Use HTTPS in production
- Consider implementing rate limiting for production use
- Validate and sanitize user inputs

## License

This project is for educational and personal use. Please respect website terms of service when scraping content.
