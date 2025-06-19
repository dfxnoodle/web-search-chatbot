# FastAPI Migration Guide

## Overview

Your web search chatbot has been successfully migrated from Flask to FastAPI, providing significant performance improvements for high-throughput scenarios. This guide covers the migration details, new features, and how to run the updated application.

## Key Improvements

### 🚀 Performance Benefits
- **Async/Await Support**: Native asynchronous request handling
- **High Throughput**: Better performance under load compared to Flask
- **Type Safety**: Full Pydantic integration for request/response validation
- **Auto Documentation**: Interactive API docs at `/docs` and `/redoc`

### 📦 Package Management with UV
- **Faster Installation**: `uv` is significantly faster than `pip`
- **Better Dependency Resolution**: More reliable dependency management
- **Modern Python Tooling**: Uses the latest Python packaging standards

## File Changes

### New Files
- `fastapi_app.py` - Main FastAPI application (replaces `app.py`)
- `run_fastapi.sh` - Startup script using `uv`
- `install_uv.sh` - Script to install `uv` package manager
- `test_fastapi.py` - Test suite for FastAPI endpoints
- `Dockerfile` - Updated Docker configuration for FastAPI
- `.dockerignore` - Docker ignore file
- `FASTAPI_MIGRATION.md` - This migration guide

### Updated Files
- `pyproject.toml` - Updated dependencies for FastAPI and `uv`

### Unchanged Files
- `index.html` - Frontend remains the same
- `static/` - CSS and JavaScript files unchanged
- `.env` - Environment variables remain the same
- All other configuration and documentation files

## API Compatibility

The FastAPI application maintains **100% API compatibility** with the original Flask application:

- Same endpoints: `/api/chat`, `/api/health`, `/api/memory/status`, `/api/memory/clear`
- Same request/response formats
- Same session management
- Same AI provider support (Azure OpenAI & Google AI)

Your existing frontend will work without any changes!

## Quick Start

### 1. Install UV Package Manager
```bash
./install_uv.sh
```

### 2. Install Dependencies
```bash
uv sync
```

### 3. Run the Application
```bash
./run_fastapi.sh
```

Or manually:
```bash
uv run uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the Application
- **Web App**: http://localhost:5000
- **API Docs**: http://localhost:5000/docs
- **Alternative Docs**: http://localhost:5000/redoc

## Testing

### Run the test suite:
```bash
uv run python test_fastapi.py
```

### Manual testing:
1. Ensure the server is running
2. Visit http://localhost:5000
3. Test chat functionality with both AI providers
4. Check API documentation at http://localhost:5000/docs

## Docker Deployment

### Build the image:
```bash
docker build -t web-search-chatbot-fastapi .
```

### Run the container:
```bash
docker run -p 5000:5000 --env-file .env web-search-chatbot-fastapi
```

## Environment Variables

No changes to environment variables are required. The same `.env` file works with both Flask and FastAPI versions:

```env
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment
AZURE_OPENAI_API_VERSION=2024-02-15-preview
FLASK_SECRET_KEY=your_secret_key  # Still used for session management
```

## Performance Comparison

### Expected improvements with FastAPI:
- **Concurrent Requests**: 2-3x better performance under load
- **Response Time**: Lower latency for API calls
- **Memory Usage**: More efficient memory utilization
- **Scalability**: Better horizontal scaling capabilities

## Development Workflow

### Using UV for development:
```bash
# Add new dependencies
uv add package_name

# Add development dependencies
uv add --dev package_name

# Update dependencies
uv sync

# Run with hot reload
uv run uvicorn fastapi_app:app --reload

# Run tests
uv run pytest

# Format code
uv run black .

# Type checking
uv run mypy .
```

## Session Management

The FastAPI version maintains the same session-based conversation memory:
- Sessions are stored in memory (for development)
- Each user gets a unique session ID
- Conversation history is preserved per session
- Memory can be cleared via API endpoint

## API Documentation

FastAPI automatically generates interactive API documentation:

### Swagger UI: `/docs`
- Interactive API testing
- Request/response schemas
- Example payloads

### ReDoc: `/redoc`
- Clean, readable documentation
- Detailed descriptions
- Code examples

## Monitoring and Health Checks

### Health endpoint: `/api/health`
Returns detailed status for:
- Application health
- Azure OpenAI configuration
- Google AI configuration
- Connectivity tests

### Memory management: `/api/memory/`
- `GET /api/memory/status` - Check conversation memory
- `POST /api/memory/clear` - Clear conversation history

## Troubleshooting

### Common issues:

1. **Port already in use**
   ```bash
   # Change port in run_fastapi.sh or set PORT environment variable
   PORT=5001 ./run_fastapi.sh
   ```

2. **UV not found**
   ```bash
   ./install_uv.sh
   source ~/.bashrc
   ```

3. **Dependencies not installed**
   ```bash
   uv sync
   ```

4. **Environment variables missing**
   ```bash
   cp .env.example .env  # Edit with your values
   ```

## Migration Benefits Summary

✅ **Better Performance**: Async handling for high throughput  
✅ **Type Safety**: Full Pydantic validation  
✅ **Auto Documentation**: Interactive API docs  
✅ **Modern Tooling**: UV package manager  
✅ **Container Ready**: Optimized Docker setup  
✅ **API Compatible**: No frontend changes needed  
✅ **Development Friendly**: Hot reload and debugging  

## Next Steps

1. **Test thoroughly** with your specific use cases
2. **Monitor performance** in your deployment environment
3. **Consider Redis** for session storage in production
4. **Set up monitoring** using FastAPI's built-in metrics
5. **Implement rate limiting** if needed for production use

## Support

For issues or questions about this migration:
1. Check the interactive API docs at `/docs`
2. Run the test suite: `uv run python test_fastapi.py`
3. Review FastAPI documentation: https://fastapi.tiangolo.com/

Happy coding with FastAPI! 🚀
