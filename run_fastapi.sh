#!/bin/bash

# FastAPI Web Search Chatbot Startup Script
# This script uses uv for package management and runs the FastAPI application

echo "Starting Web Search Chatbot with FastAPI..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
fi

# Install dependencies using uv
echo "Installing dependencies with uv..."
uv sync

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Please create it with your API keys."
    echo "Required environment variables:"
    echo "  - AZURE_OPENAI_ENDPOINT"
    echo "  - AZURE_OPENAI_API_KEY"
    echo "  - AZURE_OPENAI_DEPLOYMENT_NAME"
    echo "  - AZURE_OPENAI_API_VERSION"
    echo "  - FLASK_SECRET_KEY (optional, will be auto-generated)"
fi

# Set default port if not specified
PORT=${PORT:-5000}

echo "Starting FastAPI server on port $PORT..."
echo "API documentation will be available at: http://localhost:$PORT/docs"
echo "Application will be available at: http://localhost:$PORT"

# Run the FastAPI application using uv
uv run uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT --reload
