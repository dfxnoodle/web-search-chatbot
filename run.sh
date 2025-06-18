#!/bin/bash

# Script to run the Web Search Chatbot
# Make sure you have configured your .env file before running

echo "🔍 Starting Web Search Chatbot..."
echo "📝 Make sure you have configured your .env file with Azure OpenAI credentials"
echo "🌐 The application will be available at http://localhost:5000"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "✏️  Please edit .env file with your Azure OpenAI credentials before running again"
    exit 1
fi

# Run the application
uv run python app.py
