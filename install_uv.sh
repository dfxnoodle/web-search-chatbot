#!/bin/bash

# Install uv (fast Python package manager) if not already installed

echo "Checking for uv installation..."

if command -v uv &> /dev/null; then
    echo "✅ uv is already installed: $(uv --version)"
else
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    # Check if installation was successful
    if command -v uv &> /dev/null; then
        echo "✅ uv installed successfully: $(uv --version)"
        echo "💡 You may need to restart your terminal or run: source ~/.bashrc"
    else
        echo "❌ Failed to install uv"
        exit 1
    fi
fi

echo "🚀 Ready to use uv for package management!"
