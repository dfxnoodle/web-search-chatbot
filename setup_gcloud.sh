#!/bin/bash

# Google Cloud SDK Installation and Authentication Setup Script

echo "=== Google Cloud SDK Installation and Setup ==="
echo

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "This script is designed for Linux. Please adapt for your OS."
    exit 1
fi

# Download and install Google Cloud SDK
echo "1. Installing Google Cloud SDK..."
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-458.0.1-linux-x86_64.tar.gz

# Extract the archive
tar -xf google-cloud-cli-458.0.1-linux-x86_64.tar.gz

# Run the install script
./google-cloud-sdk/install.sh --quiet --path-update=true

# Initialize gcloud
echo
echo "2. Initializing gcloud..."
echo "This will open a browser for authentication."
echo "Please follow the instructions to:"
echo "- Log in with your Google account"
echo "- Select or create a Google Cloud project"
echo "- Enable the Vertex AI API"

# Source the path
source ~/.bashrc

# Initialize gcloud (this will prompt for login)
./google-cloud-sdk/bin/gcloud init

echo
echo "3. Setting up Application Default Credentials..."
./google-cloud-sdk/bin/gcloud auth application-default login

echo
echo "4. Enabling Vertex AI API..."
./google-cloud-sdk/bin/gcloud services enable aiplatform.googleapis.com

echo
echo "✅ Setup complete!"
echo
echo "Your Google Cloud SDK is now installed and configured."
echo "You can now run the Vertex AI tests."
