# Google Cloud Vertex AI Authentication Setup Guide

This guide will help you set up authentication for Google Vertex AI API access.

## Prerequisites

1. A Google Cloud Project
2. Vertex AI API enabled in your project
3. Billing enabled for your project

## Method 1: Using Google Cloud SDK (Recommended for Development)

### Step 1: Install Google Cloud SDK

Run the setup script:
```bash
./setup_gcloud.sh
```

Or install manually:
```bash
# Download Google Cloud SDK
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-458.0.1-linux-x86_64.tar.gz
tar -xf google-cloud-cli-458.0.1-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
```

### Step 2: Initialize and Authenticate
```bash
# Initialize gcloud (will prompt for login)
gcloud init

# Set up application default credentials
gcloud auth application-default login

# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com
```

### Step 3: Test Authentication
```bash
# Check if you're authenticated
gcloud auth list

# Test with our script
uv run python test_vertex_ai.py
```

## Method 2: Using Service Account (Recommended for Production)

### Step 1: Create a Service Account

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to "IAM & Admin" > "Service Accounts"
3. Click "Create Service Account"
4. Give it a name like "vertex-ai-client"
5. Grant these roles:
   - `Vertex AI User`
   - `AI Platform Developer` (if needed)

### Step 2: Create and Download Key

1. Click on your service account
2. Go to "Keys" tab
3. Click "Add Key" > "Create new key"
4. Choose JSON format
5. Download the key file

### Step 3: Set Environment Variable

```bash
# Set the path to your service account key
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"

# Or add to your .env file
echo "GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json" >> .env
```

### Step 4: Set Project ID

```bash
# Set your project ID
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Or add to your .env file  
echo "GOOGLE_CLOUD_PROJECT=your-project-id" >> .env
```

## Method 3: Using API Key (Limited functionality)

For some APIs, you can use an API key:

```bash
# Create an API key in Google Cloud Console
export GOOGLE_API_KEY="your-api-key"

# Add to .env file
echo "GOOGLE_API_KEY=your-api-key" >> .env
```

## Verify Your Setup

Run our test script to verify everything works:

```bash
# Test with authentication
uv run python test_vertex_ai.py

# Test endpoint accessibility
uv run python test_endpoint_simple.py
```

## Troubleshooting

### Common Issues:

1. **401 Unauthorized**: Authentication not set up correctly
   - Run `gcloud auth application-default login`
   - Check `GOOGLE_APPLICATION_CREDENTIALS` path

2. **403 Forbidden**: Missing permissions
   - Check IAM roles for your account/service account
   - Ensure Vertex AI API is enabled

3. **404 Not Found**: Wrong project ID or model name
   - Verify your project ID in `GOOGLE_CLOUD_PROJECT`
   - Check model name spelling

4. **Command not found: gcloud**
   - Add gcloud to PATH: `export PATH=$PATH:~/google-cloud-sdk/bin`
   - Or restart terminal after installation

### Check Your Configuration:

```bash
# Check gcloud configuration
gcloud config list

# Check application default credentials
gcloud auth application-default print-access-token

# List available models (requires auth)
gcloud ai models list --region=global
```

## Environment Variables Summary

Add these to your `.env` file:

```bash
# Required for service account authentication
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Required: Your Google Cloud Project ID
GOOGLE_CLOUD_PROJECT=your-project-id

# Optional: API key for some services
GOOGLE_API_KEY=your-api-key
```

## Next Steps

Once authentication is set up:

1. Update the project ID in `test_vertex_ai.py` to use your actual project
2. Run the test scripts to verify connectivity
3. Integrate Vertex AI into your web search chatbot application

## Useful Links

- [Google Cloud SDK Documentation](https://cloud.google.com/sdk/docs)
- [Vertex AI Authentication](https://cloud.google.com/vertex-ai/docs/authentication)
- [Service Account Setup](https://cloud.google.com/iam/docs/creating-managing-service-accounts)
