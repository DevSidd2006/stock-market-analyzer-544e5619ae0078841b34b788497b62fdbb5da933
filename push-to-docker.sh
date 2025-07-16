#!/bin/bash

# Docker Push Script for Stock Market Analyzer
# This script builds and pushes the Docker image to Docker Hub

echo "🐳 Building and Pushing Stock Market Analyzer to Docker Hub"
echo "============================================================"

# Set variables
DOCKER_USERNAME="devsid2006"
IMAGE_NAME="stock-market-analyzer"
TAG="latest"
FULL_IMAGE_NAME="$DOCKER_USERNAME/$IMAGE_NAME:$TAG"

echo "📋 Image details:"
echo "   Docker Hub Username: $DOCKER_USERNAME"
echo "   Image Name: $IMAGE_NAME"
echo "   Tag: $TAG"
echo "   Full Image: $FULL_IMAGE_NAME"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t $FULL_IMAGE_NAME .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
else
    echo "❌ Failed to build Docker image"
    exit 1
fi

# Login to Docker Hub (you'll be prompted for credentials)
echo ""
echo "🔐 Logging in to Docker Hub..."
echo "Please enter your Docker Hub credentials:"
docker login docker.io

if [ $? -eq 0 ]; then
    echo "✅ Successfully logged in to Docker Hub!"
else
    echo "❌ Failed to login to Docker Hub"
    exit 1
fi

# Push the image
echo ""
echo "📤 Pushing image to Docker Hub..."
docker push $FULL_IMAGE_NAME

if [ $? -eq 0 ]; then
    echo "✅ Successfully pushed image to Docker Hub!"
    echo ""
    echo "🎉 Your image is now available at:"
    echo "   https://hub.docker.com/r/$DOCKER_USERNAME/$IMAGE_NAME"
    echo ""
    echo "📋 To deploy on Render, use this image URL:"
    echo "   docker.io/$FULL_IMAGE_NAME"
else
    echo "❌ Failed to push image to Docker Hub"
    exit 1
fi

echo ""
echo "🚀 Next steps:"
echo "1. Go to Render Dashboard: https://dashboard.render.com"
echo "2. Create new Web Service"
echo "3. Choose 'Deploy an existing image from a registry'"
echo "4. Enter image URL: docker.io/$FULL_IMAGE_NAME"
echo "5. Set your environment variables"
echo ""
echo "✨ Deployment complete!"
