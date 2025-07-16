@echo off
REM Docker Push Script for Stock Market Analyzer (Windows)
REM This script builds and pushes the Docker image to Docker Hub

echo 🐳 Building and Pushing Stock Market Analyzer to Docker Hub
echo ============================================================

REM Set variables
set DOCKER_USERNAME=devsid2006
set IMAGE_NAME=stock-market-analyzer
set TAG=latest
set FULL_IMAGE_NAME=%DOCKER_USERNAME%/%IMAGE_NAME%:%TAG%

echo 📋 Image details:
echo    Docker Hub Username: %DOCKER_USERNAME%
echo    Image Name: %IMAGE_NAME%
echo    Tag: %TAG%
echo    Full Image: %FULL_IMAGE_NAME%
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker and try again.
    pause
    exit /b 1
)

REM Build the Docker image
echo 🔨 Building Docker image...
docker build -t %FULL_IMAGE_NAME% .

if %errorlevel% equ 0 (
    echo ✅ Docker image built successfully!
) else (
    echo ❌ Failed to build Docker image
    pause
    exit /b 1
)

REM Login to Docker Hub
echo.
echo 🔐 Logging in to Docker Hub...
echo Please enter your Docker Hub credentials:
docker login docker.io

if %errorlevel% equ 0 (
    echo ✅ Successfully logged in to Docker Hub!
) else (
    echo ❌ Failed to login to Docker Hub
    pause
    exit /b 1
)

REM Push the image
echo.
echo 📤 Pushing image to Docker Hub...
docker push %FULL_IMAGE_NAME%

if %errorlevel% equ 0 (
    echo ✅ Successfully pushed image to Docker Hub!
    echo.
    echo 🎉 Your image is now available at:
    echo    https://hub.docker.com/r/%DOCKER_USERNAME%/%IMAGE_NAME%
    echo.
    echo 📋 To deploy on Render, use this image URL:
    echo    docker.io/%FULL_IMAGE_NAME%
) else (
    echo ❌ Failed to push image to Docker Hub
    pause
    exit /b 1
)

echo.
echo 🚀 Next steps:
echo 1. Go to Render Dashboard: https://dashboard.render.com
echo 2. Create new Web Service
echo 3. Choose 'Deploy an existing image from a registry'
echo 4. Enter image URL: docker.io/%FULL_IMAGE_NAME%
echo 5. Set your environment variables
echo.
echo ✨ Deployment complete!
pause
