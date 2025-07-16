# Docker Registry Deployment Guide

## 🐳 Deploy from Docker Registry

This guide provides multiple ways to deploy your Stock Market Analyzer using Docker registries.

## Option 1: Deploy from Docker Hub (Recommended)

### Step 1: Set up Docker Hub
1. Create account at [Docker Hub](https://hub.docker.com)
2. Create repository: `your-username/stock-market-analyzer`

### Step 2: Configure GitHub Secrets
Add these secrets to your GitHub repository:
- `DOCKER_USERNAME`: Your Docker Hub username
- `DOCKER_PASSWORD`: Your Docker Hub access token

### Step 3: Update the workflow
Edit `.github/workflows/docker-build.yml` and change:
```yaml
IMAGE_NAME: your-username/stock-market-analyzer
```

### Step 4: Push to trigger build
```bash
git add .
git commit -m "Add Docker registry deployment"
git push origin main
```

### Step 5: Deploy on Render
1. Go to Render Dashboard
2. Create new Web Service
3. Choose "Deploy an existing image from a registry"
4. Enter image URL: `docker.io/your-username/stock-market-analyzer:latest`
5. Set environment variables:
   - `NEWS_API_KEY`: Your News API key
   - `ALPHA_VANTAGE_API_KEY`: Your Alpha Vantage key
   - `DEFAULT_LANGUAGE`: en
   - `DEFAULT_NUM_ARTICLES`: 5
   - `FLASK_ENV`: production

## Option 2: Deploy from GitHub Container Registry

### Update workflow to use GHCR:
```yaml
env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}
```

### Deploy on Render with:
```
Image URL: ghcr.io/devsid2006/stock-market-analyzer:latest
```

## Option 3: Manual Docker Build and Push

### Build locally:
```bash
# Build the image
docker build -t your-username/stock-market-analyzer:latest .

# Push to Docker Hub
docker push your-username/stock-market-analyzer:latest
```

### Deploy on Render:
Use image URL: `your-username/stock-market-analyzer:latest`

## Local Testing

### Using Docker Compose:
```bash
docker-compose up --build
```

### Using Docker directly:
```bash
# Build
docker build -t stock-analyzer .

# Run
docker run -p 5000:5000 \
  -e NEWS_API_KEY=your_key \
  -e ALPHA_VANTAGE_API_KEY=your_key \
  stock-analyzer
```

## Environment Variables Required

| Variable | Description | Example |
|----------|-------------|---------|
| `NEWS_API_KEY` | News API key | `abc123...` |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage API key | `xyz789...` |
| `DEFAULT_LANGUAGE` | Default language | `en` |
| `DEFAULT_NUM_ARTICLES` | Number of articles | `5` |
| `FLASK_ENV` | Flask environment | `production` |

## Troubleshooting

### Common Issues:
1. **Port binding**: Ensure container uses `$PORT` environment variable
2. **Environment variables**: Double-check all required env vars are set
3. **Image size**: Use slim base images to reduce build time
4. **Build failures**: Check GitHub Actions logs for details

### Health Check:
Once deployed, test these endpoints:
- `/health` - Application health status
- `/` - Main application
- `/api/analyze` - API endpoint

## Registry URLs by Platform

| Platform | Registry URL | Example |
|----------|--------------|---------|
| Docker Hub | `docker.io` | `docker.io/username/image:tag` |
| GitHub Container Registry | `ghcr.io` | `ghcr.io/username/repo:tag` |
| Google Container Registry | `gcr.io` | `gcr.io/project/image:tag` |
| AWS ECR | `account.dkr.ecr.region.amazonaws.com` | `123456789.dkr.ecr.us-east-1.amazonaws.com/image:tag` |
