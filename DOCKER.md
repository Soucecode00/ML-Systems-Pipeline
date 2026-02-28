# Loan Prediction ML System - Docker Deployment Guide

## Quick Start

### 1. Build and run with Docker Compose (Recommended)

```bash
docker-compose up --build
```

This will start both services:
- API: http://localhost:8000
- Streamlit UI: http://localhost:8501
- API Documentation: http://localhost:8000/docs

### 2. Run API only

```bash
docker build -t loan-prediction-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models loan-prediction-api
```

### 3. Run Streamlit only

```bash
docker build -f Dockerfile.streamlit -t loan-prediction-streamlit .
docker run -p 8501:8501 -v $(pwd)/models:/app/models loan-prediction-streamlit
```

## Prerequisites

Before running the Docker containers, make sure you have trained the model:

```bash
python -m src.models.train_model
```

This will create the necessary model files in the `models/` directory.

## Services

### API Service
- **Port**: 8000
- **Health Check**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs

### Streamlit Service
- **Port**: 8501
- **URL**: http://localhost:8501

## Environment Variables

You can customize the deployment by setting environment variables:

```bash
# Example
docker run -p 8000:8000 \
  -e LOG_LEVEL=debug \
  -v $(pwd)/models:/app/models \
  loan-prediction-api
```

## Volume Mounts

The following directories are mounted as volumes:
- `./models` - Model artifacts
- `./logs` - Application logs
- `./data` - Data files

## Stopping Services

```bash
# Stop Docker Compose services
docker-compose down

# Stop individual containers
docker stop loan-prediction-api
docker stop loan-prediction-streamlit
```

## Rebuilding

```bash
# Rebuild all services
docker-compose up --build

# Rebuild specific service
docker-compose up --build api
```

## Troubleshooting

### Model not found error
Make sure you've trained the model before running the containers:
```bash
python -m src.models.train_model
```

### Port already in use
Change the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8080:8000"  # Use port 8080 instead of 8000
```

### Container fails health check
Check the logs:
```bash
docker logs loan-prediction-api
```
