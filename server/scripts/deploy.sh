#!/bin/bash

set -e

echo "Pulling the latest code..."
git pull origin main

echo "Building and starting Docker containers..."
docker compose down
docker compose up -d --build

echo "Cleaning up unused Docker resources..."
docker system prune -f

echo "Deployment completed successfully!"
