# CrowdPulse Production Multi-Stage Dockerfile (Render / Cloud / Local)
# Stage 1: Build React 18 Frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Runtime Environment
FROM python:3.11-slim

WORKDIR /app

# Install system libraries for OpenCV and headless media processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code and models
COPY src/ ./src/
COPY server/ ./server/
COPY models/ ./models/
COPY app/ ./app/
COPY yolov8n.pt ./yolov8n.pt

# Copy compiled React frontend from builder stage
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

ENV PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn server.api:app --host 0.0.0.0 --port ${PORT}"]
