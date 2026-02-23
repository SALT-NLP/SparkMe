FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    ffmpeg \
    portaudio19-dev \    
    libasound-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt gunicorn
RUN pip install --no-cache-dir pyaudio

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data/logs /app/data/data /app/logs && \
    chmod -R 777 /app/data /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOGS_DIR=/app/data/logs
ENV DATA_DIR=/app/data/data
ENV PORT=8080

# Expose port (Cloud Run uses PORT env var)
EXPOSE 8080

# Use gunicorn optimized for I/O-bound workload
# 2 workers × 4 threads = 8 concurrent requests per container
# Cloud Run will auto-scale containers as needed
CMD exec gunicorn --bind :$PORT \
    --workers 1 \
    --threads 8 \
    --timeout 600 \
    --worker-class gthread \
    --access-logfile - \
    --error-logfile - \
    src.main_flask:app