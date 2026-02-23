FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install minimal dependencies (no audio libs needed)
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
RUN pip install --upgrade pip
RUN pip install --no-cache-dir Flask==3.1.2 python-dotenv==1.0.0 gunicorn

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data && chmod -R 777 /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Expose port
EXPOSE 8080

# Start gunicorn pointing to app.py
CMD exec gunicorn --bind :$PORT \
    --workers 1 \
    --threads 8 \
    --timeout 300 \
    --worker-class gthread \
    --access-logfile - \
    --error-logfile - \
    app:app