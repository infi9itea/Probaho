FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY backend/requirements.txt ./backend_requirements.txt
COPY rasa/actions/requirements.txt ./actions_requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir rasa==3.6.2 \
    && pip install --no-cache-dir -r backend_requirements.txt \
    && pip install --no-cache-dir -r actions_requirements.txt

# Copy all files
COPY . .

# Prepare static directory for FastAPI
RUN mkdir -p /app/static && \
    cp /app/index.html /app/static/ && \
    cp /app/app.js /app/static/ && \
    cp /app/style.css /app/static/

# Set permissions
RUN chmod +x /app/start.sh

# Expose the FastAPI port
EXPOSE 7860

# Start all services
CMD ["./start.sh"]
