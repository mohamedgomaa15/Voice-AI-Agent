# Dockerfile for Voice-AI-Agent
# Builds a container that runs the Flask API server (web_ui.py)

FROM python:3.12-slim

# Install system dependencies required by some packages (e.g., ffmpeg, build tools)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ffmpeg \
       git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so we can cache pip install
COPY requirements.txt ./

# Use pip to install dependencies
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy the repository source
COPY . .

# Expose the default Flask port
EXPOSE 5000

# Default runtime configuration (can be overridden via env vars)
ENV HOST=0.0.0.0
ENV PORT=5000
ENV MODEL=base

# Run the server with configurable host/port/model
CMD ["sh", "-c", "python web_ui.py --host $HOST --port $PORT --model $MODEL"]
