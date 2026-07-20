# Use official lightweight Python 3.10 slim image as base
FROM python:3.10-slim

# Set system environment variables to optimize Python runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    APP_MODE=api

# Set workspace directory
WORKDIR /app

# Install essential system dependencies required for OpenCV, PyTorch, and builds
RUN apt-get update && apt-get install -y git --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt to leverage Docker cache layering
COPY requirements.txt .

# Install Python package dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose Streamlit default port (8501) and FastAPI default port (8000)
EXPOSE 8501 8000

# Entrypoint CMD uses the APP_MODE environment variable to choose running mode.
# If APP_MODE is "api", it starts the high-performance FastAPI server with Uvicorn.
# If APP_MODE is "streamlit" (or any other value), it starts the interactive diagnostics Streamlit suite.
CMD if [ "$APP_MODE" = "api" ]; then \
        uvicorn tools.api:app --host 0.0.0.0 --port 8000; \
    else \
        streamlit run tools/app.py --server.port=8501 --server.address=0.0.0.0; \
    fi
