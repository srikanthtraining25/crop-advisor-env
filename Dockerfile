# CropAdvisor RL Environment — Standalone Dockerfile
# For local testing and Hugging Face Spaces deployment

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy the environment code, config, and inference script
COPY pyproject.toml /app/pyproject.toml
COPY crop_advisor_env/ /app/crop_advisor_env/
COPY openenv.yaml /app/openenv.yaml
COPY inference.py /app/inference.py
COPY server/ /app/server/

# Install the crop_advisor_env package so imports resolve
RUN pip install --no-cache-dir .

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "crop_advisor_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
