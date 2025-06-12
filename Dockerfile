# Dockerfile for WitsV3
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt requirements.lock setup_dependencies.py ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "numpy>=1.25.0,<3.0" --force-reinstall && \
    pip install --no-cache-dir "pydantic>=2.0.0" --force-reinstall

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Expose port if needed
# EXPOSE 8000

# Run the application
CMD ["python", "run.py"]
