# Build stage for deps and training (heavy stuff)
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install minimal build tools + clean up aggressively
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev \
    libhdf5-dev \
    libprotobuf-dev \
    protobuf-compiler \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# Copy only deps first for cache
COPY requirements.txt .

# Install deps in editable mode, no cache
RUN pip install --no-cache-dir --default-timeout=100 --retries=5 -e .

# Copy source for training
COPY . .

# Train the model (saves to ./artifacts/ and ./models/)
RUN python pipeline/training_pipeline.py

# Runtime stage - slim production image (PRODUCTION ONLY NEEDS INFERENCE)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Minimal runtime libs for models (HDF5 for .h5 files)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libatlas-base-dev \
    libhdf5-serial-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --from=builder /app/artifacts /app/artifacts
COPY --from=builder /app/models /app/models


# Copy runtime essentials only
COPY main.py .
COPY config/ ./config_GunO/  
COPY Dockerfile requirements.txt .  

# Expose port
EXPOSE 5000

# Single worker for 512MB limit
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "main:app"]
