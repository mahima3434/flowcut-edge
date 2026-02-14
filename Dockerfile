# ============================================================================
# FlowCut Edge — Dockerfile for NVIDIA GB10 (Blackwell / ASUS Ascent GX10)
# ============================================================================
# Base: NVIDIA PyTorch container (ARM64 + CUDA 13.0)
# ============================================================================

FROM nvcr.io/nvidia/pytorch:24.12-py3

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Application code
COPY api/ ./api/

# Create cache directory
RUN mkdir -p /cache/huggingface

# Expose port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
