#!/usr/bin/env bash
# ============================================================================
# FlowCut Edge — Setup for NVIDIA GB10 (ASUS Ascent GX10)
# Run this ON the GX10 device.
#
# What this does:
#   1. Installs system deps (python3, git)
#   2. Creates a Python venv + installs PyTorch & dependencies
#   3. Pre-downloads the Phi-3.5 Vision model
#
# No Docker required — runs directly on the device.
# ============================================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()   { echo -e "${GREEN}[flowcut-edge]${NC} $*"; }
warn()  { echo -e "${YELLOW}[flowcut-edge]${NC} $*"; }
error() { echo -e "${RED}[flowcut-edge]${NC} $*" >&2; }

# ── System info ─────────────────────────────────────────────────────
log "=== FlowCut Edge — Phi-3.5 Vision Setup ==="
if [[ -f /proc/device-tree/model ]]; then
    MODEL=$(tr -d '\0' < /proc/device-tree/model)
    log "Detected: $MODEL"
else
    log "Hardware: $(cat /proc/device-tree/model 2>/dev/null || echo 'NVIDIA GPU system')"
fi

# ── Check CUDA ──────────────────────────────────────────────────────
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
    log "CUDA: $CUDA_VER"
elif [[ -d /usr/local/cuda ]]; then
    log "CUDA directory found at /usr/local/cuda"
else
    warn "CUDA not found — JetPack may need to be installed"
fi

# ── Check memory ────────────────────────────────────────────────────
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
log "Total memory: ${TOTAL_MEM}GB"
if (( TOTAL_MEM < 8 )); then
    warn "Low memory (${TOTAL_MEM}GB). Phi-3.5 Vision needs ~6GB+"
fi

# ── Install system deps (no Docker) ────────────────────────────────
log "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip \
    python3-venv \
    git \
    curl \
    2>/dev/null || true

# ── Python venv ─────────────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
    log "Creating Python virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip

# ── Install PyTorch (Jetson needs NVIDIA's wheel) ───────────────────
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log "PyTorch with CUDA already installed ✓"
else
    ARCH=$(uname -m)
    if [[ "$ARCH" == "aarch64" ]]; then
        log "aarch64 detected — installing NVIDIA PyTorch wheel..."
        pip install --quiet torch torchvision \
            --extra-index-url https://pypi.jetson-ai-lab.dev \
            2>/dev/null \
        || {
            error "Auto-install failed. Install PyTorch manually:"
            error "  https://forums.developer.nvidia.com/t/pytorch-for-jetson/"
            exit 1
        }
    else
        log "x86_64 detected — installing PyTorch from PyPI..."
        pip install --quiet torch torchvision
    fi

    if python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
        log "PyTorch installed ✓"
    else
        warn "PyTorch installed but CUDA may not be available"
    fi
fi

# ── Install Python dependencies ─────────────────────────────────────
pip install --quiet -r requirements.txt

# ── Pre-download Phi-3.5 Vision model ────────────────────────────────────────
log "Pre-downloading Phi-3.5 Vision model..."
python3 -c "
from huggingface_hub import snapshot_download
import os

cache = os.path.expanduser('~/.cache/huggingface/hub')
os.makedirs(cache, exist_ok=True)

print('Downloading microsoft/Phi-3.5-vision-instruct...')
snapshot_download('microsoft/Phi-3.5-vision-instruct', cache_dir=cache)
print('Done!')
"

# ── Done ────────────────────────────────────────────────────────────
DEVICE_IP=$(hostname -I | awk '{print $1}')

log ""
log "=========================================="
log "  Setup complete!"
log "=========================================="
log ""
log "Start the server:"
log "  bash start.sh"
log ""
log "Or run in background:"
log "  nohup bash start.sh > server.log 2>&1 &"
log ""
log "Then in FlowCut, set API URL to:"
log "  http://${DEVICE_IP}:8000/v1"
log ""
log "Health check:"
log "  curl http://${DEVICE_IP}:8000/health"
log ""
