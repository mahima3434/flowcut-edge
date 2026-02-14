#!/usr/bin/env bash
# ============================================================================
# FlowCut Edge — One-command setup for NVIDIA Jetson (ASUS Ascent GX10)
# Run this ON the Jetson device.
# ============================================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()   { echo -e "${GREEN}[flowcut-edge]${NC} $*"; }
warn()  { echo -e "${YELLOW}[flowcut-edge]${NC} $*"; }
error() { echo -e "${RED}[flowcut-edge]${NC} $*" >&2; }

# ── Check we're on a Jetson ─────────────────────────────────────────
log "Checking hardware..."
if [[ -f /proc/device-tree/model ]]; then
    MODEL=$(tr -d '\0' < /proc/device-tree/model)
    log "Detected: $MODEL"
else
    warn "Could not detect Jetson hardware (not critical, continuing)"
    MODEL="unknown"
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
if (( TOTAL_MEM < 30 )); then
    warn "Low memory (${TOTAL_MEM}GB). Phi-3.5 Vision + Nemotron-Mini 4B need ~20GB+"
fi

# ── Install system deps ────────────────────────────────────────────
log "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip \
    python3-venv \
    git \
    curl \
    docker.io \
    docker-compose \
    2>/dev/null || true

# ── Ensure user can run Docker ──────────────────────────────────────
if ! groups | grep -q docker; then
    log "Adding user to docker group..."
    sudo usermod -aG docker "$USER"
    warn "You may need to log out and back in for Docker permissions"
fi

# ── Set NVIDIA runtime as default for Docker ────────────────────────
if [[ -f /etc/nvidia-container-runtime/config.toml ]]; then
    log "NVIDIA Container Runtime detected"
else
    log "Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit 2>/dev/null || true
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

# ── Create model cache directory ────────────────────────────────────
CACHE_DIR="$HOME/.cache/flowcut-edge"
mkdir -p "$CACHE_DIR"
log "Model cache: $CACHE_DIR"

# ── Python venv (for non-Docker development) ────────────────────────
if [[ ! -d ".venv" ]]; then
    log "Creating Python virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip

# ── Install PyTorch (Jetson needs NVIDIA's wheel, not PyPI) ─────────
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log "PyTorch with CUDA already installed, skipping"
else
    ARCH=$(uname -m)
    if [[ "$ARCH" == "aarch64" ]]; then
        log "Jetson (aarch64) detected — installing NVIDIA PyTorch wheel..."
        # JetPack 6.x / L4T R36 ships CUDA 12.x
        # Official NVIDIA wheel index for Jetson:
        TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0a0+ebedce2.nv24.02-cp310-cp310-linux_aarch64.whl"
        # Try the NVIDIA pip index first (covers multiple JetPack versions)
        pip install --quiet torch torchvision \
            --extra-index-url https://pypi.jetson-ai-lab.dev \
            2>/dev/null \
        || pip install --quiet "$TORCH_URL" \
            2>/dev/null \
        || {
            error "Could not install Jetson PyTorch automatically."
            error "Please install manually from: https://forums.developer.nvidia.com/t/pytorch-for-jetson/"
            error "Then re-run this script."
            exit 1
        }
    else
        log "x86_64 detected — installing PyTorch from PyPI..."
        pip install --quiet torch
    fi
    # Verify
    if python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
        log "PyTorch installed successfully"
    else
        warn "PyTorch installed but CUDA may not be available"
    fi
fi

pip install --quiet -r requirements.txt

log ""
log "=========================================="
log "  Setup complete!"
log "=========================================="
log ""
log "To start with Docker:"
log "  docker compose up -d"
log ""
log "To start without Docker (dev mode):"
log "  source .venv/bin/activate"
log "  uvicorn api.main:app --host 0.0.0.0 --port 8000"
log ""
log "Then from FlowCut, set API URL to:"
log "  http://$(hostname -I | awk '{print $1}'):8000/v1"
log ""
