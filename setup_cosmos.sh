#!/usr/bin/env bash
# ============================================================================
# FlowCut Edge — NVIDIA Cosmos Predict 2.5 Setup for ASUS Ascent GX10
# Run this ON the GX10 after deploy.sh has completed.
# ============================================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()   { echo -e "${GREEN}[cosmos-setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[cosmos-setup]${NC} $*"; }
error() { echo -e "${RED}[cosmos-setup]${NC} $*" >&2; }

COSMOS_DIR="${HOME}/cosmos-predict2.5"

# ── Step 1: Install system deps ────────────────────────────────────
log "Installing system dependencies..."
sudo apt update -qq
sudo apt install -y -qq git git-lfs curl ffmpeg libx11-dev tree wget 2>/dev/null
git lfs install

# ── Step 2: Install uv (Python package manager) ───────────────────
if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "${HOME}/.local/bin/env" 2>/dev/null || export PATH="${HOME}/.local/bin:$PATH"
else
    log "uv already installed"
fi

# ── Step 3: Clone Cosmos Predict 2.5 ──────────────────────────────
if [[ ! -d "$COSMOS_DIR" ]]; then
    log "Cloning nvidia-cosmos/cosmos-predict2.5..."
    git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git "$COSMOS_DIR"
    cd "$COSMOS_DIR"
    git lfs pull
else
    log "cosmos-predict2.5 already exists at $COSMOS_DIR"
    cd "$COSMOS_DIR"
    git pull --ff-only 2>/dev/null || true
    git lfs pull
fi

# ── Step 4: Setup Python environment (CUDA 13.0 for Blackwell) ────
cd "$COSMOS_DIR"
if [[ ! -d ".venv" ]]; then
    log "Setting up Python environment with CUDA 13.0 support..."
    uv python install
    uv sync --extra=cu130
else
    log "Python venv already exists, syncing..."
    uv sync --extra=cu130
fi
source .venv/bin/activate

# ── Step 5: HuggingFace login ─────────────────────────────────────
log "Checking HuggingFace authentication..."
if ! python -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    warn "HuggingFace login required for model downloads."
    warn "  1. Get a token from https://huggingface.co/settings/tokens"
    warn "  2. Accept the license at https://huggingface.co/nvidia/Cosmos-Guardrail1"
    echo ""
    hf auth login
else
    log "HuggingFace authenticated"
fi

# ── Step 6: Quick test (downloads checkpoints on first run) ───────
log ""
log "=========================================="
log "  Cosmos Predict 2.5 setup complete!"
log "=========================================="
log ""
log "Repo at: ${COSMOS_DIR}"
log "Python:  $(which python)"
log ""
log "To test text-to-video generation (auto-downloads ~4GB checkpoint on first run):"
log "  cd ${COSMOS_DIR}"
log "  source .venv/bin/activate"
log "  python examples/inference.py \\"
log "    -i assets/base/snowy_stop_light.json \\"
log "    -o outputs/test \\"
log "    --inference-type=text2world \\"
log "    --model=2B/post-trained"
log ""
log "Then restart FlowCut Edge server:"
log "  cd ~/flowcut-edge && bash start.sh"
log ""
