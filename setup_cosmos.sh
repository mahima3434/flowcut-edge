#!/usr/bin/env bash
# ============================================================================
# FlowCut Edge — NVIDIA Cosmos Setup for ASUS Ascent GX10
# Run this ON the GX10 after the basic deploy.sh has completed.
# ============================================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()   { echo -e "${GREEN}[cosmos-setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[cosmos-setup]${NC} $*"; }
error() { echo -e "${RED}[cosmos-setup]${NC} $*" >&2; }

COSMOS_DIR="${HOME}/Cosmos"

# ── Step 1: Install jetson-containers ───────────────────────────────
if ! command -v jetson-containers &>/dev/null; then
    log "Installing jetson-containers..."
    if [[ ! -d "${HOME}/jetson-containers" ]]; then
        git clone https://github.com/dusty-nv/jetson-containers "${HOME}/jetson-containers"
    fi
    bash "${HOME}/jetson-containers/install.sh"
else
    log "jetson-containers already installed"
fi

# ── Step 2: Clone Cosmos repo ──────────────────────────────────────
if [[ ! -d "$COSMOS_DIR" ]]; then
    log "Cloning NVIDIA Cosmos repository..."
    git clone --recursive https://github.com/NVIDIA/Cosmos.git "$COSMOS_DIR"
else
    log "Cosmos repo already exists at $COSMOS_DIR"
fi

# ── Step 3: HuggingFace login ─────────────────────────────────────
log "Checking HuggingFace authentication..."
if ! python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    warn "Please log in to HuggingFace to download Cosmos models:"
    warn "  1. Get a token from https://huggingface.co/settings/tokens"
    warn "  2. Accept the license at https://huggingface.co/nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
    echo ""
    cd "$COSMOS_DIR"
    source "${HOME}/flowcut-edge/.venv/bin/activate" 2>/dev/null || true
    python3 -m huggingface_hub.commands.huggingface_cli login
else
    log "HuggingFace authenticated"
fi

# ── Step 4: Download Cosmos models ─────────────────────────────────
CHECKPOINT_DIR="${COSMOS_DIR}/checkpoints"
if [[ ! -d "${CHECKPOINT_DIR}/Cosmos-1.0-Diffusion-7B-Text2World" ]]; then
    log "Downloading Cosmos 7B models (Text2World + Video2World)..."
    log "This may take 20-30 minutes depending on connection speed..."
    cd "$COSMOS_DIR"
    PYTHONPATH="$COSMOS_DIR" python3 cosmos1/scripts/download_diffusion.py \
        --model_sizes 7B \
        --model_types Text2World Video2World
    log "Models downloaded!"
else
    log "Cosmos checkpoints already exist"
fi

# ── Step 5: Quick test ─────────────────────────────────────────────
log ""
log "=========================================="
log "  Cosmos setup complete!"
log "=========================================="
log ""
log "Models at: ${CHECKPOINT_DIR}"
ls -la "${CHECKPOINT_DIR}/" 2>/dev/null || true
log ""
log "To test text-to-video generation:"
log "  cd ${COSMOS_DIR}"
log "  PYTHONPATH=\$(pwd) python3 cosmos1/models/diffusion/inference/text2world.py \\"
log "    --checkpoint_dir checkpoints \\"
log "    --diffusion_transformer_dir Cosmos-1.0-Diffusion-7B-Text2World \\"
log "    --prompt 'A beautiful sunset over the ocean' \\"
log "    --video_save_name test_output \\"
log "    --offload_tokenizer --offload_diffusion_transformer \\"
log "    --offload_text_encoder_model --offload_prompt_upsampler \\"
log "    --offload_guardrail_models"
log ""
log "Then restart FlowCut Edge server:"
log "  cd ~/flowcut-edge && bash start.sh"
log ""
