#!/usr/bin/env bash
# ============================================================================
# FlowCut Edge — Quick Deploy (no Docker, direct install)
# Run this ON the ASUS Ascent GX10 device.
# ============================================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[flowcut-edge]${NC} $*"; }
warn() { echo -e "${YELLOW}[flowcut-edge]${NC} $*"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── System info ─────────────────────────────────────────────────────
log "=== FlowCut Edge — Quick Deploy ==="
log "Hardware: $(cat /proc/device-tree/model 2>/dev/null || echo 'NVIDIA GPU system')"
log "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi info unavailable)"
log "RAM: $(free -h | awk '/^Mem:/{print $2}') total, $(free -h | awk '/^Mem:/{print $7}') available"
log "Disk: $(df -h / | awk 'NR==2{print $4}') free"
echo ""

# ── Install system deps ────────────────────────────────────────────
log "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv git curl 2>/dev/null

# ── Python venv ─────────────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
    log "Creating Python virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip

# ── Install Python deps ────────────────────────────────────────────
log "Installing Python dependencies..."
pip install --quiet -r requirements.txt

# ── Pre-download models ────────────────────────────────────────────
log "Pre-downloading models (this takes a few minutes on first run)..."
python3 -c "
from huggingface_hub import snapshot_download
import os

cache = os.path.expanduser('~/.cache/huggingface/hub')
os.makedirs(cache, exist_ok=True)

print('Downloading Nemotron-Mini 4B...')
snapshot_download('nvidia/Nemotron-Mini-4B-Instruct', cache_dir=cache)

print('Downloading VILA 1.5 8B...')
snapshot_download('Efficient-Large-Model/VILA1.5-8b', cache_dir=cache)

print('Done!')
"

# ── Get device IP ───────────────────────────────────────────────────
DEVICE_IP=$(hostname -I | awk '{print $1}')

log ""
log "=========================================="
log "  Deploy complete!"
log "=========================================="
log ""
log "Start the server:"
log "  cd $SCRIPT_DIR"
log "  source .venv/bin/activate"
log "  uvicorn api.main:app --host 0.0.0.0 --port 8000"
log ""
log "Or run in background:"
log "  nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &"
log ""
log "Then in FlowCut, set API URL to:"
log "  http://${DEVICE_IP}:8000/v1"
log ""
log "Health check:"
log "  curl http://${DEVICE_IP}:8000/health"
log ""
