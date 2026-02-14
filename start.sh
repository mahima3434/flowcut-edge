#!/usr/bin/env bash
# ============================================================================
# Start FlowCut Edge server
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate

DEVICE_IP=$(hostname -I | awk '{print $1}')
echo "Starting FlowCut Edge on http://${DEVICE_IP}:8000"
echo "Health: http://${DEVICE_IP}:8000/health"
echo "API:    http://${DEVICE_IP}:8000/v1"
echo ""

exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
