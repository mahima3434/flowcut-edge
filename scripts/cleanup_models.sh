#!/usr/bin/env bash
# ============================================================================
# Cleanup Vision Model - Deletes microsoft/Phi-3.5-vision-instruct
# Use this to free up disk space.
# ============================================================================
set -e

MODEL_ID="models--microsoft--Phi-3.5-vision-instruct"
BOLD="\033[1m"
GREEN="\033[32m"
RED="\033[31m"
NC="\033[0m"

echo -e "${BOLD}Checking for Phi-3.5 Vision model files...${NC}"

# 1. Clean Local Cache (~/.cache/huggingface/hub)
LOCAL_CACHE="$HOME/.cache/huggingface/hub/$MODEL_ID"
if [ -d "$LOCAL_CACHE" ]; then
    echo -e "${GREEN}Found local cache at: $LOCAL_CACHE${NC}"
    read -p "Delete local files? (y/N) " confirm
    if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
        rm -rf "$LOCAL_CACHE"
        echo "Deleted local cache."
    else
        echo "Skipped local deletion."
    fi
else
    echo "No local cache found at default location."
fi

# 2. Clean Docker Volume (flowcut-edge_model-cache)
if command -v docker &> /dev/null; then
    echo -e "\n${BOLD}Checking Docker volume 'flowcut-edge_model-cache'...${NC}"
    # We use a temporary container to inspect the volume content
    # Look for the directory inside the volume mount point /cache/huggingface/hub
    if docker run --rm -v flowcut-edge_model-cache:/cache alpine ls -d "/cache/huggingface/hub/$MODEL_ID" &> /dev/null; then
        echo -e "${GREEN}Found model in Docker volume 'flowcut-edge_model-cache'${NC}"
        read -p "Delete from Docker volume? (y/N) " confirm
        if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
            docker run --rm -v flowcut-edge_model-cache:/cache alpine rm -rf "/cache/huggingface/hub/$MODEL_ID"
            echo "Deleted from Docker volume."
        else
            echo "Skipped Docker volume deletion."
        fi
    else
        echo "Model not found in Docker volume (or volume does not exist)."
    fi
else
    echo "Docker not found, skipping volume check."
fi

echo -e "\n${GREEN}Cleanup check complete.${NC}"
