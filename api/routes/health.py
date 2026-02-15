"""
Health check endpoint.
"""

import torch
import logging
from fastapi import APIRouter, Request

logger = logging.getLogger("flowcut-edge")
router = APIRouter()


@router.get("/health")
async def health(request: Request):
    """Health check — shows LLaVA status and GPU info."""
    manager = request.app.state.model_manager

    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            ),
            "gpu_memory_used_gb": round(torch.cuda.memory_allocated(0) / 1e9, 1),
        }

    return {
        "status": "ok" if manager.is_loaded else ("error" if getattr(manager, "load_error", None) else "loading"),
        "error": getattr(manager, "load_error", None),
        "model": getattr(manager, "model_id", "unknown"),
        "models_loaded": manager.is_loaded,
        "available_models": [m["id"] for m in manager.available_models()],
        "gpu": gpu_info,
        "platform": "NVIDIA GB10 (ASUS Ascent GX10)",
    }
