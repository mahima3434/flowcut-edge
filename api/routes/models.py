"""
/v1/models endpoint — list available models.
"""

import logging
from fastapi import APIRouter, Request

logger = logging.getLogger("flowcut-edge")
router = APIRouter()


@router.get("/models")
async def list_models(request: Request):
    """List available models (OpenAI-compatible)."""
    manager = request.app.state.model_manager
    models = manager.available_models() if manager.is_loaded else []
    return {
        "object": "list",
        "data": models,
    }
