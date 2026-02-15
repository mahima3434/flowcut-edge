"""
FlowCut Edge — NVIDIA Jetson AI Service
OpenAI-compatible API serving NVIDIA VILA-2 and Nemotron-Mini on Jetson AGX Orin.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.chat import router as chat_router
from api.routes.models import router as models_router
from api.routes.health import router as health_router
from api.routes.video import router as video_router

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("flowcut-edge")

# Model paths (set by docker-compose or setup.sh)
TEXT_MODEL = os.getenv("TEXT_MODEL", "nvidia/Nemotron-Mini-4B-Instruct")
VISION_MODEL = os.getenv("VISION_MODEL", "microsoft/Phi-3.5-vision-instruct")
VIDEO_MODEL = os.getenv("VIDEO_MODEL", "2B/post-trained")

# Auto-detect CUDA, fall back to CPU
import torch as _torch
_default_device = "cuda" if _torch.cuda.is_available() else "cpu"
DEVICE = os.getenv("DEVICE", _default_device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    logger.info("=== FlowCut Edge starting on NVIDIA GB10 ===")
    logger.info(f"Text model:   {TEXT_MODEL}")
    logger.info(f"Vision model: {VISION_MODEL}")
    logger.info(f"Video model:  {VIDEO_MODEL}")
    logger.info(f"Device:       {DEVICE}")

    # Lazy import so the app can still start if deps are missing
    try:
        from api.model_manager import ModelManager
        manager = ModelManager()
        await manager.load_models(TEXT_MODEL, VISION_MODEL, DEVICE)
    except Exception as e:
        logger.exception("CRITICAL: Failed to initialize ModelManager: %s", e)
        # Create a dummy manager or handled state could be added here, 
        # but since ModelManager handles its own errors now, this catch is for import/init errors.
        # We need to ensure app.state.model_manager exists to avoid 500s in endpoints
        class DummyManager:
            is_loaded = False
            load_error = f"Startup failed: {e}"
            def available_models(self): return []
            async def unload(self): pass
        manager = DummyManager()
        
    app.state.model_manager = manager

    # Load Cosmos video generation model
    from api.cosmos_manager import CosmosVideoManager
    cosmos = CosmosVideoManager()
    try:
        await cosmos.load_model(VIDEO_MODEL, DEVICE)
        logger.info("=== Cosmos video model loaded ===")
    except Exception as e:
        logger.warning("Cosmos video model failed to load: %s (video gen disabled)", e)
    app.state.cosmos_manager = cosmos

    logger.info("=== All models loaded, ready to serve ===")
    yield

    logger.info("=== Shutting down, releasing models ===")
    await manager.unload()
    if hasattr(app.state, 'cosmos_manager'):
        await app.state.cosmos_manager.unload()


app = FastAPI(
    title="FlowCut Edge",
    description="NVIDIA Edge AI Service for FlowCut (GB10 Blackwell)",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(models_router, prefix="/v1")
app.include_router(chat_router, prefix="/v1")
app.include_router(video_router, prefix="/v1")
