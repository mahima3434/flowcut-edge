"""
FlowCut Edge — NanoVLM Vision Server
Serves NVIDIA NanoVLM (VILA-1.5-3B) on the GB10 with an OpenAI-compatible API.
Cosmos video generation is disabled (video gen handled by Runway on the client).
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.chat import router as chat_router
from api.routes.models import router as models_router
from api.routes.health import router as health_router
# from api.routes.video import router as video_router  # Cosmos disabled — using Runway

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("flowcut-edge")

# Model config (override via env vars)
VISION_MODEL = os.getenv("VISION_MODEL", "Efficient-Large-Model/VILA1.5-3b")

# Auto-detect CUDA
import torch as _torch
_default_device = "cuda" if _torch.cuda.is_available() else "cpu"
DEVICE = os.getenv("DEVICE", _default_device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load NanoVLM on startup, release on shutdown."""
    logger.info("=== FlowCut Edge starting on NVIDIA GB10 ===")
    logger.info(f"Vision model: {VISION_MODEL}")
    logger.info(f"Device:       {DEVICE}")

    try:
        from api.model_manager import ModelManager
        manager = ModelManager()
        await manager.load_models(model_id=VISION_MODEL, device=DEVICE)
    except Exception as e:
        logger.exception("CRITICAL: Failed to load NanoVLM: %s", e)
        class DummyManager:
            is_loaded = False
            load_error = f"Startup failed: {e}"
            model_id = VISION_MODEL
            def available_models(self): return []
            async def unload(self): pass
        manager = DummyManager()

    app.state.model_manager = manager

    # Cosmos video generation disabled — video gen uses Runway via client
    # from api.cosmos_manager import CosmosVideoManager
    # cosmos = CosmosVideoManager()
    # try:
    #     await cosmos.load_model(VIDEO_MODEL, DEVICE)
    # except Exception as e:
    #     logger.warning("Cosmos video model failed: %s", e)
    # app.state.cosmos_manager = cosmos

    logger.info("=== NanoVLM ready, serving requests ===")
    yield

    logger.info("=== Shutting down, releasing models ===")
    await manager.unload()


app = FastAPI(
    title="FlowCut Edge",
    description="NanoVLM Vision Server for FlowCut (GB10 Blackwell)",
    version="2.0.0",
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
# app.include_router(video_router, prefix="/v1")  # Cosmos disabled
