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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flowcut-edge")

# Model paths (set by docker-compose or setup.sh)
TEXT_MODEL = os.getenv("TEXT_MODEL", "nvidia/Nemotron-Mini-4B-Instruct")
VISION_MODEL = os.getenv("VISION_MODEL", "microsoft/Phi-3.5-vision-instruct")

# Auto-detect CUDA, fall back to CPU
import torch as _torch
_default_device = "cuda" if _torch.cuda.is_available() else "cpu"
DEVICE = os.getenv("DEVICE", _default_device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    logger.info("=== FlowCut Edge starting on NVIDIA Jetson ===")
    logger.info(f"Text model:   {TEXT_MODEL}")
    logger.info(f"Vision model: {VISION_MODEL}")
    logger.info(f"Device:       {DEVICE}")

    # Lazy import so the app can still start if deps are missing
    from api.model_manager import ModelManager
    manager = ModelManager()
    await manager.load_models(TEXT_MODEL, VISION_MODEL, DEVICE)
    app.state.model_manager = manager

    logger.info("=== Models loaded, ready to serve ===")
    yield

    logger.info("=== Shutting down, releasing models ===")
    await manager.unload()


app = FastAPI(
    title="FlowCut Edge",
    description="NVIDIA Jetson AI Service for FlowCut",
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
