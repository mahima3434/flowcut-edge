"""
/v1/video/generate — Video generation endpoints using NVIDIA Cosmos.
Supports text-to-video, image-to-video, and morph transitions.
"""

import logging
import os
from typing import Optional, List

from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger("flowcut-edge")
router = APIRouter()


# ── Request Models ───────────────────────────────────────────────────

class TextToVideoRequest(BaseModel):
    prompt: str
    duration_seconds: float = 4.0
    width: int = 1024
    height: int = 576
    num_inference_steps: int = 50


class ImageToVideoRequest(BaseModel):
    image_url: str
    prompt: str = ""
    duration_seconds: float = 4.0
    width: int = 1024
    height: int = 576
    num_inference_steps: int = 50


class MorphVideoRequest(BaseModel):
    start_image_url: str
    end_image_url: str
    prompt: str = ""
    duration_seconds: float = 4.0
    width: int = 1024
    height: int = 576
    num_inference_steps: int = 50


class VideoResponse(BaseModel):
    success: bool
    video_url: Optional[str] = None
    video_path: Optional[str] = None
    error: Optional[str] = None
    model: str = ""


# ── Endpoints ────────────────────────────────────────────────────────

@router.post("/video/generate/text", response_model=VideoResponse)
async def text_to_video(req: TextToVideoRequest, request: Request):
    """Generate video from text prompt using NVIDIA Cosmos."""
    cosmos = request.app.state.cosmos_manager

    if not cosmos.is_loaded:
        raise HTTPException(503, "Video model is still loading")

    path, error = await cosmos.generate_text_to_video(
        prompt=req.prompt,
        duration_seconds=req.duration_seconds,
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_inference_steps,
    )

    if error:
        return VideoResponse(success=False, error=error, model=cosmos.model_id)

    # Build download URL
    filename = os.path.basename(path)
    video_url = f"/v1/video/download/{filename}"

    return VideoResponse(
        success=True,
        video_url=video_url,
        video_path=path,
        model=cosmos.model_id,
    )


@router.post("/video/generate/image", response_model=VideoResponse)
async def image_to_video(req: ImageToVideoRequest, request: Request):
    """Generate video from an input image using NVIDIA Cosmos."""
    cosmos = request.app.state.cosmos_manager

    if not cosmos.is_loaded:
        raise HTTPException(503, "Video model is still loading")

    path, error = await cosmos.generate_image_to_video(
        image_path_or_url=req.image_url,
        prompt=req.prompt,
        duration_seconds=req.duration_seconds,
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_inference_steps,
    )

    if error:
        return VideoResponse(success=False, error=error, model=cosmos.model_id)

    filename = os.path.basename(path)
    video_url = f"/v1/video/download/{filename}"

    return VideoResponse(
        success=True,
        video_url=video_url,
        video_path=path,
        model=cosmos.model_id,
    )


@router.post("/video/generate/morph", response_model=VideoResponse)
async def morph_video(req: MorphVideoRequest, request: Request):
    """Generate morph transition between two frames using NVIDIA Cosmos."""
    cosmos = request.app.state.cosmos_manager

    if not cosmos.is_loaded:
        raise HTTPException(503, "Video model is still loading")

    path, error = await cosmos.generate_morph_video(
        start_image=req.start_image_url,
        end_image=req.end_image_url,
        prompt=req.prompt,
        duration_seconds=req.duration_seconds,
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_inference_steps,
    )

    if error:
        return VideoResponse(success=False, error=error, model=cosmos.model_id)

    filename = os.path.basename(path)
    video_url = f"/v1/video/download/{filename}"

    return VideoResponse(
        success=True,
        video_url=video_url,
        video_path=path,
        model=cosmos.model_id,
    )


@router.get("/video/download/{filename}")
async def download_video(filename: str):
    """Download a generated video file."""
    from api.cosmos_manager import OUTPUT_DIR

    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "Video not found")

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=filename,
    )
