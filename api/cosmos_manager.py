"""
NVIDIA Cosmos video generation manager.
Handles text-to-video, image-to-video, and morph (first+last frame) generation
using NVIDIA Cosmos-1.0-Diffusion-7B on the edge device.
"""

import logging
import asyncio
import os
import uuid
import tempfile
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger("flowcut-edge")

# Output directory for generated videos
OUTPUT_DIR = os.getenv("VIDEO_OUTPUT_DIR", "/tmp/flowcut-edge/videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class CosmosVideoManager:
    """Manages NVIDIA Cosmos model for on-device video generation."""

    def __init__(self):
        self.pipeline = None
        self.device = "cuda"
        self.model_id = ""
        self._loaded = False

    async def load_model(self, model_id: str, device: str = "cuda"):
        """Load Cosmos diffusion pipeline."""
        self.model_id = model_id
        self.device = device
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self):
        """Synchronous model loading."""
        import torch

        try:
            # Try Cosmos-specific pipeline first
            from diffusers import CosmosPipeline
            self.pipeline = CosmosPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
            logger.info("Cosmos pipeline loaded (native)")
            self._loaded = True
            return
        except (ImportError, Exception) as e:
            logger.info("CosmosPipeline not available: %s, trying DiffusionPipeline", e)

        try:
            # Fallback: generic DiffusionPipeline (diffusers auto-detects)
            from diffusers import DiffusionPipeline
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(self.device)
            logger.info("Cosmos pipeline loaded (DiffusionPipeline)")
            self._loaded = True
            return
        except Exception as e:
            logger.warning("DiffusionPipeline failed: %s, trying SVD fallback", e)

        try:
            # Last resort: Stable Video Diffusion (well-supported in diffusers)
            from diffusers import StableVideoDiffusionPipeline
            self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(self.device)
            self.model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
            logger.info("Fallback: Stable Video Diffusion loaded")
            self._loaded = True
        except Exception as e:
            logger.error("All video pipelines failed: %s", e)
            raise

    async def generate_text_to_video(
        self,
        prompt: str,
        duration_seconds: float = 4.0,
        width: int = 1024,
        height: int = 576,
        num_inference_steps: int = 50,
    ) -> Tuple[str, Optional[str]]:
        """Generate video from text prompt. Returns (file_path, error)."""
        if not self._loaded:
            return "", "Video model not loaded"

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_text_to_video_sync,
            prompt, duration_seconds, width, height, num_inference_steps,
        )

    def _generate_text_to_video_sync(
        self, prompt, duration_seconds, width, height, num_inference_steps
    ) -> Tuple[str, Optional[str]]:
        import torch

        try:
            # Calculate frames (~8 fps for Cosmos, ~14 fps for SVD)
            fps = 8
            num_frames = max(8, min(int(duration_seconds * fps), 80))

            output_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}.mp4")

            logger.info(
                "Generating text-to-video: prompt=%r, frames=%d, %dx%d",
                prompt[:60], num_frames, width, height,
            )

            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    num_frames=num_frames,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                )

            # Export frames to video
            self._frames_to_video(result.frames[0], output_path, fps)
            logger.info("Video saved: %s", output_path)
            return output_path, None

        except Exception as e:
            logger.error("Text-to-video failed: %s", e, exc_info=True)
            return "", str(e)

    async def generate_image_to_video(
        self,
        image_path_or_url: str,
        prompt: str = "",
        duration_seconds: float = 4.0,
        width: int = 1024,
        height: int = 576,
        num_inference_steps: int = 50,
    ) -> Tuple[str, Optional[str]]:
        """Generate video from a single input image. Returns (file_path, error)."""
        if not self._loaded:
            return "", "Video model not loaded"

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_img_to_video_sync,
            image_path_or_url, prompt, duration_seconds, width, height,
            num_inference_steps,
        )

    def _generate_img_to_video_sync(
        self, image_input, prompt, duration_seconds, width, height, steps
    ) -> Tuple[str, Optional[str]]:
        import torch
        from PIL import Image
        import requests
        from io import BytesIO

        try:
            # Load image
            if image_input.startswith("http"):
                img = Image.open(BytesIO(requests.get(image_input).content))
            else:
                img = Image.open(image_input)
            img = img.convert("RGB").resize((width, height))

            fps = 8
            num_frames = max(8, min(int(duration_seconds * fps), 80))
            output_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}.mp4")

            logger.info("Generating image-to-video: %dx%d, frames=%d", width, height, num_frames)

            with torch.no_grad():
                result = self.pipeline(
                    image=img,
                    prompt=prompt if prompt else None,
                    num_frames=num_frames,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                )

            self._frames_to_video(result.frames[0], output_path, fps)
            logger.info("Video saved: %s", output_path)
            return output_path, None

        except Exception as e:
            logger.error("Image-to-video failed: %s", e, exc_info=True)
            return "", str(e)

    async def generate_morph_video(
        self,
        start_image: str,
        end_image: str,
        prompt: str = "",
        duration_seconds: float = 4.0,
        width: int = 1024,
        height: int = 576,
        num_inference_steps: int = 50,
    ) -> Tuple[str, Optional[str]]:
        """Generate morph transition between two frames. Returns (file_path, error)."""
        if not self._loaded:
            return "", "Video model not loaded"

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_morph_sync,
            start_image, end_image, prompt, duration_seconds, width, height,
            num_inference_steps,
        )

    def _generate_morph_sync(
        self, start_img_input, end_img_input, prompt, duration_seconds,
        width, height, steps
    ) -> Tuple[str, Optional[str]]:
        import torch
        import numpy as np
        from PIL import Image
        import requests
        from io import BytesIO

        try:
            # Load start and end images
            def load_img(src):
                if src.startswith("http"):
                    return Image.open(BytesIO(requests.get(src).content)).convert("RGB")
                return Image.open(src).convert("RGB")

            start_img = load_img(start_img_input).resize((width, height))
            end_img = load_img(end_img_input).resize((width, height))

            fps = 8
            num_frames = max(8, min(int(duration_seconds * fps), 80))
            output_path = os.path.join(OUTPUT_DIR, f"morph_{uuid.uuid4().hex}.mp4")

            morph_prompt = prompt or "Smooth morphing transition between two scenes"

            logger.info(
                "Generating morph: %dx%d, frames=%d, prompt=%r",
                width, height, num_frames, morph_prompt[:60],
            )

            # Strategy 1: Use pipeline with start image + prompt describing end
            # The model generates a natural transition from the start frame
            with torch.no_grad():
                result = self.pipeline(
                    image=start_img,
                    prompt=morph_prompt,
                    num_frames=num_frames,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                )

            frames = list(result.frames[0])

            # Blend the last few frames toward the end image for smooth landing
            blend_count = min(len(frames) // 4, 8)
            end_array = np.array(end_img)
            for i in range(blend_count):
                alpha = (i + 1) / blend_count
                idx = len(frames) - blend_count + i
                frame_array = np.array(frames[idx])
                blended = ((1 - alpha) * frame_array + alpha * end_array).astype(np.uint8)
                frames[idx] = Image.fromarray(blended)

            self._frames_to_video(frames, output_path, fps)
            logger.info("Morph video saved: %s", output_path)
            return output_path, None

        except Exception as e:
            logger.error("Morph generation failed: %s", e, exc_info=True)
            return "", str(e)

    def _frames_to_video(self, frames, output_path: str, fps: int = 8):
        """Convert PIL Image frames to an MP4 video using ffmpeg."""
        import subprocess

        # Write frames to temp dir, then ffmpeg combine
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, frame in enumerate(frames):
                if hasattr(frame, 'save'):
                    frame.save(os.path.join(tmpdir, f"frame_{i:05d}.png"))
                else:
                    from PIL import Image
                    Image.fromarray(frame).save(os.path.join(tmpdir, f"frame_{i:05d}.png"))

            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(tmpdir, "frame_%05d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-crf", "18",
                output_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)

    async def unload(self):
        """Release GPU memory."""
        import torch
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
        logger.info("Cosmos model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded
