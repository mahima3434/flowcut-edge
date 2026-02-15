"""
NVIDIA Cosmos video generation manager — uses official Cosmos repo + scripts.
Runs inference via the Cosmos Python scripts (text2world, video2world)
either natively or inside a jetson-container.

Setup on the ASUS Ascent GX10:
  git clone --recursive https://github.com/NVIDIA/Cosmos.git ~/Cosmos
  cd ~/Cosmos
  huggingface-cli login
  PYTHONPATH=$(pwd) python3 cosmos1/scripts/download_diffusion.py --model_sizes 7B --model_types Text2World Video2World
"""

import logging
import asyncio
import os
import uuid
import subprocess
import shutil
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger("flowcut-edge")

# Paths
COSMOS_DIR = os.getenv("COSMOS_DIR", os.path.expanduser("~/Cosmos"))
CHECKPOINT_DIR = os.path.join(COSMOS_DIR, "checkpoints")
OUTPUT_DIR = os.getenv("VIDEO_OUTPUT_DIR", "/tmp/flowcut-edge/videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model config
DEFAULT_MODEL = os.getenv("COSMOS_MODEL", "Cosmos-1.0-Diffusion-7B-Text2World")

# Common offload flags for memory-efficient inference on Jetson/edge
OFFLOAD_FLAGS = [
    "--offload_tokenizer",
    "--offload_diffusion_transformer",
    "--offload_text_encoder_model",
    "--offload_prompt_upsampler",
    "--offload_guardrail_models",
]


class CosmosVideoManager:
    """Manages NVIDIA Cosmos for on-device video generation via official scripts."""

    def __init__(self):
        self.cosmos_dir = COSMOS_DIR
        self.checkpoint_dir = CHECKPOINT_DIR
        self.model_name = DEFAULT_MODEL
        self._loaded = False
        self.model_id = DEFAULT_MODEL

    async def load_model(self, model_id: str = None, device: str = "cuda"):
        """Verify Cosmos is installed and models are downloaded."""
        if model_id:
            self.model_name = model_id
            self.model_id = model_id
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._verify_setup)

    def _verify_setup(self):
        """Check that Cosmos repo and checkpoints exist."""
        if not os.path.isdir(self.cosmos_dir):
            raise FileNotFoundError(
                f"Cosmos repo not found at {self.cosmos_dir}. "
                "Run: git clone --recursive https://github.com/NVIDIA/Cosmos.git ~/Cosmos"
            )

        t2w_script = os.path.join(
            self.cosmos_dir, "cosmos1", "models", "diffusion", "inference", "text2world.py"
        )
        if not os.path.isfile(t2w_script):
            raise FileNotFoundError(
                f"Cosmos text2world script not found at {t2w_script}. "
                "Make sure you cloned with --recursive."
            )

        model_dir = os.path.join(self.checkpoint_dir, self.model_name)
        if not os.path.isdir(model_dir):
            logger.warning(
                "Model checkpoint not found at %s. Download with:\n"
                "  cd %s && PYTHONPATH=$(pwd) python3 cosmos1/scripts/download_diffusion.py "
                "--model_sizes 7B --model_types Text2World Video2World",
                model_dir, self.cosmos_dir,
            )
        else:
            logger.info("Cosmos checkpoint found: %s", model_dir)

        self._loaded = True
        logger.info("Cosmos video manager ready (model=%s)", self.model_name)

    async def generate_text_to_video(
        self,
        prompt: str,
        duration_seconds: float = 4.0,
        width: int = 1024,
        height: int = 576,
        num_inference_steps: int = 50,
    ) -> Tuple[str, Optional[str]]:
        """Generate video from text prompt via Cosmos text2world."""
        if not self._loaded:
            return "", "Cosmos not loaded. Run setup first."
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_text2world, prompt)

    def _run_text2world(self, prompt: str) -> Tuple[str, Optional[str]]:
        """Run Cosmos text2world inference script."""
        video_name = f"flowcut_{uuid.uuid4().hex[:8]}"
        output_subdir = os.path.join(self.cosmos_dir, "outputs")
        os.makedirs(output_subdir, exist_ok=True)

        script_path = os.path.join(
            self.cosmos_dir, "cosmos1", "models", "diffusion", "inference", "text2world.py"
        )

        cmd = [
            "python3", script_path,
            "--checkpoint_dir", self.checkpoint_dir,
            "--diffusion_transformer_dir", self.model_name,
            "--prompt", prompt,
            "--video_save_name", video_name,
            *OFFLOAD_FLAGS,
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = self.cosmos_dir

        logger.info("Cosmos text2world: name=%s prompt=%s", video_name, prompt[:100])

        try:
            result = subprocess.run(
                cmd, cwd=self.cosmos_dir, env=env,
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                logger.error("Cosmos text2world failed:\nstderr: %s", result.stderr[-500:])
                return "", f"Cosmos inference failed: {result.stderr[-200:]}"

            output_path = self._find_output_video(video_name, output_subdir)
            if output_path:
                final_path = os.path.join(OUTPUT_DIR, f"{video_name}.mp4")
                shutil.copy2(output_path, final_path)
                logger.info("Video generated: %s", final_path)
                return final_path, None
            return "", "Cosmos completed but no output video found"

        except subprocess.TimeoutExpired:
            return "", "Cosmos inference timed out (>10 min)"
        except Exception as e:
            logger.error("Cosmos text2world error: %s", e, exc_info=True)
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
        """Generate video from an input image via Cosmos video2world."""
        if not self._loaded:
            return "", "Cosmos not loaded."
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._run_video2world_from_image, image_path_or_url, prompt,
        )

    def _run_video2world_from_image(
        self, image_input: str, prompt: str
    ) -> Tuple[str, Optional[str]]:
        """Run Cosmos video2world with a single image input."""
        from PIL import Image
        from io import BytesIO
        import requests

        # Load input image to local file
        try:
            if image_input.startswith("http"):
                resp = requests.get(image_input, timeout=30)
                img = Image.open(BytesIO(resp.content))
            else:
                img = Image.open(image_input)
            tmp_img = os.path.join(OUTPUT_DIR, f"input_{uuid.uuid4().hex[:8]}.png")
            img.convert("RGB").save(tmp_img)
        except Exception as e:
            return "", f"Failed to load input image: {e}"

        video_name = f"flowcut_i2v_{uuid.uuid4().hex[:8]}"
        output_subdir = os.path.join(self.cosmos_dir, "outputs")
        os.makedirs(output_subdir, exist_ok=True)

        # Try Video2World model
        v2w_model = self.model_name.replace("Text2World", "Video2World")
        script_path = os.path.join(
            self.cosmos_dir, "cosmos1", "models", "diffusion", "inference", "video2world.py"
        )

        if not os.path.isfile(script_path):
            logger.warning("video2world script not found, falling back to text2world")
            enhanced = f"Starting from this scene: {prompt}" if prompt else "Animate this scene"
            return self._run_text2world(enhanced)

        cmd = [
            "python3", script_path,
            "--checkpoint_dir", self.checkpoint_dir,
            "--diffusion_transformer_dir", v2w_model,
            "--input_image_or_video_path", tmp_img,
            "--video_save_name", video_name,
            *OFFLOAD_FLAGS,
        ]
        if prompt:
            cmd.extend(["--prompt", prompt])

        env = os.environ.copy()
        env["PYTHONPATH"] = self.cosmos_dir

        logger.info("Cosmos video2world (image): name=%s", video_name)

        try:
            result = subprocess.run(
                cmd, cwd=self.cosmos_dir, env=env,
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                logger.warning("video2world failed, falling back to text2world: %s",
                               result.stderr[-200:])
                return self._run_text2world(prompt or "Animate this scene")

            output_path = self._find_output_video(video_name, output_subdir)
            if output_path:
                final_path = os.path.join(OUTPUT_DIR, f"{video_name}.mp4")
                shutil.copy2(output_path, final_path)
                return final_path, None
            return "", "No output video found"

        except subprocess.TimeoutExpired:
            return "", "Cosmos video2world timed out"
        except Exception as e:
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
        """Generate morph transition between two frames using Cosmos."""
        if not self._loaded:
            return "", "Cosmos not loaded."
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._run_morph, start_image, end_image, prompt,
        )

    def _run_morph(
        self, start_img_input: str, end_img_input: str, prompt: str,
    ) -> Tuple[str, Optional[str]]:
        """
        Generate morph: use Cosmos video2world starting from first frame,
        then blend the ending frames toward the target end image.
        """
        import numpy as np
        from PIL import Image
        from io import BytesIO
        import requests

        def load_img(src):
            if src.startswith("http"):
                return Image.open(BytesIO(requests.get(src, timeout=30).content)).convert("RGB")
            return Image.open(src).convert("RGB")

        try:
            start_img = load_img(start_img_input)
            end_img = load_img(end_img_input)
        except Exception as e:
            return "", f"Failed to load morph images: {e}"

        # Save start image
        start_path = os.path.join(OUTPUT_DIR, f"morph_start_{uuid.uuid4().hex[:8]}.png")
        start_img.save(start_path)

        morph_prompt = prompt or "Smoothly transition and transform this scene with fluid cinematic motion"

        # Generate video from start frame
        video_path, error = self._run_video2world_from_image(start_path, morph_prompt)
        if error:
            logger.warning("Image morph failed (%s), trying text2world", error)
            video_path, error = self._run_text2world(morph_prompt)

        if error or not video_path:
            return "", error or "Morph generation failed"

        # Post-process: blend last 25% of frames toward end image
        try:
            video_path = self._blend_end_frames(video_path, end_img)
        except Exception as e:
            logger.warning("End-frame blending failed (using raw): %s", e)

        return video_path, None

    def _blend_end_frames(self, video_path: str, end_img) -> str:
        """Blend the last 25% of frames toward the end image."""
        import numpy as np
        from PIL import Image
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="morph_blend_")

        # Extract frames
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            os.path.join(tmpdir, "frame_%05d.png"),
        ], capture_output=True, check=True)

        frames = sorted([
            os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".png")
        ])
        if not frames:
            return video_path

        # Match end image to frame size
        sample = Image.open(frames[0])
        end_resized = end_img.resize(sample.size)
        end_array = np.array(end_resized)

        # Blend last 25% toward end
        blend_count = max(1, len(frames) // 4)
        for i in range(blend_count):
            alpha = (i + 1) / blend_count
            idx = len(frames) - blend_count + i
            frame = np.array(Image.open(frames[idx]))
            blended = ((1 - alpha) * frame + alpha * end_array).astype(np.uint8)
            Image.fromarray(blended).save(frames[idx])

        # Get fps from original
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0",
            video_path,
        ], capture_output=True, text=True)
        fps = "8"
        if probe.returncode == 0 and "/" in probe.stdout.strip():
            num, den = probe.stdout.strip().split("/")
            fps = str(int(int(num) / max(1, int(den))))

        output_path = os.path.join(OUTPUT_DIR, f"morph_blended_{uuid.uuid4().hex[:8]}.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-framerate", fps,
            "-i", os.path.join(tmpdir, "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18",
            output_path,
        ], capture_output=True, check=True)

        shutil.rmtree(tmpdir, ignore_errors=True)
        return output_path

    def _find_output_video(self, video_name: str, output_dir: str) -> Optional[str]:
        """Find the generated video file in Cosmos outputs."""
        for ext in [".mp4", ".avi", ".mkv", ".webm"]:
            candidate = os.path.join(output_dir, f"{video_name}{ext}")
            if os.path.isfile(candidate):
                return candidate
        # Search recursively
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if video_name in f and f.endswith((".mp4", ".avi", ".mkv", ".webm")):
                    return os.path.join(root, f)
        return None

    async def unload(self):
        """Nothing to unload — Cosmos runs as subprocess."""
        self._loaded = False
        logger.info("Cosmos manager stopped")

    @property
    def is_loaded(self) -> bool:
        return self._loaded