"""
NVIDIA Cosmos Predict 2.5 video generation manager.
Uses the official nvidia-cosmos/cosmos-predict2.5 repo for inference via subprocess.
Supports text2world, image2world, video2world, and morph transitions.

Setup on the ASUS Ascent GX10 (NVIDIA GB10 Blackwell):
  git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git ~/cosmos-predict2.5
  cd ~/cosmos-predict2.5 && git lfs pull
  curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
  uv python install && uv sync --extra=cu130
  hf auth login   # accept license at https://huggingface.co/nvidia/Cosmos-Guardrail1
  # Checkpoints auto-download on first inference
"""

import logging
import asyncio
import json
import os
import uuid
import subprocess
import shutil
import tempfile
from typing import Optional, Tuple

logger = logging.getLogger("flowcut-edge")

# Paths
COSMOS_DIR = os.getenv("COSMOS_DIR", os.path.expanduser("~/cosmos-predict2.5"))
OUTPUT_DIR = os.getenv("VIDEO_OUTPUT_DIR", "/tmp/flowcut-edge/videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model variant: "2B/post-trained" (fast, edge-friendly), "2B/distilled" (fastest, text2world only), "14B/post-trained"
DEFAULT_MODEL = os.getenv("COSMOS_MODEL", "2B/post-trained")

# Timeout for inference (seconds) — 2B model is much faster than 14B
INFERENCE_TIMEOUT = int(os.getenv("COSMOS_TIMEOUT", "600"))


class CosmosVideoManager:
    """Manages NVIDIA Cosmos Predict 2.5 for on-device video generation."""

    def __init__(self):
        self.cosmos_dir = COSMOS_DIR
        self.model_variant = DEFAULT_MODEL
        self._loaded = False
        self.model_id = f"cosmos-predict2.5-{DEFAULT_MODEL.replace('/', '-')}"

    async def load_model(self, model_id: str = None, device: str = "cuda"):
        """Verify Cosmos Predict 2.5 is installed."""
        if model_id:
            # Allow passing variant like "2B/post-trained" or full name
            if "/" in model_id and not model_id.startswith("nvidia"):
                self.model_variant = model_id
            self.model_id = f"cosmos-predict2.5-{self.model_variant.replace('/', '-')}"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._verify_setup)

    def _verify_setup(self):
        """Check that cosmos-predict2.5 repo exists and is set up."""
        if not os.path.isdir(self.cosmos_dir):
            raise FileNotFoundError(
                f"cosmos-predict2.5 repo not found at {self.cosmos_dir}. "
                "Run: git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git "
                f"{self.cosmos_dir}"
            )

        inference_script = os.path.join(self.cosmos_dir, "examples", "inference.py")
        if not os.path.isfile(inference_script):
            raise FileNotFoundError(
                f"Inference script not found at {inference_script}. "
                "Make sure the repo is cloned correctly with git lfs pull."
            )

        # Check if venv exists
        venv_python = os.path.join(self.cosmos_dir, ".venv", "bin", "python")
        if not os.path.isfile(venv_python):
            logger.warning(
                "Cosmos venv not found. Run setup:\n"
                "  cd %s && uv python install && uv sync --extra=cu130",
                self.cosmos_dir,
            )
        else:
            logger.info("Cosmos venv found at %s", venv_python)

        self._loaded = True
        logger.info(
            "Cosmos Predict 2.5 ready (model=%s, dir=%s)",
            self.model_variant, self.cosmos_dir,
        )

    def _get_python(self) -> str:
        """Get the Python binary from the Cosmos venv."""
        venv_python = os.path.join(self.cosmos_dir, ".venv", "bin", "python")
        if os.path.isfile(venv_python):
            return venv_python
        return "python3"

    def _write_input_json(
        self,
        inference_type: str,
        name: str,
        prompt: str,
        input_path: Optional[str] = None,
    ) -> str:
        """Write a JSON input file for the inference script."""
        data = {
            "inference_type": inference_type,
            "name": name,
            "prompt": prompt,
        }
        if input_path:
            data["input_path"] = input_path

        json_path = os.path.join(OUTPUT_DIR, f"{name}_input.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        return json_path

    def _run_inference(
        self,
        inference_type: str,
        name: str,
        prompt: str,
        input_path: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """Run cosmos-predict2.5 inference and return (video_path, error)."""
        json_path = self._write_input_json(inference_type, name, prompt, input_path)
        output_dir = os.path.join(self.cosmos_dir, "outputs", "flowcut")

        script = os.path.join(self.cosmos_dir, "examples", "inference.py")
        python = self._get_python()

        cmd = [
            python, script,
            "-i", json_path,
            "-o", output_dir,
            f"--inference-type={inference_type}",
            f"--model={self.model_variant}",
        ]

        env = os.environ.copy()
        # Ensure HF_HOME is inherited for auto-download
        if "HF_HOME" not in env:
            env["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

        logger.info(
            "Cosmos %s: name=%s model=%s prompt=%s",
            inference_type, name, self.model_variant, prompt[:100],
        )
        logger.info("CMD: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd, cwd=self.cosmos_dir, env=env,
                capture_output=True, text=True, timeout=INFERENCE_TIMEOUT,
            )
            if result.returncode != 0:
                stderr_tail = result.stderr[-500:] if result.stderr else ""
                stdout_tail = result.stdout[-500:] if result.stdout else ""
                logger.error(
                    "Cosmos %s failed (rc=%d):\nstderr: %s\nstdout: %s",
                    inference_type, result.returncode, stderr_tail, stdout_tail,
                )
                return "", f"Cosmos inference failed: {stderr_tail[-200:]}"

            # Find output video
            output_path = self._find_output_video(name, output_dir)
            if output_path:
                final_path = os.path.join(OUTPUT_DIR, f"{name}.mp4")
                shutil.copy2(output_path, final_path)
                logger.info("Video generated: %s", final_path)
                return final_path, None

            return "", "Cosmos completed but no output video found"

        except subprocess.TimeoutExpired:
            return "", f"Cosmos inference timed out (>{INFERENCE_TIMEOUT}s)"
        except Exception as e:
            logger.error("Cosmos %s error: %s", inference_type, e, exc_info=True)
            return "", str(e)

    # ── Public API ──────────────────────────────────────────────────

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
        name = f"flowcut_t2w_{uuid.uuid4().hex[:8]}"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._run_inference, "text2world", name, prompt, None,
        )

    async def generate_image_to_video(
        self,
        image_path_or_url: str,
        prompt: str = "",
        duration_seconds: float = 4.0,
        width: int = 1024,
        height: int = 576,
        num_inference_steps: int = 50,
    ) -> Tuple[str, Optional[str]]:
        """Generate video from an input image via Cosmos image2world."""
        if not self._loaded:
            return "", "Cosmos not loaded."

        # Download remote image to local file
        local_image = await asyncio.get_event_loop().run_in_executor(
            None, self._prepare_image, image_path_or_url,
        )
        if not local_image:
            return "", f"Failed to load input image: {image_path_or_url}"

        name = f"flowcut_i2w_{uuid.uuid4().hex[:8]}"
        effective_prompt = prompt or "Animate this scene with natural cinematic motion"

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._run_inference, "image2world", name, effective_prompt, local_image,
        )

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

    # ── Helpers ─────────────────────────────────────────────────────

    def _prepare_image(self, image_input: str) -> Optional[str]:
        """Download/copy image to a local path Cosmos can read."""
        from PIL import Image
        from io import BytesIO
        import requests

        try:
            if image_input.startswith("http"):
                resp = requests.get(image_input, timeout=30)
                img = Image.open(BytesIO(resp.content))
            else:
                img = Image.open(image_input)
            local_path = os.path.join(OUTPUT_DIR, f"input_{uuid.uuid4().hex[:8]}.png")
            img.convert("RGB").save(local_path)
            return local_path
        except Exception as e:
            logger.error("Failed to prepare image: %s", e)
            return None

    def _run_morph(
        self, start_img_input: str, end_img_input: str, prompt: str,
    ) -> Tuple[str, Optional[str]]:
        """
        Generate morph: use Cosmos image2world starting from first frame,
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
        name = f"flowcut_morph_{uuid.uuid4().hex[:8]}"

        # Generate video from start frame via image2world
        video_path, error = self._run_inference("image2world", name, morph_prompt, start_path)
        if error:
            logger.warning("image2world morph failed (%s), trying text2world", error)
            video_path, error = self._run_inference("text2world", name, morph_prompt, None)

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
        fps = "24"
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
        # Check direct match
        for ext in [".mp4", ".avi", ".mkv", ".webm"]:
            candidate = os.path.join(output_dir, f"{video_name}{ext}")
            if os.path.isfile(candidate):
                return candidate

        # Check in inference-type subdirectories
        for subdir in ["text2world", "image2world", "video2world"]:
            for ext in [".mp4", ".avi", ".mkv", ".webm"]:
                candidate = os.path.join(output_dir, subdir, f"{video_name}{ext}")
                if os.path.isfile(candidate):
                    return candidate

        # Search recursively
        if os.path.isdir(output_dir):
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