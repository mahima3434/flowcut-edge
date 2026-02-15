"""
Model manager — NanoVLM (VILA 1.5-3B) vision model on GB10.
Handles loading, inference, and unloading of the vision-language model.
"""

import asyncio
import base64
import io
import logging
from typing import Optional, List, Dict, Any

import torch

logger = logging.getLogger("flowcut-edge")

# ── Default model ────────────────────────────────────────────────────
DEFAULT_MODEL_ID = "Efficient-Large-Model/VILA1.5-3b"


class ModelManager:
    """
    Manages NanoVLM (VILA-1.5-3B) for vision + text tasks.
    Provides generate_vision() and generate_text() methods used by the chat route.
    """

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded = False
        self.load_error: Optional[str] = None
        self.model_id: str = DEFAULT_MODEL_ID

    # ── Loading ──────────────────────────────────────────────────────

    async def load_models(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        vision_model_id: str = "",  # ignored — single model handles both
        device: str = "cuda",
    ):
        """Load the NanoVLM model."""
        self._device = device
        self.model_id = model_id if model_id and model_id != "disabled" else DEFAULT_MODEL_ID

        logger.info("Loading NanoVLM model: %s on %s", self.model_id, self._device)

        try:
            await asyncio.to_thread(self._load_sync)
            self._loaded = True
            self.load_error = None
            logger.info("NanoVLM model loaded ✓")
        except Exception as e:
            self._loaded = False
            self.load_error = str(e)
            logger.error("Failed to load NanoVLM: %s", e, exc_info=True)
            raise

    def _load_sync(self):
        """Synchronous model loading (run in thread)."""
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )

        # Try Vision2Seq first, then CausalLM (model architecture varies)
        try:
            from transformers import AutoModelForVision2Seq
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                device_map="auto" if self._device == "cuda" else None,
                trust_remote_code=True,
            )
            logger.info("Loaded as Vision2Seq model")
        except Exception:
            from transformers import AutoModelForCausalLM
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                device_map="auto" if self._device == "cuda" else None,
                trust_remote_code=True,
            )
            logger.info("Loaded as CausalLM model")

    # ── Inference ────────────────────────────────────────────────────

    async def generate_vision(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Generate response from text + images (vision mode)."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        text, images = self._extract_content(messages)
        content = await asyncio.to_thread(
            self._generate_sync, text, images, max_tokens, temperature
        )
        return {"content": content, "finish_reason": "stop"}

    async def generate_text(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Generate response from text-only input."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        text, _ = self._extract_content(messages)
        content = await asyncio.to_thread(
            self._generate_sync, text, [], max_tokens, temperature
        )
        return {"content": content, "finish_reason": "stop"}

    def _generate_sync(
        self,
        text: str,
        images: list,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Synchronous generation (run in thread)."""
        try:
            if images:
                if hasattr(self._processor, "apply_chat_template"):
                    msgs = [{"role": "user", "content": text}]
                    prompt = self._processor.apply_chat_template(
                        msgs, add_generation_prompt=True
                    )
                    inputs = self._processor(
                        text=prompt,
                        images=images if len(images) > 1 else images[0],
                        return_tensors="pt",
                    ).to(self._model.device)
                else:
                    inputs = self._processor(
                        text=text,
                        images=images if len(images) > 1 else images[0],
                        return_tensors="pt",
                    ).to(self._model.device)
            else:
                if hasattr(self._processor, "apply_chat_template"):
                    msgs = [{"role": "user", "content": text}]
                    prompt = self._processor.apply_chat_template(
                        msgs, add_generation_prompt=True
                    )
                    inputs = self._processor(
                        text=prompt, return_tensors="pt"
                    ).to(self._model.device)
                else:
                    inputs = self._processor(
                        text=text, return_tensors="pt"
                    ).to(self._model.device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 0.01),
                    do_sample=temperature > 0.01,
                )

            input_len = inputs.get("input_ids", torch.tensor([])).shape[-1] if "input_ids" in inputs else 0
            generated = output_ids[0][input_len:]
            return self._processor.decode(generated, skip_special_tokens=True).strip()

        except Exception as e:
            logger.error("Generation failed: %s", e, exc_info=True)
            raise

    # ── Helpers ──────────────────────────────────────────────────────

    def _extract_content(self, messages: List[Dict[str, Any]]):
        """Extract text and PIL images from OpenAI-style messages."""
        from PIL import Image

        text_parts = []
        images = []

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            url_data = item.get("image_url", {})
                            url = url_data.get("url", "") if isinstance(url_data, dict) else url_data
                            if url:
                                try:
                                    img = self._decode_image(url)
                                    images.append(img)
                                except Exception as e:
                                    logger.warning("Failed to decode image: %s", e)

        return "\n".join(text_parts), images

    @staticmethod
    def _decode_image(data_url: str):
        """Decode base64 data URL to PIL Image."""
        from PIL import Image
        if data_url.startswith("data:"):
            _, encoded = data_url.split(",", 1)
        else:
            encoded = data_url
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

    # ── Lifecycle ────────────────────────────────────────────────────

    async def unload(self):
        """Release model from GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
        logger.info("NanoVLM model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def available_models(self) -> List[Dict[str, str]]:
        if self._loaded:
            return [
                {"id": "nanovlm", "object": "model", "owned_by": "nvidia-edge"},
            ]
        return []
