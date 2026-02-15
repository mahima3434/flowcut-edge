"""
Model manager — Phi-3.5 Vision on GB10.
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
DEFAULT_MODEL_ID = "microsoft/Phi-3.5-vision-instruct"


class _ProcessorShim:
    """
    A small wrapper that behaves like a processor:
      - __call__(text=..., images=...) -> dict of tensors
      - decode(ids) -> string
      - apply_chat_template(msgs, add_generation_prompt=True) if tokenizer supports it
    """

    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def apply_chat_template(self, msgs, add_generation_prompt: bool = True):
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=add_generation_prompt, tokenize=False
            )
        # Fallback: join message contents
        return "\n".join(m.get("content", "") for m in msgs if isinstance(m, dict))

    def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):
        out: Dict[str, torch.Tensor] = {}

        if text is not None:
            tok = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=False,
                truncation=False,
            )
            out.update(tok)

        if images is not None:
            img = self.image_processor(images=images, return_tensors=return_tensors)
            out.update(img)

        return out

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


class ModelManager:
    """
    Manages Phi-3.5 Vision for vision + text tasks.
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
        """Load the vision model."""
        self._device = device
        self.model_id = model_id if model_id and model_id != "disabled" else DEFAULT_MODEL_ID

        logger.info("Loading vision model: %s on %s", self.model_id, self._device)

        try:
            await asyncio.to_thread(self._load_sync)
            self._loaded = True
            self.load_error = None
            logger.info("Vision model loaded ✓")
        except Exception as e:
            self._loaded = False
            self.load_error = str(e)
            logger.error("Failed to load vision model: %s", e, exc_info=True)
            raise

    def _load_sync(self):
        """Synchronous model loading (run in thread)."""
        from transformers import (
            AutoConfig,
            AutoProcessor,
            AutoTokenizer,
            AutoImageProcessor,
            AutoModelForCausalLM,
            LlavaConfig,       # <--- ADD THIS
            CONFIG_MAPPING,    # <--- ADD THIS
        )

        # ---------------------------------------------------------------------
        # PATCH: Register 'llava_llama' so AutoConfig doesn't crash
        # ---------------------------------------------------------------------
        try:
            CONFIG_MAPPING["llava_llama"] = LlavaConfig
        except Exception as e:
            logger.warning(f"Could not patch CONFIG_MAPPING: {e}")
        # ---------------------------------------------------------------------

        # ---- Processor (robust) --------------------------------------------
        try:
            # Works for models that ship a root-level processor config
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            logger.info("Loaded processor via AutoProcessor")
        except Exception as e:
            logger.warning(
                "AutoProcessor failed (%s). Falling back to llm/ + vision_tower/ loaders.",
                str(e),
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                subfolder="llm",
                trust_remote_code=True,
                use_fast=False,
            )
            image_processor = AutoImageProcessor.from_pretrained(
                self.model_id,
                subfolder="vision_tower",
                trust_remote_code=True,
            )
            self._processor = _ProcessorShim(tokenizer=tokenizer, image_processor=image_processor)
            logger.info("Loaded processor via fallback shim (llm/ + vision_tower/)")

        # ---- Model ----------------------------------------------------------
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        device_map = "auto" if self._device == "cuda" else None

        # Now this line won't crash because we patched CONFIG_MAPPING above
        config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        
        # We still normalize the name to "llava" so the rest of the pipeline is happy
        if getattr(config, "model_type", None) == "llava_llama":
            logger.info("Legacy 'llava_llama' model detected. Normalizing to 'llava'.")
            config.model_type = "llava"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            config=config,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        logger.info("Loaded as CasualLM model")

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

    def _to_device(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move tensor dict to the model device."""
        dev = self._model.device if self._model is not None else self._device
        return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}

    def _generate_sync(
        self,
        text: str,
        images: list,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Synchronous generation for Phi-3.5 Vision."""
        try:
            # Build content with <|image_N|> placeholders for each image
            if images:
                placeholders = "".join(
                    f"<|image_{i+1}|>\n" for i in range(len(images))
                )
                user_content = f"{placeholders}{text}"
            else:
                user_content = text

            msgs = [{"role": "user", "content": user_content}]

            # Phi-3.5 Vision: use processor.tokenizer for chat template
            tokenizer = getattr(self._processor, "tokenizer", self._processor)
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

            # Build processor inputs
            if images:
                inputs = self._processor(
                    text=prompt,
                    images=images,
                    return_tensors="pt",
                )
            else:
                inputs = self._processor(
                    text=prompt,
                    return_tensors="pt",
                )

            inputs = self._to_device(inputs)

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
        logger.info("Vision model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def available_models(self) -> List[Dict[str, str]]:
        if self._loaded:
            return [
                {"id": "phi-3.5-vision", "object": "model", "owned_by": "microsoft"},
            ]
        return []
