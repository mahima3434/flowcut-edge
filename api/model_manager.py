"""
Model manager — loads and manages NVIDIA models on Jetson.
Uses TensorRT-LLM or Hugging Face Transformers depending on availability.
"""

import logging
import asyncio
import torch
from typing import Optional, List, Dict, Any

logger = logging.getLogger("flowcut-edge")


class ModelManager:
    """Manages Phi-3.5 Vision and Nemotron-Mini (text) models."""

    def __init__(self):
        self.text_model = None
        self.text_tokenizer = None
        self.vision_model = None
        self.vision_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_model_name = ""
        self.vision_model_name = ""
        self._loaded = False
        self.load_error = None

    async def load_models(
        self,
        text_model_id: str,
        vision_model_id: str,
        device: str = "cuda",
    ):
        """Load both models. Runs in thread to avoid blocking."""
        self.device = device
        self.text_model_name = text_model_id
        self.vision_model_name = vision_model_id
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self):
        """Synchronous model loading."""
        try:
            logger.info("Loading text model: %s", self.text_model_name)
            self._load_text_model()
            
            if self.vision_model_name and self.vision_model_name.lower() not in ("none", "disabled", "false", ""):
                logger.info("Loading vision model: %s", self.vision_model_name)
                self._load_vision_model()
            else:
                logger.info("Vision model loading disabled by configuration")
                
            self._loaded = True
            self.load_error = None
        except Exception as e:
            logger.error("Failed to load models: %s", e)
            self.load_error = str(e)
            self._loaded = False

    def _load_text_model(self):
        """Load Nemotron-Mini for text/tool-calling."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.text_tokenizer = AutoTokenizer.from_pretrained(
                self.text_model_name,
                trust_remote_code=True,
            )
            load_kwargs = dict(
                trust_remote_code=True,
            )
            if self.device == "cuda":
                load_kwargs["torch_dtype"] = torch.bfloat16
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["torch_dtype"] = torch.float32
            self.text_model = AutoModelForCausalLM.from_pretrained(
                self.text_model_name,
                **load_kwargs,
            )
            logger.info("Text model loaded successfully")
        except Exception as e:
            logger.error("Failed to load text model: %s", e)
            # We don't raise here anymore to allow partial loading or graceful failure
            raise e

    def _load_vision_model(self):
        """Load Phi-3.5 Vision for vision-language tasks."""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig

            self.vision_processor = AutoProcessor.from_pretrained(
                self.vision_model_name,
                trust_remote_code=True,
            )
            config = AutoConfig.from_pretrained(
                self.vision_model_name,
                trust_remote_code=True,
            )
            config._attn_implementation = "eager"

            vision_kwargs = dict(
                config=config,
                trust_remote_code=True,
            )
            if self.device == "cuda":
                vision_kwargs["torch_dtype"] = torch.bfloat16
                vision_kwargs["device_map"] = "auto"
            else:
                vision_kwargs["torch_dtype"] = torch.float32
            self.vision_model = AutoModelForCausalLM.from_pretrained(
                self.vision_model_name,
                **vision_kwargs,
            )
            logger.info("Vision model loaded successfully")
        except Exception as e:
            logger.error("Failed to load vision model: %s", e)
            raise e

    async def generate_text(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Generate text completion (OpenAI-compatible format)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_text_sync,
            messages,
            max_tokens,
            temperature,
            tools,
        )

    def _generate_text_sync(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tools: Optional[List[Dict]],
    ) -> Dict[str, Any]:
        """Synchronous text generation."""
        # Build prompt from messages
        prompt = self.text_tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.text_tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.text_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=0.9,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = self.text_tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {
            "content": response_text,
            "finish_reason": "stop",
        }

    async def generate_vision(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Generate vision-language completion."""
        if not self.vision_model:
            raise RuntimeError("Vision model is disabled or not loaded.")
            
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_vision_sync,
            messages,
            max_tokens,
            temperature,
        )

    def _generate_vision_sync(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        """Synchronous vision generation."""
        import base64
        from io import BytesIO
        from PIL import Image
        import requests

        # Extract images and text from messages
        images = []
        text_parts = []

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(part["text"])
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            # Base64 encoded
                            b64_data = url.split(",", 1)[1]
                            img = Image.open(BytesIO(base64.b64decode(b64_data)))
                        else:
                            # URL
                            img = Image.open(BytesIO(requests.get(url).content))
                        images.append(img)
            elif isinstance(content, str):
                text_parts.append(content)

        prompt = "\n".join(text_parts) if text_parts else "Describe this image."

        # Build Phi-3.5 Vision prompt with image placeholders
        image_placeholders = "".join(
            [f"<|image_{i+1}|>\n" for i in range(len(images))]
        ) if images else ""
        full_prompt = f"{image_placeholders}{prompt}"

        # Process with Phi-3.5 Vision
        if images:
            inputs = self.vision_processor(
                text=full_prompt,
                images=images,
                return_tensors="pt",
            ).to(self.device)
        else:
            inputs = self.vision_processor(
                text=full_prompt,
                return_tensors="pt",
            ).to(self.device)

        with torch.no_grad():
            output_ids = self.vision_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                eos_token_id=self.vision_processor.tokenizer.eos_token_id,
            )

        # Decode only new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response_text = self.vision_processor.decode(
            new_tokens, skip_special_tokens=True
        )

        return {
            "content": response_text,
            "finish_reason": "stop",
        }

    async def unload(self):
        """Release GPU memory."""
        if self.text_model:
            del self.text_model
            self.text_model = None
        if self.vision_model:
            del self.vision_model
            self.vision_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
        logger.info("Models unloaded, GPU memory released")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def available_models(self) -> List[Dict[str, str]]:
        """Return list of available models in OpenAI format."""
        models = []
        if self.text_model:
            models.append({
                "id": "nemotron-mini-4b",
                "object": "model",
                "owned_by": "nvidia",
                "type": "text",
            })
        if self.vision_model:
            models.append({
                "id": "phi-3.5-vision",
                "object": "model",
                "owned_by": "microsoft",
                "type": "vision",
            })
        return models
