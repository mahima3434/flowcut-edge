"""
Model manager — Placeholder for future use.
Text and Vision models (Nemotron-Mini, Phi-3.5) have been removed to save resources.
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any

logger = logging.getLogger("flowcut-edge")


class ModelManager:
    """
    Stub manager.
    Phi-3.5 Vision and Nemotron-Mini have been removed.
    """

    def __init__(self):
        self._loaded = True 
        self.load_error = None
        # Keep these properties to avoid breaking health/main
        self.text_model_name = "disabled"
        self.vision_model_name = "disabled"

    async def load_models(
        self,
        text_model_id: str,
        vision_model_id: str,
        device: str = "cuda",
    ):
        """No-op."""
        logger.info("Text/Vision models are disabled in code. Skipping load.")
        pass

    async def generate_text(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Stub."""
        raise NotImplementedError("Text generation is disabled on this device.")

    async def generate_vision(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Stub."""
        raise NotImplementedError("Vision generation is disabled on this device.")

    async def unload(self):
        """No-op."""
        pass

    @property
    def is_loaded(self) -> bool:
        return True

    def available_models(self) -> List[Dict[str, str]]:
        """Return empty list."""
        return []

