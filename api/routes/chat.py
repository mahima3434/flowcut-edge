"""
OpenAI-compatible /v1/chat/completions endpoint.
Routes all requests to LLaVA-NeXT for both vision and text.
"""

import time
import uuid
import logging
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("flowcut-edge")
router = APIRouter()


# ── Request/Response Models (OpenAI-compatible) ─────────────────────

class ChatMessage(BaseModel):
    role: str
    content: Any  # str or list of content parts (for vision)
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


class FunctionDef(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict] = None


class ToolDef(BaseModel):
    type: str = "function"
    function: FunctionDef


class ChatCompletionRequest(BaseModel):
    model: str = "llava-v1.6"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    tools: Optional[List[ToolDef]] = None
    tool_choice: Optional[Any] = None
    stream: Optional[bool] = False


class ChatChoice(BaseModel):
    index: int = 0
    message: Dict[str, Any]
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage


# ── Helpers ──────────────────────────────────────────────────────────

def _has_images(messages: List[ChatMessage]) -> bool:
    """Check if any message contains image content."""
    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


# ── Endpoint ─────────────────────────────────────────────────────────

@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    """OpenAI-compatible chat completions — routes to LLaVA."""
    manager = request.app.state.model_manager

    if not manager.is_loaded:
        raise HTTPException(503, "Vision model is still loading, please wait...")

    if req.stream:
        raise HTTPException(501, "Streaming not yet supported")

    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    use_vision = _has_images(req.messages)

    try:
        if use_vision:
            logger.info("LLaVA vision request — %d messages", len(messages))
            result = await manager.generate_vision(
                messages=messages,
                max_tokens=req.max_tokens or 1024,
                temperature=req.temperature or 0.7,
            )
        else:
            logger.info("LLaVA text request — %d messages", len(messages))
            result = await manager.generate_text(
                messages=messages,
                max_tokens=req.max_tokens or 1024,
                temperature=req.temperature or 0.7,
                tools=[t.model_dump() for t in req.tools] if req.tools else None,
            )
    except Exception as e:
        logger.error("Generation failed: %s", e, exc_info=True)
        raise HTTPException(500, f"Generation failed: {e}")

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model="llava-v1.6",
        choices=[
            ChatChoice(
                message={
                    "role": "assistant",
                    "content": result["content"],
                },
                finish_reason=result.get("finish_reason", "stop"),
            )
        ],
        usage=Usage(),
    )
