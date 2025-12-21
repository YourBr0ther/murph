"""
Murph - Ollama LLM Provider
Local Ollama server integration for LLM capabilities.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import LLMProvider

if TYPE_CHECKING:
    from ..config import LLMConfig
    from ..types import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """
    Ollama local model provider.

    Connects to a local Ollama server for both text and vision inference.
    Supports models like llama3.2 (text) and llama3.2-vision (multimodal).

    Ollama API:
    - POST /api/chat - Chat completions
    - GET /api/tags - List available models
    """

    def __init__(self, config: LLMConfig) -> None:
        """
        Initialize Ollama provider.

        Args:
            config: LLM configuration with Ollama settings
        """
        super().__init__(config)
        self._base_url = config.ollama_base_url.rstrip("/")
        self._text_model = config.ollama_model
        self._vision_model = config.ollama_vision_model
        self._timeout = config.request_timeout_seconds
        self._session: Any = None  # aiohttp.ClientSession

    async def _ensure_session(self) -> Any:
        """Lazy-initialize HTTP session."""
        if self._session is None or self._session.closed:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=self._timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
        return self._session

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Send completion request to Ollama.

        Args:
            request: The completion request

        Returns:
            LLM response

        Raises:
            Exception: If request fails
        """
        from ..types import LLMResponse

        session = await self._ensure_session()
        model = request.model or self._text_model

        payload = {
            "model": model,
            "messages": [m.to_dict() for m in request.messages],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        start = time.time()
        try:
            async with session.post(
                f"{self._base_url}/api/chat",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

            latency_ms = (time.time() - start) * 1000

            # Extract usage info
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            }
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

            # Update stats
            self._stats.record_call(
                latency_ms=latency_ms,
                tokens_in=usage["prompt_tokens"],
                tokens_out=usage["completion_tokens"],
            )

            return LLMResponse(
                content=data["message"]["content"],
                model=data.get("model", model),
                usage=usage,
                latency_ms=latency_ms,
                provider="ollama",
            )

        except Exception as e:
            self._stats.record_error()
            error_msg = str(e) if str(e) else type(e).__name__
            logger.error(f"Ollama completion error: {error_msg}")
            raise

    async def complete_with_vision(
        self,
        request: LLMRequest,
        image: np.ndarray,
    ) -> LLMResponse:
        """
        Send vision completion request to Ollama.

        Args:
            request: The completion request
            image: RGB numpy array (H, W, 3)

        Returns:
            LLM response

        Raises:
            Exception: If request fails
        """
        from ..types import LLMResponse

        session = await self._ensure_session()

        # Encode image to base64 JPEG
        image_b64 = self._encode_image(image)

        # Build messages with image on the last user message
        messages = []
        for i, msg in enumerate(request.messages):
            msg_dict: dict[str, Any] = {"role": msg.role, "content": msg.content}
            # Attach image to the last message
            if i == len(request.messages) - 1 and msg.role == "user":
                msg_dict["images"] = [image_b64]
            messages.append(msg_dict)

        payload = {
            "model": self._vision_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        start = time.time()
        try:
            async with session.post(
                f"{self._base_url}/api/chat",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

            latency_ms = (time.time() - start) * 1000

            # Extract usage info
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            }
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

            # Update stats
            self._stats.record_call(
                latency_ms=latency_ms,
                tokens_in=usage["prompt_tokens"],
                tokens_out=usage["completion_tokens"],
            )

            return LLMResponse(
                content=data["message"]["content"],
                model=data.get("model", self._vision_model),
                usage=usage,
                latency_ms=latency_ms,
                provider="ollama",
            )

        except Exception as e:
            self._stats.record_error()
            error_msg = str(e) if str(e) else type(e).__name__
            logger.error(f"Ollama vision error: {error_msg}")
            raise

    def _encode_image(self, image: np.ndarray, quality: int = 85) -> str:
        """
        Encode image to base64 JPEG.

        Args:
            image: RGB numpy array
            quality: JPEG quality (1-100)

        Returns:
            Base64-encoded JPEG string
        """
        import cv2

        # Convert RGB to BGR for cv2
        if len(image.shape) == 3 and image.shape[2] == 3:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            bgr = image

        # Encode as JPEG
        _, buffer = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )

        return base64.b64encode(buffer).decode("utf-8")

    def is_available(self) -> bool:
        """
        Quick check if Ollama is available.

        Always returns True since Ollama is local and we can't
        synchronously check without blocking.
        """
        return True

    async def health_check(self) -> bool:
        """
        Check if Ollama server is reachable.

        Returns:
            True if server responds to /api/tags
        """
        try:
            session = await self._ensure_session()
            async with session.get(f"{self._base_url}/api/tags") as resp:
                if resp.status == 200:
                    self._health_check_result = True
                    return True
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")

        self._health_check_result = False
        return False

    async def list_models(self) -> list[str]:
        """
        List available models on Ollama server.

        Returns:
            List of model names
        """
        try:
            session = await self._ensure_session()
            async with session.get(f"{self._base_url}/api/tags") as resp:
                resp.raise_for_status()
                data = await resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def __repr__(self) -> str:
        return (
            f"OllamaProvider(url={self._base_url}, "
            f"text={self._text_model}, vision={self._vision_model})"
        )
