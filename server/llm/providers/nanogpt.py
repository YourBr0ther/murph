"""
Murph - NanoGPT LLM Provider
NanoGPT cloud API integration (OpenAI-compatible).
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


class NanoGPTProvider(LLMProvider):
    """
    NanoGPT cloud API provider.

    OpenAI-compatible API supporting 475+ models including
    Claude, GPT, Gemini, and DeepSeek.

    API Documentation: https://nano-gpt.com/api
    - POST /chat/completions - Chat completions
    - GET /models - List available models
    """

    def __init__(self, config: LLMConfig) -> None:
        """
        Initialize NanoGPT provider.

        Args:
            config: LLM configuration with NanoGPT settings
        """
        super().__init__(config)
        self._base_url = config.nanogpt_base_url.rstrip("/")
        self._api_key = config.nanogpt_api_key
        self._text_model = config.nanogpt_model
        self._vision_model = config.nanogpt_vision_model
        self._timeout = config.request_timeout_seconds
        self._session: Any = None  # aiohttp.ClientSession

    async def _ensure_session(self) -> Any:
        """Lazy-initialize HTTP session."""
        if self._session is None or self._session.closed:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=self._timeout)
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
            )
        return self._session

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Send completion request to NanoGPT.

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
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        start = time.time()
        try:
            async with session.post(
                f"{self._base_url}/chat/completions",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

            latency_ms = (time.time() - start) * 1000

            # Extract response content
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            # Update stats
            self._stats.record_call(
                latency_ms=latency_ms,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
            )

            return LLMResponse(
                content=content,
                model=data.get("model", model),
                usage=usage,
                latency_ms=latency_ms,
                provider="nanogpt",
            )

        except Exception as e:
            self._stats.record_error()
            logger.error(f"NanoGPT completion error: {e}")
            raise

    async def complete_with_vision(
        self,
        request: LLMRequest,
        image: np.ndarray,
    ) -> LLMResponse:
        """
        Send vision completion request to NanoGPT.

        Uses OpenAI-compatible multimodal message format.

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

        # Build messages with multimodal content
        messages = []
        for i, msg in enumerate(request.messages):
            if i == len(request.messages) - 1 and msg.role == "user":
                # Last user message gets the image
                messages.append({
                    "role": msg.role,
                    "content": [
                        {"type": "text", "text": msg.content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                    ],
                })
            else:
                messages.append(msg.to_dict())

        payload = {
            "model": self._vision_model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        start = time.time()
        try:
            async with session.post(
                f"{self._base_url}/chat/completions",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

            latency_ms = (time.time() - start) * 1000

            # Extract response content
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            # Update stats
            self._stats.record_call(
                latency_ms=latency_ms,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
            )

            return LLMResponse(
                content=content,
                model=data.get("model", self._vision_model),
                usage=usage,
                latency_ms=latency_ms,
                provider="nanogpt",
            )

        except Exception as e:
            self._stats.record_error()
            logger.error(f"NanoGPT vision error: {e}")
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
        Quick check if NanoGPT is available.

        Returns True if API key is configured.
        """
        return self._api_key is not None and len(self._api_key) > 0

    async def health_check(self) -> bool:
        """
        Check if NanoGPT API is reachable.

        Returns:
            True if API responds to /models endpoint
        """
        if not self.is_available():
            return False

        try:
            session = await self._ensure_session()
            async with session.get(f"{self._base_url}/models") as resp:
                if resp.status == 200:
                    self._health_check_result = True
                    return True
        except Exception as e:
            logger.warning(f"NanoGPT health check failed: {e}")

        self._health_check_result = False
        return False

    async def list_models(self) -> list[dict[str, Any]]:
        """
        List available models on NanoGPT.

        Returns:
            List of model info dicts
        """
        if not self.is_available():
            return []

        try:
            session = await self._ensure_session()
            async with session.get(f"{self._base_url}/models") as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("data", [])
        except Exception as e:
            logger.error(f"Failed to list NanoGPT models: {e}")
            return []

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def __repr__(self) -> str:
        key_display = "***" if self._api_key else "None"
        return (
            f"NanoGPTProvider(url={self._base_url}, "
            f"api_key={key_display}, "
            f"text={self._text_model}, vision={self._vision_model})"
        )
