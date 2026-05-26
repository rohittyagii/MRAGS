from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from mrags.errors import VLMTimeoutError


class VLMClient(ABC):
    @abstractmethod
    async def describe_image(self, base64_image: str) -> str:
        """Time Complexity: O(N)
        Space Complexity: O(N)
        """
        raise NotImplementedError


class NoopVLMClient(VLMClient):
    def __init__(self, fallback_summary: str = "Image summary unavailable.") -> None:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self._fallback_summary = fallback_summary

    async def describe_image(self, base64_image: str) -> str:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self._fallback_summary


class OpenAIVLMClient(VLMClient):
    def __init__(
        self,
        api_key: str,
        model: str,
        prompt: str,
        timeout_s: int,
        semaphore: asyncio.Semaphore,
        session: aiohttp.ClientSession,
    ) -> None:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self._api_key = api_key
        self._model = model
        self._prompt = prompt
        self._timeout_s = timeout_s
        self._semaphore = semaphore
        self._session = session

    async def describe_image(self, base64_image: str) -> str:
        """Time Complexity: O(N)
        Space Complexity: O(N)
        """
        async with self._semaphore:
            payload = _build_payload(self._model, self._prompt, base64_image)
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            try:
                async with self._session.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self._timeout_s,
                ) as response:
                    data = await response.json()
                    return _parse_response(data)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                raise VLMTimeoutError(str(exc)) from exc


def _build_payload(model: str, prompt: str, base64_image: str) -> dict[str, Any]:
    """Time Complexity: O(1)
    Space Complexity: O(1)
    """
    return {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
    }


def _parse_response(data: dict[str, Any]) -> str:
    """Time Complexity: O(1)
    Space Complexity: O(1)
    """
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return (message.get("content") or "").strip()
