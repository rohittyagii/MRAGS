from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from mrags.models import LMMAnswer, RetrievedElement

from .prompt_builder import PromptBuilder


class LMMClient:
    """Wrapper around an Async OpenAI client to ask LLM questions using context.

    Use this when `LMM_BACKEND` is OpenAI. The class handles building the
    chat messages and returning a clean `LMMAnswer`.
    """

    def __init__(self, client: AsyncOpenAI, model: str, system_prompt: str) -> None:
        """Initialize the OpenAI-backed LMM client.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self._client = client
        self._model = model
        self._prompt_builder = PromptBuilder(system_prompt)

    async def answer(self, question: str, elements: list[RetrievedElement]) -> LMMAnswer:
        """Ask the remote model to answer using the retrieved elements as context.

        Time Complexity: O(N)
        Space Complexity: O(N)
        """
        messages = self._prompt_builder.build_messages(question, elements)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.2,
        )
        content = response.choices[0].message.content or ""
        return LMMAnswer(answer=content.strip(), sources=elements)


class LocalLMMClient:
    """Local LLM client using `llama_cpp` to run GGUF models on-device.

    This is used when `LMM_BACKEND` is set to `local` and a `LMM_MODEL_PATH`
    points at a GGUF file.
    """

    def __init__(
        self,
        model_path: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        n_ctx: int,
        n_gpu_layers: int,
        n_threads: int,
    ) -> None:
        """Initialize the on-device Llama instance.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        from llama_cpp import Llama

        self._llama = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
        )
        self._prompt_builder = PromptBuilder(system_prompt)
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def answer(self, question: str, elements: list[RetrievedElement]) -> LMMAnswer:
        """Generate an answer locally by building a prompt and calling the model.

        Time Complexity: O(N)
        Space Complexity: O(N)
        """
        prompt = self._prompt_builder.build_text_prompt(question, elements)
        return await asyncio.to_thread(self._generate, prompt, elements)

    def _generate(self, prompt: str, elements: list[RetrievedElement]) -> LMMAnswer:
        """Synchronous generation helper that calls the llama instance.

        Time Complexity: O(N)
        Space Complexity: O(N)
        """
        output = self._llama(
            prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        text = output["choices"][0]["text"]
        return LMMAnswer(answer=text.strip(), sources=elements)
