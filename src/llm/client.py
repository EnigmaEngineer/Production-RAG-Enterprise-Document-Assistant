"""
LLM client for vLLM (OpenAI-compatible API).

Handles prompt construction, citation-aware system prompts,
retries, circuit breaking, and timeout enforcement.
"""

from __future__ import annotations

import time

from openai import OpenAI, APITimeoutError, APIConnectionError

from src.config import settings
from src.models.schemas import Chunk
from src.citation.formatter import citation_formatter
from src.utils.logging import get_logger
from src.utils.resilience import llm_breaker, llm_retry

log = get_logger(__name__)

# ── System prompt instructing the LLM to use citations ──────────────────────

SYSTEM_PROMPT = """\
You are a precise enterprise document assistant. Answer the user's question
using ONLY the provided reference passages below.

Rules:
1. Cite sources using bracket notation [1], [2], etc. matching the passage numbers.
2. If the passages do not contain enough information, say so explicitly.
3. Never invent facts not present in the passages.
4. Keep answers concise and professional.
5. When multiple passages support a claim, cite all of them.

Reference passages:
{context}
"""


class LLMClient:
    """
    Synchronous client wrapping the OpenAI SDK pointed at vLLM.
    Designed for use inside FastAPI's thread pool (run_in_executor).
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ):
        self._base_url = base_url or settings.llm_base_url
        self._model = model or settings.llm_model
        self._max_tokens = max_tokens or settings.llm_max_tokens
        self._temperature = (
            temperature if temperature is not None else settings.llm_temperature
        )
        self._timeout = timeout or settings.llm_timeout_s

        self._client = OpenAI(
            base_url=self._base_url,
            api_key="not-needed",  # vLLM doesn't require an API key
            timeout=self._timeout,
        )
        log.info(
            "llm_client.init",
            base_url=self._base_url,
            model=self._model,
        )

    @llm_retry
    def generate(self, query: str, context_chunks: list[Chunk]) -> tuple[str, float]:
        """
        Generate an answer with citations.

        Returns:
            (answer_text, generation_latency_ms)
        """
        # Build context from citation formatter
        context_text = citation_formatter.format_context_for_llm(context_chunks)
        system = SYSTEM_PROMPT.format(context=context_text)

        t0 = time.perf_counter()
        try:
            response = llm_breaker.call(
                self._client.chat.completions.create,
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )

            answer = response.choices[0].message.content or ""
            elapsed = (time.perf_counter() - t0) * 1000

            log.info(
                "llm_client.generate",
                model=self._model,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens
                if response.usage
                else 0,
                latency_ms=round(elapsed, 1),
            )
            return answer.strip(), elapsed

        except (APITimeoutError, APIConnectionError) as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            log.error(
                "llm_client.timeout", error=str(exc), latency_ms=round(elapsed, 1)
            )
            raise TimeoutError(f"LLM request timed out after {elapsed:.0f}ms") from exc

        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            log.error("llm_client.error", error=str(exc), latency_ms=round(elapsed, 1))
            raise

    def health_check(self) -> bool:
        """Quick check that vLLM is responding."""
        try:
            models = self._client.models.list()
            return len(models.data) > 0
        except Exception:
            return False


class DummyLLMClient:
    """
    Mock LLM client for testing without a running vLLM instance.
    Returns a canned response that references the provided chunks.
    """

    def __init__(self, model: str = "dummy-model"):
        self._model = model

    def generate(self, query: str, context_chunks: list[Chunk]) -> tuple[str, float]:
        t0 = time.perf_counter()
        # Build a simple answer referencing each chunk
        refs = ", ".join(f"[{i + 1}]" for i in range(len(context_chunks)))
        answer = (
            f"Based on the provided documents {refs}, here is the answer to your question: "
            f"'{query}'. The relevant information can be found in the referenced passages."
        )
        elapsed = (time.perf_counter() - t0) * 1000
        return answer, elapsed

    def health_check(self) -> bool:
        return True


def get_llm_client(use_dummy: bool = False, **kwargs) -> LLMClient | DummyLLMClient:
    """Factory for LLM client."""
    if use_dummy:
        return DummyLLMClient(model=kwargs.get("model", "dummy-model"))
    return LLMClient(**kwargs)
