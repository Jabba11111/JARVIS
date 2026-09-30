# RESEARCH: NVIDIA API catalog (build.nvidia.com) exposes an OpenAI-compatible
#   /v1/chat/completions endpoint; the official openai SDK works with base_url.
# DECISION: Reuse the openai SDK (already a dependency) + the Claude prompt/parser
# ALT: langchain-nvidia-ai-endpoints — extra dependency for no added value here

from __future__ import annotations

import json

from loguru import logger

from config import Settings
from key_pool import get_key_pool
from synthesis.anthropic_engine import SYNTHESIS_PROMPT, AnthropicSynthesisEngine
from synthesis.models import SynthesisRequest, SynthesisResult


class NvidiaSynthesisEngine(AnthropicSynthesisEngine):
    """Synthesizes person intelligence reports using an NVIDIA-hosted model."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._key_pool = get_key_pool("nvidia", settings.nvidia_api_key)
        self._clients: dict[str, object] = {}

    @property
    def configured(self) -> bool:
        return len(self._key_pool) > 0

    def _get_client(self, api_key: str | None = None):
        if self._client is not None:  # injected (tests)
            return self._client
        key = api_key or self._key_pool.next_key()
        if key not in self._clients:
            from openai import AsyncOpenAI

            self._clients[key] = AsyncOpenAI(
                api_key=key,
                base_url=self._settings.nvidia_base_url,
                timeout=60.0,
                max_retries=0,  # the key pool handles retries across keys
            )
        return self._clients[key]

    async def _complete(self, prompt: str) -> str:
        """Call the model, rotating to the next key on rate-limit or auth errors."""
        from openai import AuthenticationError, PermissionDeniedError, RateLimitError

        attempts = max(1, len(self._key_pool))
        last_exc: Exception | None = None
        for _ in range(attempts):
            key = self._key_pool.next_key()
            try:
                response = await self._get_client(key).chat.completions.create(
                    model=self._settings.nvidia_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=4096,
                )
                if not response.choices:
                    return ""
                return response.choices[0].message.content or ""
            except RateLimitError as exc:
                last_exc = exc
                if key:
                    self._key_pool.mark_failed(key, cooldown_s=60.0)
            except (AuthenticationError, PermissionDeniedError) as exc:
                last_exc = exc
                if key:
                    self._key_pool.mark_failed(key, cooldown_s=3600.0)
        assert last_exc is not None
        raise last_exc

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Synthesize enrichment data into a structured person report."""
        model = self._settings.nvidia_model
        logger.info(
            "NvidiaSynthesisEngine.synthesize person={} model={}", request.person_name, model
        )

        if not self.configured:
            return SynthesisResult(
                person_name=request.person_name,
                success=False,
                error="NVIDIA API key not configured (NVIDIA_API_KEY missing)",
            )

        try:
            prompt = SYNTHESIS_PROMPT.format(
                person_name=request.person_name,
                raw_data=self._build_raw_data_block(request),
            )
            response_text = await self._complete(prompt)
            if not response_text:
                return SynthesisResult(
                    person_name=request.person_name,
                    success=False,
                    error="NVIDIA model returned empty response",
                )

            dossier = self._parse_response(response_text, request.person_name)
            return SynthesisResult(
                person_name=request.person_name,
                summary=dossier.summary,
                occupation=dossier.title,
                organization=dossier.company,
                dossier=dossier,
                confidence_score=0.7,
            )

        except json.JSONDecodeError as e:
            logger.error("Failed to parse NVIDIA response as JSON: {}", e)
            return SynthesisResult(
                person_name=request.person_name,
                success=False,
                error=f"NVIDIA response was not valid JSON: {e}",
            )
        except Exception as e:
            logger.error("NVIDIA synthesis failed: {}", e)
            return SynthesisResult(
                person_name=request.person_name,
                success=False,
                error=f"Synthesis error: {e}",
            )
