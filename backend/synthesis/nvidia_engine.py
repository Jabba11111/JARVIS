# RESEARCH: NVIDIA API catalog (build.nvidia.com) exposes an OpenAI-compatible
#   /v1/chat/completions endpoint; the official openai SDK works with base_url.
# DECISION: Reuse the openai SDK (already a dependency) + the Claude prompt/parser
# ALT: langchain-nvidia-ai-endpoints — extra dependency for no added value here

from __future__ import annotations

import json

from loguru import logger

from config import Settings
from synthesis.anthropic_engine import SYNTHESIS_PROMPT, AnthropicSynthesisEngine
from synthesis.models import SynthesisRequest, SynthesisResult


class NvidiaSynthesisEngine(AnthropicSynthesisEngine):
    """Synthesizes person intelligence reports using an NVIDIA-hosted model."""

    def __init__(self, settings: Settings):
        super().__init__(settings)

    @property
    def configured(self) -> bool:
        return bool(self._settings.nvidia_api_key)

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self._settings.nvidia_api_key,
                base_url=self._settings.nvidia_base_url,
                timeout=60.0,
            )
        return self._client

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
            response = await self._get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=4096,
            )
            response_text = (response.choices[0].message.content or "") if response.choices else ""
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
