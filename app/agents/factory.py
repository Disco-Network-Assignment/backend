"""Builds the SDK Agent for a pipeline stage - the OpenAI Agents SDK layer and nothing else.

WHAT a stage asks (the prompt, the inputs, the checks) belongs to the executor; HOW an agent is
configured (model per stage, reasoning effort, output cap, timeout, the shared client) lives
here, so swapping a model or tuning effort is a settings change."""

from dataclasses import dataclass

from agents import Agent, ModelSettings, set_default_openai_client, set_tracing_disabled
from openai import AsyncOpenAI
from openai.types.shared import Reasoning
from pydantic import BaseModel

from app.config import Settings
from app.enums import Stage


@dataclass(frozen=True)
class StageModelSpec:
    model: str
    reasoning_effort: str
    max_output_tokens: int
    timeout_s: float


class AgentFactory:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncOpenAI | None = None

    def spec(self, stage: Stage) -> StageModelSpec:
        s = self._settings
        models = {
            Stage.INTAKE: (s.openai_intake_model, s.intake_reasoning_effort),
            Stage.MATCH: (s.openai_match_model, s.match_reasoning_effort),
            Stage.PERSONAS: (s.openai_persona_model, s.persona_reasoning_effort),
            Stage.CREATIVE: (s.openai_creative_model, s.creative_reasoning_effort),
            Stage.SUMMARY: (s.openai_summary_model, s.summary_reasoning_effort),
        }
        try:
            model, effort = models[stage]
        except KeyError:
            raise ValueError(f"stage {stage} has no agent") from None
        return StageModelSpec(model=model, reasoning_effort=effort,
                              max_output_tokens=s.llm_max_output_tokens, timeout_s=s.llm_timeout_s)

    def build(self, stage: Stage, instructions: str, output_type: type[BaseModel]) -> Agent:
        self._ensure_client()
        spec = self.spec(stage)
        return Agent(
            name=f"disco-{stage}",
            instructions=instructions,
            model=spec.model,
            model_settings=ModelSettings(
                reasoning=Reasoning(effort=spec.reasoning_effort),
                verbosity="low",
                max_tokens=spec.max_output_tokens,
                timeout=spec.timeout_s,
            ),
            output_type=output_type,
        )

    def _ensure_client(self) -> None:
        """One client per process: every run shares its connection pool and the first request
        alone pays the TLS handshake. Tracing rides the same key unless disabled."""
        if self._client is not None:
            return
        self._client = AsyncOpenAI(api_key=self._settings.openai_api_key or None)
        set_default_openai_client(self._client, use_for_tracing=self._settings.tracing_enabled)
        if not self._settings.tracing_enabled:
            set_tracing_disabled(True)
