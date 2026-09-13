"""Builds the SDK Agent for a pipeline stage - the OpenAI Agents SDK layer and nothing else.
The executor decides WHAT to ask; this decides HOW the agent is configured."""

from agents import Agent, ModelSettings, set_default_openai_client, set_tracing_disabled
from openai import AsyncOpenAI
from openai.types.shared import Reasoning
from pydantic import BaseModel

from app.config import Settings
from app.enums import Stage


class AgentFactory:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncOpenAI | None = None

    def model_for(self, stage: Stage) -> str:
        s = self._settings
        return s.openai_match_model if stage is Stage.MATCH else s.openai_model

    def effort_for(self, stage: Stage) -> str:
        s = self._settings
        return s.match_reasoning_effort if stage is Stage.MATCH else s.reasoning_effort

    def build(self, stage: Stage, instructions: str, output_type: type[BaseModel]) -> Agent:
        self._ensure_client()
        return Agent(
            name=f"disco-{stage}",
            instructions=instructions,
            model=self.model_for(stage),
            model_settings=ModelSettings(
                reasoning=Reasoning(effort=self.effort_for(stage)),
                verbosity="low",
                max_tokens=self._settings.llm_max_output_tokens,
                timeout=self._settings.llm_timeout_s,
            ),
            output_type=output_type,
        )

    def _ensure_client(self) -> None:
        """One client per process so every run shares the connection pool."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._settings.openai_api_key or None)
            set_default_openai_client(self._client, use_for_tracing=self._settings.tracing_enabled)
            if not self._settings.tracing_enabled:
                set_tracing_disabled(True)
