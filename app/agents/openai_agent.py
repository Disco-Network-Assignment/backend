"""The OpenAI Agents SDK layer: build one agent for a stage, run it once, get typed output.

`AgentFactory` decides HOW an agent is configured (model per stage, reasoning effort, output
schema, the shared client). `StructuredRunner` makes the call and owns the single retry: when
the output does not fit the schema, or the stage's business check rejects it, the errors are
quoted back to the model once. Provider and SDK failures are mapped to a FailureKind so the API
can stream a typed `failed` event."""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from agents import (
    Agent,
    ModelSettings,
    RunConfig,
    Runner,
    set_default_openai_client,
    set_tracing_disabled,
)
from agents.exceptions import (
    AgentsException,
    MaxTurnsExceeded,
    ModelBehaviorError,
    ModelRefusalError,
    ModelTimeoutError,
)
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from openai.types.shared import Reasoning
from pydantic import BaseModel

from app.enums import ExecutionMode, FailureKind, Stage
from app.errors import StageError
from app.prompts.loader import PromptLoader, RenderedPrompt
from app.schemas import StageMeta
from app.settings import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# exception -> (kind, message); first match wins
FAILURES: tuple[tuple[type[BaseException] | tuple[type[BaseException], ...], FailureKind, str], ...] = (
    (ModelRefusalError, FailureKind.REFUSAL, "the model declined this request"),
    ((ModelTimeoutError, APITimeoutError, TimeoutError), FailureKind.TIMEOUT, "the model call timed out"),
    (RateLimitError, FailureKind.RATE_LIMIT, "rate limited by the model provider; retry in a minute"),
    ((APIConnectionError, APIError), FailureKind.API, "model provider error"),
    (MaxTurnsExceeded, FailureKind.VALIDATION, "the agent did not finish in one turn"),
    (AgentsException, FailureKind.UNKNOWN, "agent runtime error"),
)


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


@dataclass(frozen=True)
class StageRun(Generic[T]):
    output: T
    meta: StageMeta


class StructuredRunner:
    def __init__(self, settings: Settings, prompts: PromptLoader) -> None:
        self._settings = settings
        self._prompts = prompts

    async def run(self, stage: Stage, agent: Agent, prompt: RenderedPrompt, output_type: type[T],
                  check: Callable[[T], list[str]] | None = None) -> StageRun[T]:
        started = time.perf_counter()
        config = RunConfig(workflow_name="disco-campaign-brain",
                           tracing_disabled=not self._settings.tracing_enabled)
        conversation: str | list = prompt.input
        tokens_in = tokens_out = 0
        for attempt in (0, 1):
            result = None
            try:
                result = await asyncio.wait_for(
                    Runner.run(agent, conversation, max_turns=1, run_config=config),
                    timeout=self._settings.llm_timeout_s + 5)
            except ModelBehaviorError as e:  # the output did not fit the schema
                errors = [f"output did not match the required schema: {e}"]
            except Exception as e:
                raise self._classify(stage, e) from e
            else:
                for response in result.raw_responses:
                    tokens_in += response.usage.input_tokens
                    tokens_out += response.usage.output_tokens
                output = result.final_output_as(output_type)
                errors = check(output) if check else []
                if not errors:
                    return StageRun(output, StageMeta(
                        stage=stage, ms=round((time.perf_counter() - started) * 1000),
                        mode=ExecutionMode.LLM, model=agent.model if isinstance(agent.model, str) else None,
                        reasoning_effort=getattr(agent.model_settings.reasoning, "effort", None),
                        prompt_version=prompt.version, input_tokens=tokens_in,
                        output_tokens=tokens_out, retried=attempt == 1))
            if attempt == 1:
                raise StageError(stage, FailureKind.VALIDATION, "; ".join(errors))
            logger.warning("[RUNNER] stage=%s retrying: %s", stage, errors)
            history = result.to_input_list() if result else [{"role": "user", "content": prompt.input}]
            feedback = self._prompts.render_fragment(
                "validation_retry", errors="\n".join(f"- {e}" for e in errors))
            conversation = [*history, {"role": "user", "content": feedback}]
        raise AssertionError("unreachable")

    @staticmethod
    def _classify(stage: Stage, error: Exception) -> BaseException:
        for exc_types, kind, message in FAILURES:
            if isinstance(error, exc_types):
                return StageError(stage, kind, f"{message}: {error}")
        return error  # a bug, not a provider failure: let it surface as-is
