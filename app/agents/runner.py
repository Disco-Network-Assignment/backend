"""Runs one stage agent and returns typed output plus timings and token usage.

One retry is allowed: when the output does not fit the schema, or the stage's business check
rejects it, the errors are quoted back to the model once. Provider and SDK failures are mapped
to a FailureKind so the API can stream a typed `failed` event."""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from agents import Agent, RunConfig, Runner
from agents.exceptions import (
    AgentsException,
    MaxTurnsExceeded,
    ModelBehaviorError,
    ModelRefusalError,
    ModelTimeoutError,
)
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
from pydantic import BaseModel

from app.config import Settings
from app.enums import ExecutionMode, FailureKind, Stage
from app.errors import StageError
from app.prompts.registry import PromptRegistry, RenderedPrompt
from app.schemas import StageMeta

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# exception -> (kind, message); order matters, first match wins
FAILURES: tuple[tuple[type[BaseException] | tuple[type[BaseException], ...], FailureKind, str], ...] = (
    (ModelRefusalError, FailureKind.REFUSAL, "the model declined this request"),
    ((ModelTimeoutError, APITimeoutError, TimeoutError), FailureKind.TIMEOUT, "the model call timed out"),
    (RateLimitError, FailureKind.RATE_LIMIT, "rate limited by the model provider; retry in a minute"),
    ((APIConnectionError, APIError), FailureKind.API, "model provider error"),
    (MaxTurnsExceeded, FailureKind.VALIDATION, "the agent did not finish in one turn"),
    (AgentsException, FailureKind.UNKNOWN, "agent runtime error"),
)


@dataclass(frozen=True)
class StageRun(Generic[T]):
    output: T
    meta: StageMeta


class StructuredRunner:
    def __init__(self, settings: Settings, prompts: PromptRegistry) -> None:
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
