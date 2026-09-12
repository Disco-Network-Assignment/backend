"""Runs one stage agent and returns validated, typed output with timings and token usage.

Owns the single retry the pipeline allows: when the SDK cannot parse the model's output into the
stage schema, or the stage's business check rejects it (a publisher missing from the matcher's
list, say), the errors are quoted back to the model once. Every SDK or provider failure is
classified into a FailureKind so the API can stream a typed `failed` event."""

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
Check = Callable[[T], list[str]]

WORKFLOW_NAME = "disco-campaign-brain"


@dataclass(frozen=True)
class StageRun(Generic[T]):
    output: T
    meta: StageMeta


class StructuredRunner:
    def __init__(self, settings: Settings, prompts: PromptRegistry) -> None:
        self._settings = settings
        self._prompts = prompts

    async def run(self, stage: Stage, agent: Agent, prompt: RenderedPrompt,
                  output_type: type[T], check: Check[T] | None = None) -> StageRun[T]:
        started = time.perf_counter()
        run_config = RunConfig(workflow_name=WORKFLOW_NAME,
                               tracing_disabled=not self._settings.tracing_enabled,
                               trace_metadata={"stage": str(stage)})
        conversation: str | list = prompt.input
        input_tokens = output_tokens = 0
        retried = False
        for attempt in (0, 1):
            result = None
            try:
                result = await asyncio.wait_for(
                    Runner.run(agent, conversation, max_turns=1, run_config=run_config),
                    timeout=self._settings.llm_timeout_s + 5,
                )
                errors: list[str] = []
            except ModelBehaviorError as e:  # output did not fit the schema
                errors = [f"output did not match the required schema: {e}"]
            except ModelRefusalError as e:
                raise StageError(stage, FailureKind.REFUSAL, f"the model declined: {e}") from e
            except (ModelTimeoutError, APITimeoutError, TimeoutError) as e:
                raise StageError(stage, FailureKind.TIMEOUT, "the model call timed out") from e
            except RateLimitError as e:
                raise StageError(stage, FailureKind.RATE_LIMIT,
                                 "rate limited by the model provider; retry in a minute") from e
            except (APIConnectionError, APIError) as e:
                raise StageError(stage, FailureKind.API, f"model provider error: {e}") from e
            except MaxTurnsExceeded as e:
                raise StageError(stage, FailureKind.VALIDATION,
                                 "the agent did not finish in one turn") from e
            except AgentsException as e:
                raise StageError(stage, FailureKind.UNKNOWN, f"agent runtime error: {e}") from e

            if result is not None:
                for response in result.raw_responses:
                    input_tokens += response.usage.input_tokens
                    output_tokens += response.usage.output_tokens
                output = result.final_output_as(output_type)
                errors = check(output) if check else []
                if not errors:
                    return StageRun(output=output, meta=StageMeta(
                        stage=stage, ms=_elapsed_ms(started), mode=ExecutionMode.LLM,
                        model=agent.model if isinstance(agent.model, str) else None,
                        reasoning_effort=_effort(agent), prompt_version=prompt.version,
                        input_tokens=input_tokens, output_tokens=output_tokens, retried=retried))

            if attempt == 1:
                raise StageError(stage, FailureKind.VALIDATION, "; ".join(errors))
            logger.warning("[RUNNER] stage=%s retrying after validation errors: %s", stage, errors)
            retried = True
            conversation = self._retry_conversation(prompt, result, errors)

        raise AssertionError("unreachable")

    def _retry_conversation(self, prompt: RenderedPrompt, result, errors: list[str]) -> list:
        """The retry sees its own previous answer (when the SDK could keep it) plus the errors."""
        history = result.to_input_list() if result is not None else [
            {"role": "user", "content": prompt.input}]
        feedback = self._prompts.render_fragment("validation_retry", errors="\n".join(f"- {e}" for e in errors))
        return [*history, {"role": "user", "content": feedback}]


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _effort(agent: Agent) -> str | None:
    reasoning = agent.model_settings.reasoning if agent.model_settings else None
    return getattr(reasoning, "effort", None) if reasoning else None
