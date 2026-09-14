"""The OpenAI Agents SDK layer: build an agent, run it, get typed output.

`AgentFactory` decides HOW an agent is configured (model per stage, reasoning effort, tools,
handoffs, output schema, the shared client). `StructuredRunner` makes the call with the shared
local context and optional session memory, lets the SDK loop over tool calls and handoffs, and
owns the single retry: when the final output is not one of the expected types, or the stage's
business check rejects it, the errors are quoted back to the model once. Provider and SDK
failures are mapped to a FailureKind so the API can stream a typed `failed` event."""

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from agents import (
    Agent,
    Handoff,
    HandoffCallItem,
    ModelSettings,
    RunConfig,
    Runner,
    Session,
    SessionSettings,
    Tool,
    ToolCallItem,
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

from app.agents.context import RunContext
from app.enums import FailureKind, Stage
from app.errors import StageError
from app.prompts.loader import PromptLoader
from app.schemas import StageMeta
from app.settings import Settings

logger = logging.getLogger(__name__)

# exception -> (kind, message); first match wins
FAILURES: tuple[tuple[type[BaseException] | tuple[type[BaseException], ...], FailureKind, str], ...] = (
    (ModelRefusalError, FailureKind.REFUSAL, "the model declined this request"),
    ((ModelTimeoutError, APITimeoutError, TimeoutError), FailureKind.TIMEOUT, "the model call timed out"),
    (RateLimitError, FailureKind.RATE_LIMIT, "rate limited by the model provider; retry in a minute"),
    ((APIConnectionError, APIError), FailureKind.API, "model provider error"),
    (MaxTurnsExceeded, FailureKind.VALIDATION, "the agent did not finish within the turn limit"),
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

    def build(self, stage: Stage, name: str, instructions: str, output_type: type[BaseModel] | None = None,
              tools: Sequence[Tool] = (), handoffs: Sequence[Agent | Handoff] = ()) -> Agent[RunContext]:
        self._ensure_client()
        return Agent[RunContext](
            name=name,
            instructions=instructions,
            model=self.model_for(stage),
            model_settings=ModelSettings(
                reasoning=Reasoning(effort=self.effort_for(stage)),
                verbosity="low",
                max_tokens=self._settings.llm_max_output_tokens,
                timeout=self._settings.llm_timeout_s,
            ),
            tools=list(tools),
            handoffs=list(handoffs),
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
class StageRun:
    output: Any
    meta: StageMeta


class StructuredRunner:
    def __init__(self, settings: Settings, prompts: PromptLoader) -> None:
        self._settings = settings
        self._prompts = prompts

    async def run(self, stage: Stage, agent: Agent[RunContext], user_input: str, expect: tuple[type, ...],
                  context: RunContext, *, prompt_version: str, check: Callable[[Any], list[str]] | None = None,
                  session: Session | None = None, max_turns: int = 6) -> StageRun:
        started = time.perf_counter()
        config = RunConfig(workflow_name="disco-campaign-brain",
                           tracing_disabled=not self._settings.tracing_enabled,
                           session_settings=SessionSettings(limit=self._settings.session_history_limit))
        conversation: str | list = user_input
        tokens_in = tokens_out = tool_calls = handoffs = 0
        for attempt in (0, 1):
            result = None
            try:
                result = await asyncio.wait_for(
                    Runner.run(agent, conversation, context=context, session=session if attempt == 0 else None,
                               max_turns=max_turns, run_config=config),
                    timeout=self._settings.llm_timeout_s * max_turns)
            except ModelBehaviorError as e:  # the output did not fit the schema
                errors = [f"output did not match the required schema: {e}"]
            except Exception as e:
                raise self._classify(stage, e) from e
            else:
                for response in result.raw_responses:
                    tokens_in += response.usage.input_tokens
                    tokens_out += response.usage.output_tokens
                tool_calls += sum(isinstance(item, ToolCallItem) for item in result.new_items)
                handoffs += sum(isinstance(item, HandoffCallItem) for item in result.new_items)
                output = result.final_output
                if not isinstance(output, expect):
                    errors = [f"expected {' or '.join(t.__name__ for t in expect)}, got "
                              f"{type(output).__name__}; finish with the structured output or hand off"]
                else:
                    errors = check(output) if check else []
                if not errors:
                    return StageRun(output, StageMeta(
                        stage=stage, ms=round((time.perf_counter() - started) * 1000),
                        agent=result.last_agent.name,
                        model=agent.model if isinstance(agent.model, str) else None,
                        reasoning_effort=getattr(agent.model_settings.reasoning, "effort", None),
                        prompt_version=prompt_version, input_tokens=tokens_in, output_tokens=tokens_out,
                        tool_calls=tool_calls, handoffs=handoffs, retried=attempt == 1))
            if attempt == 1:
                raise StageError(stage, FailureKind.VALIDATION, "; ".join(errors))
            logger.warning("[RUNNER] stage=%s retrying: %s", stage, errors)
            history = result.to_input_list() if result else [{"role": "user", "content": user_input}]
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
