"""The OpenAI Agents SDK layer: build an agent, run it, get typed output.

`AgentFactory` decides HOW an agent is configured: which model and reasoning effort a stage
gets, its tools, handoffs and output schema, and the one shared OpenAI client.

`StructuredRunner` makes the call. It passes the shared local context and optional session
memory, lets the SDK loop over tool calls and handoffs, and owns the single retry: when the
final output is not one of the expected types, or the stage's business check rejects it, the
errors are quoted back to the model once. Provider and SDK failures are mapped to a
FailureKind so the API can stream a typed `failed` event.
"""

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

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKFLOW_NAME = "disco-campaign-brain"   # groups the runs in the OpenAI traces dashboard
DEFAULT_MAX_TURNS = 6                    # tool calls + handoffs one stage may take
MAX_ATTEMPTS = 2                         # one run plus one retry with the errors quoted back

NO_CREDITS_MESSAGE = ("the OpenAI account behind OPENAI_API_KEY has no credits remaining; add credits at "
                      "platform.openai.com/settings/organization/billing and retry")

# provider / SDK exception -> what the API reports; first match wins
FAILURES = (
    (ModelRefusalError, FailureKind.REFUSAL, "the model declined this request"),
    ((ModelTimeoutError, APITimeoutError, TimeoutError), FailureKind.TIMEOUT, "the model call timed out"),
    (RateLimitError, FailureKind.RATE_LIMIT, "rate limited by the model provider; retry in a minute"),
    ((APIConnectionError, APIError), FailureKind.API, "model provider error"),
    (MaxTurnsExceeded, FailureKind.VALIDATION, "the agent did not finish within the turn limit"),
    (AgentsException, FailureKind.UNKNOWN, "agent runtime error"),
)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


class AgentFactory:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None

    def model_for(self, stage: Stage) -> str:
        """The matcher is the judgement-heavy stage, so it gets the strongest model."""
        if stage is Stage.MATCH:
            return self.settings.openai_match_model
        return self.settings.openai_model

    def effort_for(self, stage: Stage) -> str:
        if stage is Stage.MATCH:
            return self.settings.match_reasoning_effort
        return self.settings.reasoning_effort

    def build(self, stage: Stage, name: str, instructions: str, output_type: type[BaseModel] | None = None,
              tools: Sequence[Tool] = (), handoffs: Sequence[Agent | Handoff] = ()) -> Agent[RunContext]:
        self._ensure_client()
        model_settings = ModelSettings(
            reasoning=Reasoning(effort=self.effort_for(stage)),
            verbosity="low",
            max_tokens=self.settings.llm_max_output_tokens,
            timeout=self.settings.llm_timeout_s,
        )
        return Agent[RunContext](
            name=name,
            instructions=instructions,
            model=self.model_for(stage),
            model_settings=model_settings,
            tools=list(tools),
            handoffs=list(handoffs),
            output_type=output_type,
        )

    def _ensure_client(self):
        """One client per process so every run shares the connection pool."""
        if self._client is not None:
            return
        self._client = AsyncOpenAI(api_key=self.settings.openai_api_key or None)
        set_default_openai_client(self._client, use_for_tracing=self.settings.tracing_enabled)
        if not self.settings.tracing_enabled:
            set_tracing_disabled(True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageRun:
    output: Any        # the typed object the stage produced
    meta: StageMeta    # what it cost and which agent finished, for the trace


class StructuredRunner:
    def __init__(self, settings: Settings, prompts: PromptLoader):
        self.settings = settings
        self.prompts = prompts

    async def run(self, stage: Stage, agent: Agent[RunContext], user_input: str, expect: tuple[type, ...],
                  context: RunContext, *, prompt_version: str, check: Callable[[Any], list[str]] | None = None,
                  session: Session | None = None, max_turns: int = DEFAULT_MAX_TURNS) -> StageRun:
        """
        Run `agent` on `user_input` and return its final output, which must be an instance of
        one of the `expect` types (a handoff chain may end at a different agent, so more than
        one type can be acceptable). `check` is the stage's business validation: it returns a
        list of problems, and an empty list means the output is accepted.

        On a schema failure, a wrong type, or check errors, the model gets one more attempt
        with the conversation so far plus the errors quoted back.
        """
        started = time.perf_counter()
        run_config = RunConfig(
            workflow_name=WORKFLOW_NAME,
            tracing_disabled=not self.settings.tracing_enabled,
            session_settings=SessionSettings(limit=self.settings.session_history_limit),
        )
        # the SDK accepts either a plain string or a list of conversation items
        conversation = user_input

        # totals across both attempts, so the trace shows what the stage really cost
        input_tokens = 0
        output_tokens = 0
        tool_calls = 0
        handoffs = 0

        for attempt in range(1, MAX_ATTEMPTS + 1):
            is_retry = attempt > 1
            result = None
            errors = []

            # --- run the agent ---
            try:
                result = await asyncio.wait_for(
                    Runner.run(
                        agent, conversation, context=context,
                        # session memory only on the first attempt: the retry already carries
                        # the full conversation, and must not be appended to the session twice
                        session=None if is_retry else session,
                        max_turns=max_turns, run_config=run_config,
                    ),
                    timeout=self.settings.llm_timeout_s * max_turns,
                )
            except ModelBehaviorError as e:
                # the model's answer did not fit the output schema
                errors.append(f"output did not match the required schema: {e}")
            except Exception as e:
                raise self._classify(stage, e) from e

            # --- account for what happened ---
            if result is not None:
                for response in result.raw_responses:
                    input_tokens += response.usage.input_tokens
                    output_tokens += response.usage.output_tokens
                for item in result.new_items:
                    if isinstance(item, ToolCallItem):
                        tool_calls += 1
                    if isinstance(item, HandoffCallItem):
                        handoffs += 1

                # --- validate the final output ---
                output = result.final_output
                if not isinstance(output, expect):
                    expected_names = " or ".join(t.__name__ for t in expect)
                    errors.append(f"expected {expected_names}, got {type(output).__name__}; "
                                  f"finish with the structured output or hand off")
                elif check is not None:
                    errors = check(output)

                if not errors:
                    elapsed_ms = round((time.perf_counter() - started) * 1000)
                    meta = StageMeta(
                        stage=stage, ms=elapsed_ms,
                        agent=result.last_agent.name,
                        model=agent.model if isinstance(agent.model, str) else None,
                        reasoning_effort=getattr(agent.model_settings.reasoning, "effort", None),
                        prompt_version=prompt_version,
                        input_tokens=input_tokens, output_tokens=output_tokens,
                        tool_calls=tool_calls, handoffs=handoffs, retried=is_retry,
                    )
                    return StageRun(output, meta)

            # --- out of attempts ---
            if attempt == MAX_ATTEMPTS:
                raise StageError(stage, FailureKind.VALIDATION, "; ".join(errors))

            # --- prepare the retry: the conversation so far, then the errors as a user message ---
            log.warning("[RUNNER] stage=%s retrying: %s", stage, errors)
            if result is not None:
                history = result.to_input_list()
            else:
                history = [{"role": "user", "content": user_input}]
            error_lines = "\n".join(f"- {error}" for error in errors)
            feedback = self.prompts.render_fragment("validation_retry", errors=error_lines)
            conversation = history + [{"role": "user", "content": feedback}]

        raise AssertionError("unreachable: the loop returns or raises")

    @staticmethod
    def _classify(stage: Stage, error: Exception) -> BaseException:
        """Map a provider or SDK failure to a StageError; anything else is a bug and surfaces as-is."""
        # OpenAI reports an empty credit balance as a 429 too, but waiting will not fix it
        if isinstance(error, RateLimitError) and "insufficient_quota" in str(error):
            return StageError(stage, FailureKind.API, NO_CREDITS_MESSAGE)
        for exception_types, kind, message in FAILURES:
            if isinstance(error, exception_types):
                return StageError(stage, kind, f"{message}: {error}")
        return error
