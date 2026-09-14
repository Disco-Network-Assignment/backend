"""The SDK layer against a fake model: no network, real Agents SDK plumbing (structured-output
parsing, function tools reading the shared context, handoffs with typed input, replay on retry,
usage accounting)."""

import json

import pytest
from agents import Agent, RunContextWrapper, handoff
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from pydantic import BaseModel

from app.agents.context import RunContext
from app.agents.openai_agent import StructuredRunner
from app.agents.tools import fit_signals
from app.domain.fit_signals import SignalCalculator
from app.enums import FailureKind, Stage
from app.errors import StageError
from app.schemas import HandoffReason
from tests.helpers import create_sample_brief


class Reply(BaseModel):
    answer: str
    score: float


def message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(id="msg", role="assistant", status="completed", type="message",
                                 content=[ResponseOutputText(type="output_text", text=text, annotations=[])])


def tool_call(name: str, arguments: dict) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(id="fc", call_id="call_1", name=name, type="function_call",
                                    arguments=json.dumps(arguments), status="completed")


class FakeModel(Model):
    """Returns canned output items in order and records every input it was given."""

    def __init__(self, replies: list) -> None:
        self._replies = list(replies)
        self.inputs: list = []

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema,
                           handoffs, tracing, *, previous_response_id, conversation_id, prompt):
        self.inputs.append(input)
        return ModelResponse(output=[self._replies.pop(0)], response_id=None,
                             usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15))

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError
        yield  # pragma: no cover


VALID = message(json.dumps({"answer": "ok", "score": 0.9}))
OTHER = message(json.dumps({"answer": "meh", "score": 0.1}))


@pytest.fixture
def ctx(catalog):
    return RunContext(catalog=catalog, signals=SignalCalculator(catalog),
                      description="We sell premium dog food.", brief=create_sample_brief())


@pytest.fixture
def runner(test_settings, prompts):
    return StructuredRunner(test_settings, prompts)


def make_agent(model: FakeModel, **kwargs) -> Agent[RunContext]:
    return Agent[RunContext](name="specialist", instructions="Answer.", model=model, output_type=Reply, **kwargs)


class TestStructuredRunner:
    async def test_parses_structured_output_and_records_usage(self, runner, ctx):
        model = FakeModel([VALID])
        run = await runner.run(Stage.INTAKE, make_agent(model), "Question?", (Reply,), ctx, prompt_version="7")
        assert run.output == Reply(answer="ok", score=0.9)
        assert run.meta.input_tokens == 10 and run.meta.output_tokens == 5
        assert run.meta.prompt_version == "7" and run.meta.agent == "specialist" and not run.meta.retried
        # the SDK normalises the string input into one user message before the model sees it
        assert model.inputs == [[{"content": "Question?", "role": "user"}]]

    async def test_retries_once_with_errors_when_the_check_rejects(self, runner, ctx):
        model = FakeModel([OTHER, VALID])
        check = lambda reply: [] if reply.answer == "ok" else ["answer must be 'ok'"]  # noqa: E731
        run = await runner.run(Stage.MATCH, make_agent(model), "Q", (Reply,), ctx, prompt_version="1", check=check)
        assert run.output.answer == "ok" and run.meta.retried and run.meta.input_tokens == 20
        assert "answer must be 'ok'" in json.dumps(model.inputs[1])

    async def test_schema_failure_is_retried_then_raised(self, runner, ctx):
        model = FakeModel([message("not json"), message("still not json")])
        with pytest.raises(StageError) as info:
            await runner.run(Stage.INTAKE, make_agent(model), "Q", (Reply,), ctx, prompt_version="1")
        assert info.value.kind is FailureKind.VALIDATION and len(model.inputs) == 2


class TestToolsAndHandoffs:
    async def test_tool_reads_the_shared_context_and_feeds_the_model(self, runner, ctx):
        model = FakeModel([tool_call("fit_signals", {"publisher_id": "pub_007"}), VALID])
        agent = make_agent(model, tools=[fit_signals])
        run = await runner.run(Stage.MATCH, agent, "Score Pawline", (Reply,), ctx, prompt_version="1")
        assert run.meta.tool_calls == 1 and run.output.answer == "ok"
        assert "pub_007" in ctx.fit_signals  # the tool cached its computation in the context
        assert "function_call_output" in json.dumps(model.inputs[1])  # and the model saw the result

    async def test_handoff_switches_agent_and_records_the_reason(self, runner, ctx):
        model = FakeModel([tool_call("transfer_to_specialist", {"reason": "a real business"}), VALID])
        specialist = make_agent(model)

        async def on_handoff(wrapper: RunContextWrapper[RunContext], data: HandoffReason) -> None:
            wrapper.context.handoff_reason = data.reason

        triage = Agent[RunContext](name="triage", instructions="Route.", model=model,
                                   handoffs=[handoff(specialist, tool_name_override="transfer_to_specialist",
                                                     on_handoff=on_handoff, input_type=HandoffReason)])
        run = await runner.run(Stage.INTAKE, triage, "We sell dog food", (Reply,), ctx, prompt_version="1")
        assert run.output.answer == "ok"
        assert run.meta.handoffs == 1 and run.meta.agent == "specialist"
        assert ctx.handoff_reason == "a real business"

    async def test_wrong_final_type_is_a_validation_error(self, runner, ctx):
        model = FakeModel([message("I would rather chat"), message("still chatting")])
        triage = Agent[RunContext](name="triage", instructions="Route.", model=model)
        with pytest.raises(StageError) as info:
            await runner.run(Stage.INTAKE, triage, "hello", (Reply,), ctx, prompt_version="1")
        assert info.value.kind is FailureKind.VALIDATION and "expected Reply" in info.value.message


class TestFailureClassification:
    def test_no_credits_is_not_reported_as_a_transient_rate_limit(self):
        import httpx
        from openai import RateLimitError

        response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
        error = RateLimitError("Error code: 429 - {'error': {'type': 'insufficient_quota'}}", response=response, body=None)
        classified = StructuredRunner._classify(Stage.INTAKE, error)
        assert isinstance(classified, StageError) and classified.kind is FailureKind.API
        assert "no credits" in classified.message

    def test_a_real_rate_limit_still_says_retry(self):
        import httpx
        from openai import RateLimitError

        response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
        error = RateLimitError("Error code: 429 - {'error': {'type': 'rate_limit_exceeded'}}", response=response, body=None)
        classified = StructuredRunner._classify(Stage.INTAKE, error)
        assert isinstance(classified, StageError) and classified.kind is FailureKind.RATE_LIMIT

