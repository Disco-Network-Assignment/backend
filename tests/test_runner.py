"""The StructuredRunner against a fake SDK model: no network, real Agents SDK plumbing
(structured-output parsing, conversation replay on retry, usage accounting)."""

import json

import pytest
from agents import Agent
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import ResponseOutputMessage, ResponseOutputText
from pydantic import BaseModel

from app.agents.runner import StructuredRunner
from app.enums import FailureKind, Stage
from app.errors import StageError
from app.prompts.registry import RenderedPrompt


class Reply(BaseModel):
    answer: str
    score: float


class FakeModel(Model):
    """Returns canned JSON texts in order and records every input it was given."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.inputs: list = []

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema,
                           handoffs, tracing, *, previous_response_id, conversation_id, prompt):
        self.inputs.append(input)
        text = self._replies.pop(0)
        message = ResponseOutputMessage(
            id=f"msg_{len(self.inputs)}", role="assistant", status="completed", type="message",
            content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
        )
        return ModelResponse(output=[message], response_id=None,
                             usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15))

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError
        yield  # pragma: no cover


PROMPT = RenderedPrompt(name="test", version="7", instructions="Answer.", input="Question?")
VALID = json.dumps({"answer": "ok", "score": 0.9})
OTHER = json.dumps({"answer": "meh", "score": 0.1})


def make_runner(test_settings, prompts):
    return StructuredRunner(test_settings, prompts)


def make_agent(model: FakeModel) -> Agent:
    return Agent(name="test", instructions="Answer.", model=model, output_type=Reply)


class TestStructuredRunner:
    async def test_parses_structured_output_and_records_usage(self, test_settings, prompts):
        model = FakeModel([VALID])
        run = await make_runner(test_settings, prompts).run(Stage.INTAKE, make_agent(model), PROMPT, Reply)
        assert run.output == Reply(answer="ok", score=0.9)
        assert run.meta.input_tokens == 10 and run.meta.output_tokens == 5
        assert run.meta.prompt_version == "7" and not run.meta.retried and not run.meta.cached
        # the SDK normalises the string input into one user message before the model sees it
        assert model.inputs == [[{"content": "Question?", "role": "user"}]]

    async def test_retries_once_with_errors_when_the_check_rejects(self, test_settings, prompts):
        model = FakeModel([OTHER, VALID])
        check = lambda reply: [] if reply.answer == "ok" else ["answer must be 'ok'"]  # noqa: E731
        run = await make_runner(test_settings, prompts).run(Stage.MATCH, make_agent(model), PROMPT, Reply, check)
        assert run.output.answer == "ok" and run.meta.retried
        assert run.meta.input_tokens == 20
        retry_input = model.inputs[1]
        assert isinstance(retry_input, list)
        assert "answer must be 'ok'" in json.dumps(retry_input)

    async def test_schema_failure_is_retried_then_raised(self, test_settings, prompts):
        model = FakeModel(["not json", "still not json"])
        with pytest.raises(StageError) as info:
            await make_runner(test_settings, prompts).run(Stage.INTAKE, make_agent(model), PROMPT, Reply)
        assert info.value.kind is FailureKind.VALIDATION and info.value.stage is Stage.INTAKE
        assert len(model.inputs) == 2

    async def test_schema_failure_then_success(self, test_settings, prompts):
        model = FakeModel(["not json", VALID])
        run = await make_runner(test_settings, prompts).run(Stage.INTAKE, make_agent(model), PROMPT, Reply)
        assert run.output.answer == "ok" and run.meta.retried
