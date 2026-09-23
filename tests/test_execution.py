import asyncio
from io import StringIO

from langchain_core.messages import AIMessage

from agent import PlanningValidationError
from agent.configuration import ComponentProviderConfiguration, ProviderConfiguration
from agent.execution import DirectMode, ExecutionAnswer, PlanExecuteMode, ReactMode, ToolAgentMode, create_execution_mode
from agent.trace import RunTrace


class TextModel:
    def __init__(self, *responses): self.responses = iter(responses); self.requests = []
    async def stream_text(self, input, **kwargs):
        self.requests.append((input, kwargs))
        yield next(self.responses)


class ToolModel:
    def __init__(self, *responses): self.responses = iter(responses); self.requests = []; self.tools = None
    def bind_tools(self, tools): self.tools = tools; return self
    async def ainvoke(self, messages):
        self.requests.append(list(messages))
        return next(self.responses)


class Runtime:
    def __init__(self): self.tools = []; self.calls = []
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def invoke(self, call): self.calls.append(call); return {"value": "done"}


def test_plan_execute_reports_a_safe_actionable_provider_status():
    class ProviderFailure(RuntimeError):
        status_code = 402

    class Durable:
        async def run(self, *_args, **_kwargs):
            raise ProviderFailure("provider payload that must not reach the caller")

    answer = asyncio.run(PlanExecuteMode(Durable(), run_id="run-1").run("Inspect components"))

    assert answer == ExecutionAnswer(
        None,
        "failed",
        "plan-execute execution failed: provider returned HTTP 402 (Insufficient Balance)",
    )


def test_plan_execute_reports_the_safe_planning_validation_error():
    class Durable:
        async def run(self, *_args, **_kwargs):
            raise PlanningValidationError("invalid line or text outside a block")

    answer = asyncio.run(PlanExecuteMode(Durable(), run_id="run-1").run("Inspect components"))

    assert answer == ExecutionAnswer(
        None,
        "failed",
        "planning failed: invalid line or text outside a block",
    )


def test_direct_mode_decodes_answer_and_prompts_for_line_protocol():
    model = TextModel('BEGIN ANSWER\nTEXT="Ready"\nEND ANSWER')
    answer = asyncio.run(DirectMode(model).run("Prepare"))
    assert answer == ExecutionAnswer("Ready", "completed")
    instructions = model.requests[0][1]["instructions"]
    assert "BEGIN ANSWER" in instructions
    assert "TEXT=<JSON string literal>" in instructions
    assert 'TEXT="A concise answer"' in instructions
    assert '`{"text":"..."}`' in instructions
    assert "text_format" not in model.requests[0][1]


def test_direct_mode_rejects_invalid_answer():
    model = TextModel("Ready", 'BEGIN ANSWER\nTEXT="Ready"\nEND ANSWER')

    assert asyncio.run(DirectMode(model).run("Prepare")) == ExecutionAnswer("Ready", "completed")
    assert len(model.requests) == 2
    assert "Validation error: invalid line or text outside a block" in model.requests[1][1]["instructions"]


def test_direct_mode_rejects_a_second_invalid_answer():
    assert asyncio.run(DirectMode(TextModel("Ready", "Still ready")).run("Prepare")).status == "failed"


def test_tool_agent_prompts_for_answer_and_decodes_no_tool_response():
    model = ToolModel(AIMessage(content='BEGIN ANSWER\nTEXT="No tool needed"\nEND ANSWER'))
    answer = asyncio.run(ToolAgentMode(model, Runtime()).run("Answer"))
    assert answer == ExecutionAnswer("No tool needed", "completed")
    prompt = model.requests[0][0].content
    assert "BEGIN ANSWER" in prompt
    assert "TEXT=<JSON string literal>" in prompt
    assert 'TEXT="A concise answer"' in prompt


def test_tool_agent_retries_invalid_no_tool_response_with_validation_feedback():
    model = ToolModel(
        AIMessage(content="No tool needed"),
        AIMessage(content='BEGIN ANSWER\nTEXT="No tool needed"\nEND ANSWER'),
    )

    answer = asyncio.run(ToolAgentMode(model, Runtime()).run("Answer"))

    assert answer == ExecutionAnswer("No tool needed", "completed")
    assert len(model.requests) == 2
    prompt = model.requests[1][0].content
    assert "Validation error: invalid line or text outside a block" in prompt
    assert 'Previous response (JSON-encoded): "No tool needed"' in prompt


def test_tool_agent_executes_a_tool_call_with_accompanying_text():
    model = ToolModel(AIMessage(content='BEGIN ANSWER\nTEXT="bad"\nEND ANSWER', tool_calls=[{"name": "x", "args": {}, "id": "1"}]))
    runtime = Runtime()
    answer = asyncio.run(ToolAgentMode(model, runtime).run("Answer"))
    assert answer == ExecutionAnswer('{"value": "done"}', "completed")
    assert [{key: call[key] for key in ("name", "args", "id")} for call in runtime.calls] == [
        {"name": "x", "args": {}, "id": "1"}
    ]


def test_react_returns_freeform_content_when_no_tool_call_is_present():
    model = ToolModel(AIMessage(content="Done\n\n- inspected the components"))

    answer = asyncio.run(ReactMode(model, Runtime()).run("Complete"))

    assert answer == ExecutionAnswer("Done\n\n- inspected the components", "completed")
    assert len(model.requests) == 1
    assert "BEGIN NO_TOOL" not in model.requests[0][0].content
    assert "GOAL_COMPLETION" not in model.requests[0][0].content
    assert "requested outcome as the fixed" in model.requests[0][0].content
    assert "completion standard" in model.requests[0][0].content
    assert "Before every tool call, check all three conditions" in model.requests[0][0].content
    assert "Do not repeat an" in model.requests[0][0].content
    assert "equivalent inspection" in model.requests[0][0].content


def test_react_rejects_an_empty_final_response_without_tool_calls():
    model = ToolModel(AIMessage(content=""))

    answer = asyncio.run(ReactMode(model, Runtime()).run("Complete"))

    assert answer == ExecutionAnswer(
        None,
        "failed",
        error="react model returned an empty final response",
    )


def test_react_executes_tool_calls_with_accompanying_text():
    model = ToolModel(
        AIMessage(content="I'll inspect the source tree.", tool_calls=[{"name": "list_dir", "args": {}, "id": "1"}]),
        AIMessage(content="The source tree is inspected."),
    )
    runtime = Runtime()
    answer = asyncio.run(ReactMode(model, runtime).run("Inspect"))
    assert answer == ExecutionAnswer("The source tree is inspected.", "completed")
    assert len(model.requests) == 2
    assert all(
        "Before every tool call, check all three conditions" in request[0].content
        for request in model.requests
    )
    assert [{key: call[key] for key in ("name", "args", "id")} for call in runtime.calls] == [
        {"name": "list_dir", "args": {}, "id": "1"}
    ]


def test_react_logs_the_original_exception_before_returning_safe_error():
    class FailingModel(ToolModel):
        async def ainvoke(self, messages):
            raise RuntimeError("provider exploded")

    output = StringIO()
    trace = RunTrace(output, level="error", run_id="run-123")

    answer = asyncio.run(ReactMode(FailingModel(), Runtime(), trace=trace).run("Inspect"))

    assert answer == ExecutionAnswer(None, "failed", error="react execution failed")
    assert "[run-123] [execution error] component=react" in output.getvalue()
    assert "RuntimeError: provider exploded" in output.getvalue()


def test_react_factory_owns_and_closes_a_fresh_http_client_per_asyncio_run(monkeypatch):
    """Interactive chat starts a new event loop for each submitted turn."""
    clients = []

    class ChatModel:
        def __init__(self, **kwargs):
            clients.append(kwargs["http_async_client"])

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            return AIMessage(content="Done")

    monkeypatch.setattr("agent.execution.ChatOpenAI", ChatModel)
    provider = ProviderConfiguration("https://provider.test/v1", "key", "model")
    configuration = ComponentProviderConfiguration(provider, provider, provider, react=provider)

    async def submit_turn():
        mode = create_execution_mode("react", configuration=configuration)
        assert await mode.run("Complete") == ExecutionAnswer("Done", "completed")

    asyncio.run(submit_turn())
    asyncio.run(submit_turn())

    assert len(clients) == 2
    assert clients[0] is not clients[1]
    assert all(client.is_closed for client in clients)
