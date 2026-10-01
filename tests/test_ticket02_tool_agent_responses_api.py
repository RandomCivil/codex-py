import asyncio
from types import SimpleNamespace

from agent.execution import ExecutionAnswer, ToolAgentMode
from agent.configuration import ComponentProviderConfiguration, ProviderConfiguration
from agent.execution import create_execution_mode
from llm.llm import LLM


class ResponsesToolAgentModel(LLM):
    def __init__(self, events):
        self.events = events
        self.requests = []

    async def stream_events(self, input, *, instructions=None, tools=None):
        if callable(getattr(self, "_on_request", None)):
            self._on_request({"input": input, "instructions": instructions, "tools": tools})
        self.requests.append({"input": input, "instructions": instructions, "tools": tools})
        events = self.events.pop(0) if self.events and isinstance(self.events[0], list) else self.events
        for event in events:
            yield event


class EmptyRuntime:
    tools = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        pass


class Tool:
    name = "read_file"
    description = "Read one file."
    args_schema = {"type": "object", "properties": {"path": {"type": "string"}}}


class ToolRuntime(EmptyRuntime):
    tools = [Tool()]

    def __init__(self):
        self.calls = []

    async def invoke(self, call):
        self.calls.append(call)
        return "host result"


def test_tool_agent_no_tool_response_uses_streamed_responses_api():
    model = ResponsesToolAgentModel(
        [
            SimpleNamespace(type="response.output_text.delta", delta='BEGIN ANSWER\nTEXT="Done"\nEND ANSWER'),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(output_text='BEGIN ANSWER\nTEXT="Done"\nEND ANSWER'),
            ),
        ]
    )

    answer = asyncio.run(ToolAgentMode(model, EmptyRuntime()).run("Answer"))

    assert answer == ExecutionAnswer("Done", "completed")
    assert len(model.requests) == 1
    assert model.requests[0]["input"] == [{"role": "user", "content": "Answer"}]
    assert model.requests[0]["tools"] == []


def test_tool_agent_executes_first_native_function_call_and_host_renders_result():
    runtime = ToolRuntime()
    model = ResponsesToolAgentModel(
        [
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    output_text="",
                    output=[
                        SimpleNamespace(
                            type="function_call",
                            name="read_file",
                            arguments='{"path":"README.md"}',
                            call_id="call-1",
                        ),
                        SimpleNamespace(
                            type="function_call",
                            name="read_file",
                            arguments='{"path":"CONTEXT.md"}',
                            call_id="call-2",
                        ),
                    ],
                ),
            )
        ]
    )

    answer = asyncio.run(ToolAgentMode(model, runtime).run("Read README"))

    assert answer == ExecutionAnswer("host result", "completed")
    assert [
        {key: call[key] for key in ("name", "args", "id")}
        for call in runtime.calls
    ] == [{"name": "read_file", "args": {"path": "README.md"}, "id": "call-1"}]
    assert model.requests[0]["tools"] == [
        {
            "type": "function",
            "name": "read_file",
            "description": "Read one file.",
            "parameters": Tool.args_schema,
        }
    ]


def test_tool_agent_repairs_invalid_line_protocol_through_responses_stream():
    model = ResponsesToolAgentModel(
        [
            [
                SimpleNamespace(type="response.output_text.delta", delta="not protocol"),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(output_text="not protocol"),
                ),
            ],
            [
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        output_text='BEGIN ANSWER\nTEXT="Repaired"\nEND ANSWER'
                    ),
                )
            ],
        ]
    )

    answer = asyncio.run(ToolAgentMode(model, EmptyRuntime()).run("Answer"))

    assert answer == ExecutionAnswer("Repaired", "completed")
    assert len(model.requests) == 2
    assert "Validation error:" in model.requests[1]["instructions"]


def test_configured_tool_agent_uses_responses_llm():
    mode = create_execution_mode(
        "tool_agent",
        configuration=ComponentProviderConfiguration(
            planner=ProviderConfiguration("https://provider.test/v1", "key", "model"),
            executor=ProviderConfiguration("https://provider.test/v1", "key", "model"),
            task_analyzer=ProviderConfiguration("https://provider.test/v1", "key", "model"),
            tool_agent=ProviderConfiguration("https://provider.test/v1", "key", "tool-model"),
        ),
        tool_runtime=EmptyRuntime(),
    )

    assert isinstance(mode._model, LLM)
    assert mode._model._model_name == "tool-model"
    assert mode._owned_http_clients == (mode._model,)


def test_tool_agent_does_not_duplicate_provider_request_attribution():
    class Trace:
        def __init__(self):
            self.requests = []

        def llm_request(self, component, request, **_kwargs):
            self.requests.append((component, request))

    trace = Trace()
    model = ResponsesToolAgentModel(
        [SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(output_text='BEGIN ANSWER\nTEXT="Done"\nEND ANSWER'),
        )]
    )
    model._on_request = lambda request: trace.llm_request("tool_agent", request)
    mode = ToolAgentMode(model, EmptyRuntime(), trace=trace)
    asyncio.run(mode.run("Answer"))

    assert len(trace.requests) == 1
    assert trace.requests[0][0] == "tool_agent"
