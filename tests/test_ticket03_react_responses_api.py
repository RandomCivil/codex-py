import asyncio
from io import StringIO
from types import SimpleNamespace

from agent.execution import ExecutionAnswer, ReactMode
from llm import LLM
from agent.trace import RunTrace
from tests.helpers import Runtime


def _event(event_type, **values):
    return SimpleNamespace(type=event_type, **values)


class ResponsesModel(LLM):
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    async def stream_events(self, input, *, instructions=None, tools=None):
        self.requests.append({"input": input, "instructions": instructions, "tools": tools})
        for event in next(self.responses):
            if callable(getattr(self, "_on_event", None)):
                self._on_event(event)
            yield event


def _text_response(text, output_tokens=2):
    return [
        _event("response.output_text.delta", delta=text),
        _event(
            "response.completed",
            response=SimpleNamespace(
                output_text=text,
                output=[],
                usage={"input_tokens": 3, "output_tokens": output_tokens},
            ),
        ),
    ]


def test_react_uses_streamed_responses_native_tool_continuation_and_attribution():
    model = ResponsesModel(
        [
            _event(
                "response.output_item.done",
                item=SimpleNamespace(
                    type="function_call",
                    name="inspect",
                    arguments='{"path":"README.md"}',
                    call_id="call-1",
                ),
            ),
            _event(
                "response.completed",
                response=SimpleNamespace(
                    output_text="",
                    output=[
                        SimpleNamespace(
                            type="function_call",
                            name="inspect",
                            arguments='{"path":"README.md"}',
                            call_id="call-1",
                        )
                    ],
                    usage={"input_tokens": 5, "output_tokens": 1},
                ),
            ),
        ],
        _text_response(
            'BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="Inspected"\nEND REACT_DECISION'
        ),
        _text_response(
            'BEGIN GOAL_JUDGMENT\nCOMPLETED=true\nEVIDENCE="The inspection is reported"\nEND GOAL_JUDGMENT'
        ),
    )
    trace_output = StringIO()
    trace = RunTrace(trace_output, level="error")
    runtime = Runtime(result={"path": "README.md", "present": True})
    runtime.tools = [SimpleNamespace(name="inspect", description="Inspect a path", args_schema={"type": "object"})]

    answer = asyncio.run(ReactMode(model, runtime, trace=trace).run("Inspect README"))

    assert answer == ExecutionAnswer("Inspected", "completed")
    assert [
        {key: call[key] for key in ("name", "args", "id")}
        for call in runtime.calls
    ] == [{"name": "inspect", "args": {"path": "README.md"}, "id": "call-1"}]
    assert len(model.requests) == 3
    assert model.requests[0]["tools"]
    assert model.requests[1]["tools"]
    assert model.requests[2]["tools"] == []
    continuation = model.requests[1]["input"]
    assert {item.get("type") for item in continuation} >= {"function_call", "function_call_output"}
    assert any(item.get("call_id") == "call-1" for item in continuation)
    trace_text = trace_output.getvalue()
    assert "component=react" in trace_text
    assert "component=completion_judge" in trace_text


def test_react_completion_judge_repair_is_streamed_tool_free_and_independently_traced():
    model = ResponsesModel(
        _text_response(
            'BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="Done"\nEND REACT_DECISION'
        ),
        _text_response("not a judgment"),
        _text_response(
            'BEGIN GOAL_JUDGMENT\nCOMPLETED=true\nEVIDENCE="Verified"\nEND GOAL_JUDGMENT'
        ),
    )
    trace_output = StringIO()
    trace = RunTrace(trace_output, level="error")

    answer = asyncio.run(ReactMode(model, Runtime(), trace=trace).run("Finish"))

    assert answer == ExecutionAnswer("Done", "completed")
    assert len(model.requests) == 3
    assert model.requests[1]["tools"] == []
    assert model.requests[2]["tools"] == []
    assert sum(
        "[llm usage] component=completion_judge" in line
        for line in trace_output.getvalue().splitlines()
    ) == 2
    assert sum(
        "[llm timing] component=completion_judge" in line
        for line in trace_output.getvalue().splitlines()
    ) == 2


def test_react_does_not_duplicate_adapter_events_or_relabel_completion_judge():
    model = ResponsesModel(
        _text_response(
            'BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="Done"\nEND REACT_DECISION'
        ),
        _text_response(
            'BEGIN GOAL_JUDGMENT\nCOMPLETED=true\nEVIDENCE="Verified"\nEND GOAL_JUDGMENT'
        ),
    )
    trace_output = StringIO()
    trace = RunTrace(trace_output, level="error")
    model._on_event = trace.llm_event

    assert asyncio.run(ReactMode(model, Runtime(), trace=trace).run("Finish")) == ExecutionAnswer(
        "Done", "completed"
    )

    lines = trace_output.getvalue().splitlines()
    assert sum(line.startswith("[llm usage]") for line in lines) == 2
    assert sum(line.startswith("[llm timing]") for line in lines) == 2
    assert sum("component=react" in line for line in lines) == 2
    assert sum("component=completion_judge" in line for line in lines) == 2
