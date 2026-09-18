from io import StringIO
from types import SimpleNamespace

from agent.trace import RunTrace


def test_llm_event_prints_complete_event_payload():
    output = StringIO()
    trace = RunTrace(output)

    trace.llm_event(
        SimpleNamespace(
            type="response.completed",
            id="resp_123",
            status="completed",
            usage={"total_tokens": 7},
        )
    )

    assert output.getvalue().splitlines()[0] == (
        '[llm event] data={"id": "resp_123", "status": "completed", '
        '"type": "response.completed", "usage": {"total_tokens": 7}}'
    )


def test_error_level_suppresses_streaming_llm_events_but_keeps_final_output():
    output = StringIO()
    trace = RunTrace(output, level="error")

    trace.llm_event(SimpleNamespace(type="response.created"))
    trace.llm_event(SimpleNamespace(type="response.output_text.delta", delta="partial"))
    trace.llm_event(SimpleNamespace(type="response.completed"))

    assert output.getvalue().splitlines() == ["[llm output complete] partial"]


def test_error_level_keeps_tool_results_and_only_names_tool_calls():
    output = StringIO()
    trace = RunTrace(output, level="error")

    trace.tool_call("list_tasks", {"cwd": "/workspace"}, "call_1")
    trace.tool_result("list_tasks", {"tasks": ["one"]})

    assert output.getvalue().splitlines() == [
        '[tool call] list_tasks id=call_1 args={"cwd": "/workspace"}',
        "[tool result] list_tasks",
    ]


def test_error_level_prints_complete_mcp_error_result():
    output = StringIO()
    trace = RunTrace(output, level="error")

    trace.tool_result(
        "list_tasks",
        {"isError": True, "content": [{"type": "text", "text": "permission denied"}]},
        error=True,
    )

    assert output.getvalue().splitlines() == [
        '[tool result error] list_tasks result={"content": [{"text": "permission denied", "type": "text"}], "isError": true}'
    ]


def test_trace_prefixes_each_line_with_run_id():
    output = StringIO()
    trace = RunTrace(output, run_id="run-123")

    trace.plan("started", 1)

    assert output.getvalue() == "[run-123] [plan] started revision=1\n"


def test_trace_prints_every_loaded_recovery_context_value():
    output = StringIO()
    trace = RunTrace(output, run_id="run-123")

    trace.recovery_context(
        ["README.md"],
        ["agent/durable.py"],
        ["The checkpointed work is trustworthy."],
    )

    assert output.getvalue() == (
        '[run-123] [recovery context] data={"files_modified": ["agent/durable.py"], '
        '"files_read": ["README.md"], "observations": ["The checkpointed work is trustworthy."]}\n'
    )
