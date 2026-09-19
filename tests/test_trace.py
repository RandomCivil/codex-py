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
    assert output.getvalue().splitlines()[1:] == [
        "[llm usage] input_tokens=None output_tokens=None total_tokens=7 cached_tokens=None reasoning_tokens=None",
        '[llm final] data={"id": "resp_123", "status": "completed", "type": "response.completed", "usage": {"total_tokens": 7}}',
    ]


def test_error_level_suppresses_streaming_llm_events_and_final_response_but_keeps_usage():
    output = StringIO()
    trace = RunTrace(output, level="error")

    trace.llm_event(SimpleNamespace(type="response.created"))
    trace.llm_event(SimpleNamespace(type="response.output_text.delta", delta="partial"))
    trace.llm_event(
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                usage={
                    "input_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 4},
                    "output_tokens": 3,
                    "output_tokens_details": {"reasoning_tokens": 1},
                    "total_tokens": 13,
                }
            ),
        )
    )

    assert output.getvalue().splitlines() == [
        "[llm usage] input_tokens=10 output_tokens=3 total_tokens=13 cached_tokens=4 reasoning_tokens=1",
        "[llm output complete] partial",
    ]


def test_llm_usage_is_printed_from_completion_event_at_info_level():
    output = StringIO()
    trace = RunTrace(output)

    trace.llm_event(
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                usage={
                    "input_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 4},
                    "output_tokens": 3,
                    "total_tokens": 13,
                }
            ),
        )
    )

    assert output.getvalue().splitlines()[-1] == (
        '[llm final] data={"usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 4}, "output_tokens": 3, "total_tokens": 13}}'
    )


def test_llm_response_at_error_level_keeps_usage_but_hides_final_response():
    output = StringIO()
    trace = RunTrace(output, level="error")

    trace.llm_response(
        SimpleNamespace(
            content="final answer",
            usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        )
    )

    assert output.getvalue().splitlines() == [
        "[llm usage] input_tokens=2 output_tokens=1 total_tokens=3 cached_tokens=None reasoning_tokens=None",
    ]


def test_llm_usage_supports_langchain_message_metadata_and_cached_tokens():
    output = StringIO()
    trace = RunTrace(output)

    trace.llm_usage(
        SimpleNamespace(
            usage_metadata={
                "input_tokens": 20,
                "output_tokens": 5,
                "total_tokens": 25,
                "input_token_details": {"cache_read": 8},
            }
        )
    )

    assert output.getvalue() == (
        "[llm usage] input_tokens=20 output_tokens=5 total_tokens=25 "
        "cached_tokens=8 reasoning_tokens=None\n"
    )


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


def test_trace_prints_llm_context():
    output = StringIO()
    trace = RunTrace(output, run_id="run-123")

    trace.llm_context({"input": [{"role": "user", "content": "hello"}]})

    assert output.getvalue() == (
        '[run-123] [llm context] data={"input": [{"content": "hello", "role": "user"}]}\n'
    )


def test_error_level_hides_llm_context():
    output = StringIO()
    trace = RunTrace(output, level="error")

    trace.llm_context({"input": [{"role": "user", "content": "hello"}]})

    assert output.getvalue() == ""


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
