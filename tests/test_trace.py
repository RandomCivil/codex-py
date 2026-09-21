from io import StringIO
from types import SimpleNamespace
import re

from agent.trace import RunTrace


def test_model_request_usage_is_attributed_to_a_safe_deterministic_request_family():
    first_output = StringIO()
    first = RunTrace(first_output, level="error")

    first.llm_request(
        "planner",
        {
            "model": "test-model",
            "instructions": "Return a plan.",
            "input": "secret goal one",
            "tools": None,
        },
    )
    first.llm_event(
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                usage={
                    "input_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 4},
                    "total_tokens": 10,
                }
            ),
        )
    )

    second_output = StringIO()
    second = RunTrace(second_output, level="error")
    second.llm_request(
        "planner",
        {
            "model": "test-model",
            "instructions": "Return a plan.",
            "input": "different secret goal",
            "tools": None,
        },
    )
    second.llm_usage(
        SimpleNamespace(
            usage_metadata={
                "input_tokens": 11,
                "input_token_details": {"cache_read": 5},
                "total_tokens": 11,
            }
        )
    )

    first_line = first_output.getvalue().strip()
    second_line = second_output.getvalue().strip()
    family = re.search(r"request_family=(sha256:[0-9a-f]{64})", first_line).group(1)

    assert "component=planner" in first_line
    assert "secret goal one" not in first_line
    assert re.search(r"request_family=(sha256:[0-9a-f]{64})", second_line).group(1) == family
    assert "cached_tokens=4" in first_line
    assert "cached_tokens=5" in second_line


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


def test_error_level_hides_streamed_and_complete_model_text():
    output = StringIO()
    trace = RunTrace(output, level="error")

    trace.llm_event(SimpleNamespace(type="response.created"))
    trace.llm_event(SimpleNamespace(type="response.output_text.delta", delta="secret"))
    trace.llm_event(
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                output_text="secret",
                usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            ),
        )
    )
    trace.llm_complete(1, "step-1", "secret")

    assert "secret" not in output.getvalue()


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


def test_runtime_context_response_usage_is_attributed_without_global_request_state():
    output = StringIO()
    trace = RunTrace(output, level="error")

    trace.runtime_context_llm_request([{"dynamic": "tool output"}])
    trace.runtime_context_llm_response(
        SimpleNamespace(
            usage_metadata={"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}
        ),
        static_shape={
            "instructions": "Create the Observation.",
            "request_kind": "observation",
        },
    )

    assert output.getvalue().startswith(
        "[llm usage] component=runtime_context request_family=sha256:"
    )


def test_runtime_context_usage_does_not_consume_pending_react_attribution():
    output = StringIO()
    trace = RunTrace(output, level="error")
    trace.llm_request("react", {}, static_shape={"request_kind": "tool_round"})

    trace.runtime_context_llm_response(
        SimpleNamespace(usage_metadata={"total_tokens": 2}),
        static_shape={"request_kind": "observation"},
    )
    trace.llm_response(SimpleNamespace(usage_metadata={"total_tokens": 3}))

    usage_lines = output.getvalue().splitlines()
    assert "component=runtime_context" in usage_lines[0]
    assert "component=react" in usage_lines[1]


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


def test_trace_reports_task_routing_and_safe_analysis_fallback():
    output = StringIO()
    trace = RunTrace(output, level="error", run_id="run-123")

    trace.task_route("react")
    trace.task_route("plan_execute", analysis_failed=True)

    assert output.getvalue().splitlines() == [
        "[run-123] [task route] mode=react",
        "[run-123] [task route] mode=plan_execute reason=task_analysis_failed",
    ]


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
