from io import StringIO
from types import SimpleNamespace
import re
from unittest.mock import patch

from agent.trace import RunTrace


def test_stream_timing_uses_first_to_last_text_delta_and_request_attribution():
    output = StringIO()
    trace = RunTrace(output, level="error")

    with patch("agent.trace.time.perf_counter", side_effect=(10.0, 10.5, 11.0, 12.0)):
        trace.llm_request("react", {"model": "test-model"}, static_shape={"request_kind": "tool_round"})
        trace.llm_event(SimpleNamespace(type="response.output_text.delta", delta="first"))
        trace.llm_event(SimpleNamespace(type="response.output_text.delta", delta="last"))
        trace.llm_event(
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(usage={"output_tokens": 4}),
            )
        )

    lines = output.getvalue().splitlines()
    assert re.fullmatch(
        r"\[llm usage\] component=react request_family=sha256:[0-9a-f]{64} "
        r"input_tokens=None output_tokens=4 total_tokens=None cached_tokens=None reasoning_tokens=None",
        lines[0],
    )
    assert re.fullmatch(
        r"\[llm timing\] component=react request_family=sha256:[0-9a-f]{64} "
        r"request_status=completed first_token_ms=500.00 stream_duration_ms=2000.00 "
        r"time_per_output_token_ms=166.67 tokens_per_second=6.00",
        lines[1],
    )


def test_failed_stream_emits_failed_timing_without_usage():
    output = StringIO()
    trace = RunTrace(output, level="error")

    with patch("agent.trace.time.perf_counter", side_effect=(20.0, 20.25, 20.75)):
        trace.llm_request("tool_agent", {"model": "test-model"})
        trace.llm_event(SimpleNamespace(type="response.output_text.delta", delta="partial"))
        trace.llm_stream_end()

    assert re.fullmatch(
        r"\[llm timing\] component=tool_agent request_family=sha256:[0-9a-f]{64} "
        r"request_status=failed first_token_ms=250.00 stream_duration_ms=750.00 "
        r"time_per_output_token_ms=None tokens_per_second=None",
        output.getvalue().strip(),
    )


def test_stream_timing_keeps_generation_metrics_null_without_text_deltas():
    output = StringIO()
    trace = RunTrace(output, level="error")

    with patch("agent.trace.time.perf_counter", side_effect=(30.0, 31.0)):
        trace.llm_request("react", {"model": "test-model"})
        trace.llm_event(
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(usage={"output_tokens": 4}),
            )
        )

    assert "first_token_ms=None" in output.getvalue()
    assert "time_per_output_token_ms=None tokens_per_second=None" in output.getvalue()


def test_stream_timing_keeps_generation_metrics_null_for_one_output_token():
    output = StringIO()
    trace = RunTrace(output, level="error")

    with patch("agent.trace.time.perf_counter", side_effect=(40.0, 40.5, 41.0)):
        trace.llm_request("react", {"model": "test-model"})
        trace.llm_event(SimpleNamespace(type="response.output_text.delta", delta="only"))
        trace.llm_event(
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(usage={"output_tokens": 1}),
            )
        )

    assert "time_per_output_token_ms=None tokens_per_second=None" in output.getvalue()
