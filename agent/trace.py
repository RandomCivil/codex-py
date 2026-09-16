"""Human-readable progress tracing for an Agent run.

Tracing is deliberately sent to stderr so the CLI's final JSON result remains
safe for callers that consume stdout as a machine-readable interface.
"""

import json
import sys
from collections.abc import Mapping
from typing import Any


class RunTrace:
    """Print the useful parts of a run as they happen."""

    def __init__(self, stream: Any = None) -> None:
        self._stream = stream or sys.stderr
        self._output_buffer = ""

    def llm_event(self, event: Any) -> None:
        event_type = getattr(event, "type", "")
        if event_type in {"response.reasoning.delta", "response.reasoning_summary_text.delta"}:
            self._write("thought", getattr(event, "delta", ""))
        elif event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            self._output_buffer += delta
            self._write("output", delta)
        elif event_type == "response.created":
            self._output_buffer = ""
        elif event_type == "response.completed" and self._output_buffer:
            self._write("output complete", self._output_buffer)
            self._output_buffer = ""

    def llm_text(self, label: str, value: Any) -> None:
        self._write(label, value)

    def llm_complete(self, revision: int, step_id: str, message: Any) -> None:
        self._line(
            f"[llm complete] revision={revision} step={step_id} "
            f"data={_compact(message)}"
        )

    def plan(self, status: str, revision: int | None = None) -> None:
        suffix = f" revision={revision}" if revision is not None else ""
        self._line(f"[plan] {status}{suffix}")

    def execute(self, status: str, revision: int, step_id: str, detail: str | None = None) -> None:
        suffix = f" {detail}" if detail else ""
        self._line(f"[execute] {status} revision={revision} step={step_id}{suffix}")

    def tool_call(self, name: str, arguments: Any, call_id: str | None = None) -> None:
        suffix = f" id={call_id}" if call_id else ""
        self._line(f"[tool call] {name}{suffix} args={_compact(arguments)}")

    def tool_result(self, name: str, result: Any, *, error: bool = False) -> None:
        label = "tool result error" if error else "tool result"
        self._line(f"[{label}] {name} result={_compact(result)}")

    def _write(self, label: str, value: Any) -> None:
        text = str(value)
        if text:
            print(f"[llm {label}] {text}", file=self._stream, flush=True)

    def _line(self, text: str) -> None:
        print(text, file=self._stream, flush=True)


def _compact(value: Any) -> str:
    if isinstance(value, Mapping | list | tuple):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            pass
    return str(value)
