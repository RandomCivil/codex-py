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

    def __init__(self, stream: Any = None, *, level: str = "info", run_id: str | None = None) -> None:
        if level not in {"info", "error"}:
            raise ValueError("level must be either 'info' or 'error'")
        self._stream = stream or sys.stderr
        self._level = level
        self._run_id = run_id
        self._output_buffer = ""

    def llm_event(self, event: Any) -> None:
        event_type = getattr(event, "type", "")
        if self._level == "error":
            if event_type == "response.created":
                self._output_buffer = ""
            elif event_type == "response.output_text.delta":
                self._output_buffer += getattr(event, "delta", "")
            elif event_type == "response.completed" and self._output_buffer:
                self._write("output complete", self._output_buffer)
                self._output_buffer = ""
            return

        # Keep the complete provider event available for debugging.  The
        # specialised lines below are still useful for following a run, but
        # they intentionally omit fields such as ids, status, usage, and
        # provider-specific metadata.
        self._line(f"[llm event] data={_compact(event)}")
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
        if self._level == "error":
            return
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

    def recovery_context(self, files_read: list[str], files_modified: list[str], observations: list[str]) -> None:
        """Show every trustworthy Step-context value loaded for a resume."""
        self._line(
            "[recovery context] data="
            + _compact(
                {
                    "files_read": files_read,
                    "files_modified": files_modified,
                    "observations": observations,
                }
            )
        )

    def tool_call(self, name: str, arguments: Any, call_id: str | None = None) -> None:
        if self._level == "error":
            suffix = f" id={call_id}" if call_id else ""
            self._line(f"[tool call] {name}{suffix} args={_compact(arguments)}")
            return
        suffix = f" id={call_id}" if call_id else ""
        self._line(f"[tool call] {name}{suffix} args={_compact(arguments)}")

    def tool_result(self, name: str, result: Any, *, error: bool = False) -> None:
        label = "tool result error" if error else "tool result"
        # Error results contain the information needed to diagnose or correct
        # the MCP call, so keep their complete payload at every log level.
        if self._level == "error" and not error:
            self._line(f"[{label}] {name}")
            return
        self._line(f"[{label}] {name} result={_compact(result)}")

    def _write(self, label: str, value: Any) -> None:
        text = str(value)
        if text:
            self._line(f"[llm {label}] {text}")

    def _line(self, text: str) -> None:
        prefix = f"[{self._run_id}] " if self._run_id else ""
        print(f"{prefix}{text}", file=self._stream, flush=True)


def _compact(value: Any) -> str:
    value = _as_serializable(value)
    if isinstance(value, Mapping | list | tuple):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            pass
    return str(value)


def _as_serializable(value: Any) -> Any:
    """Convert SDK model objects to their complete response payload."""
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except (TypeError, ValueError):
            return model_dump()
    if isinstance(value, Mapping | list | tuple):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return value
