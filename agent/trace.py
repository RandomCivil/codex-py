"""Human-readable progress tracing for an Agent run.

Tracing is deliberately sent to stderr so the CLI's final JSON result remains
safe for callers that consume stdout as a machine-readable interface.
"""

import hashlib
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
        self._llm_attribution: tuple[str, str] | None = None

    def llm_request(
        self,
        component: str,
        request: Any,
        *,
        static_shape: Any | None = None,
    ) -> None:
        """Associate the next model usage record with a safe request family.

        The request itself is still emitted only through the existing info-level
        context trace.  Usage records retain just a digest of the static shape,
        never the dynamic input or provider credentials.
        """
        label = component.strip() if isinstance(component, str) and component.strip() else "unknown"
        shape = _cache_relevant_shape(request) if static_shape is None else static_shape
        self._llm_attribution = (label, _request_family(shape))
        self.llm_context(request)

    def llm_event(self, event: Any) -> None:
        event_type = getattr(event, "type", "")
        if self._level == "error":
            if event_type == "response.created":
                self._output_buffer = ""
            elif event_type == "response.output_text.delta":
                self._output_buffer += getattr(event, "delta", "")
            elif event_type == "response.completed":
                # Token accounting remains useful at every level.  The full
                # request/response payload is reserved for info-level tracing.
                self._llm_usage(event)
                self.llm_final(getattr(event, "response", event))
                if self._output_buffer:
                    self._write("output complete", self._output_buffer)
                    self._output_buffer = ""
                self._llm_attribution = None
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
        if event_type == "response.completed":
            self._llm_usage(event)
            self.llm_final(getattr(event, "response", event))
            self._llm_attribution = None

    def _llm_usage(self, event: Any) -> None:
        response = getattr(event, "response", event)
        usage = _get_field(response, "usage")
        if usage is None:
            usage = _get_field(event, "usage_metadata")
        if usage is None:
            metadata = _get_field(event, "response_metadata") or {}
            usage = _get_field(metadata, "token_usage") or _get_field(metadata, "usage")
        if usage is None:
            return
        self._write_llm_usage(usage)

    def _write_llm_usage(self, usage: Any) -> None:
        input_details = (
            _get_field(usage, "input_tokens_details")
            or _get_field(usage, "input_token_details")
            or _get_field(usage, "prompt_tokens_details")
            or {}
        )
        output_details = (
            _get_field(usage, "output_tokens_details")
            or _get_field(usage, "output_token_details")
            or _get_field(usage, "completion_tokens_details")
            or {}
        )
        input_tokens = _first_defined(
            _get_field(usage, "input_tokens"), _get_field(usage, "prompt_tokens")
        )
        output_tokens = _first_defined(
            _get_field(usage, "output_tokens"), _get_field(usage, "completion_tokens")
        )
        cached_tokens = _get_field(input_details, "cached_tokens")
        if cached_tokens is None:
            cached_tokens = _get_field(input_details, "cache_read")
        attribution = ""
        if self._llm_attribution is not None:
            component, family = self._llm_attribution
            attribution = f"component={component} request_family={family} "
        self._line(
            "[llm usage] "
            + attribution
            + f"input_tokens={input_tokens} "
            f"output_tokens={output_tokens} "
            f"total_tokens={_get_field(usage, 'total_tokens')} "
            f"cached_tokens={cached_tokens} "
            f"reasoning_tokens={_get_field(output_details, 'reasoning_tokens')}"
        )

    def llm_usage(self, message: Any) -> None:
        """Print usage returned by a LangChain model message."""
        self._llm_usage(message)
        self._llm_attribution = None

    def llm_final(self, response: Any) -> None:
        """Print a model's terminal response at info level."""
        if self._level != "info":
            return
        self._line(f"[llm final] data={_compact(response)}")

    def llm_response(
        self,
        response: Any,
        *,
        component: str | None = None,
        static_shape: Any | None = None,
    ) -> None:
        """Record the terminal response and its usage from a non-streaming model."""
        if component is not None:
            self.llm_request(component, {}, static_shape=static_shape or {})
        self._llm_usage(response)
        self.llm_final(response)
        self._llm_attribution = None

    def llm_context(self, context: Any) -> None:
        """Print the exact request context submitted to a model at info level."""
        if self._level != "info":
            return
        self._line(f"[llm context] data={_compact(context)}")

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

    def tool_ignored(self, name: str, call_id: str | None = None) -> None:
        suffix = f" id={call_id}" if call_id else ""
        self._line(f"[tool call ignored] {name}{suffix} reason=single-call limit")

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
            return _as_serializable(model_dump(mode="json"))
        except (TypeError, ValueError):
            return _as_serializable(model_dump())
    if isinstance(value, Mapping):
        return {key: _as_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_as_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_as_serializable(item) for item in value]
    if hasattr(value, "__dict__"):
        return _as_serializable(vars(value))
    return value


def _get_field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _first_defined(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _cache_relevant_shape(request: Any) -> dict[str, Any]:
    """Select only invariant Responses request fields for request-family IDs."""
    if not isinstance(request, Mapping):
        return {}
    return {
        key: _as_serializable(request[key])
        for key in ("model", "instructions", "tools", "text", "response_format")
        if key in request
    }


def _request_family(shape: Any) -> str:
    """Return a deterministic, non-reversible identifier for static request shape."""
    encoded = json.dumps(
        _as_serializable(shape),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
