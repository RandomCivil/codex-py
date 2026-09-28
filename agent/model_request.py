"""Shared helpers for tracing model requests and describing bound MCP tools."""

from collections.abc import Iterable
from typing import Any


async def stream_model_response(model: Any, messages: Any, trace: Any | None = None) -> Any:
    """Consume one LangChain model response through its streaming interface."""
    if trace is not None:
        callback = getattr(trace, "llm_stream_start", None)
        if callable(callback):
            callback()
    response = None
    try:
        # Keep injected legacy test doubles usable. Production ChatOpenAI
        # instances implement astream and therefore never take this branch.
        if not hasattr(model, "astream"):
            response = await model.ainvoke(messages)
        else:
            async for chunk in model.astream(messages):
                if trace is not None:
                    callback = getattr(trace, "llm_stream_chunk", None)
                    if callable(callback):
                        callback(chunk)
                response = chunk if response is None else response + chunk
    except Exception:
        if trace is not None:
            callback = getattr(trace, "llm_stream_end", None)
            if callable(callback):
                callback()
        raise
    if trace is not None:
        callback = getattr(trace, "llm_stream_end", None)
        if callable(callback):
            callback()
    return response


def trace_llm_context(trace: Any | None, context: Any) -> None:
    """Send request context to legacy trace doubles when available."""
    if trace is not None:
        callback = getattr(trace, "llm_context", None)
        if callback is not None:
            callback(context)


def trace_llm_request(
    trace: Any | None,
    component: str,
    context: Any,
    *,
    static_shape: Any,
) -> None:
    """Trace a model request while retaining legacy trace-double compatibility."""
    if trace is None:
        return
    callback = getattr(trace, "llm_request", None)
    if callback is not None:
        callback(component, context, static_shape=static_shape)
        return
    trace_llm_context(trace, context)


def trace_llm_validation_retry(trace: Any | None, component: str, error: ValueError) -> None:
    """Record a local-validation repair attempt when tracing is enabled."""
    if trace is None:
        return
    callback = getattr(trace, "llm_validation_retry", None)
    if callable(callback):
        callback(component, error)


def tool_request_shape(tools: Iterable[Any] | None) -> list[dict[str, Any]]:
    """Describe bound tools for a request-family digest without request content."""
    return [
        {
            "name": getattr(tool, "name", None),
            "description": getattr(tool, "description", None),
            "schema": _tool_schema(tool),
        }
        for tool in (tools or ())
    ]


def _tool_schema(tool: Any) -> Any:
    schema = getattr(tool, "args_schema", None)
    model_json_schema = getattr(schema, "model_json_schema", None)
    if callable(model_json_schema):
        return model_json_schema()
    return schema
