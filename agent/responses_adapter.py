"""Translate LangChain messages and MCP tools to Responses API values."""

import json
from collections.abc import Iterable, Mapping
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def responses_input(messages: Iterable[Any]) -> list[dict[str, Any]]:
    """Render internal message history as Responses input items."""
    result: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.tool_call_id),
                    "output": message_text(message),
                }
            )
            continue
        role = (
            "assistant"
            if isinstance(message, AIMessage)
            else "system"
            if isinstance(message, SystemMessage)
            else "user"
        )
        content = message_text(message)
        if content:
            result.append({"role": role, "content": content})
        if isinstance(message, AIMessage):
            for call in getattr(message, "tool_calls", ()) or ():
                result.append(
                    {
                        "type": "function_call",
                        "call_id": str(call.get("id") or "unknown"),
                        "name": str(call.get("name") or "unknown"),
                        "arguments": json.dumps(call.get("args", {}), ensure_ascii=False),
                    }
                )
    return result


def responses_tools(tools: Iterable[Any]) -> list[dict[str, Any]]:
    """Render MCP tool definitions as Responses function tools."""
    return [
        {
            "type": "function",
            "name": str(getattr(tool, "name", "unknown")),
            "description": str(getattr(tool, "description", "") or ""),
            "parameters": _tool_schema(tool),
        }
        for tool in tools
    ]


def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)


def _tool_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "args_schema", None)
    schema_method = getattr(schema, "model_json_schema", None)
    if callable(schema_method):
        value = schema_method()
    elif isinstance(schema, Mapping):
        value = dict(schema)
    else:
        value = getattr(tool, "args", None) or {"type": "object", "properties": {}}
    return value if isinstance(value, dict) else {"type": "object", "properties": {}}
