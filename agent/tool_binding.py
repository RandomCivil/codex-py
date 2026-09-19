"""Stable MCP tool-definition ordering for Tool-calling model bindings."""

from typing import Any, Iterable


def canonical_mcp_tool_set(tools: Iterable[Any]) -> list[Any]:
    """Return the same effective MCP tools ordered by their canonical names."""
    return sorted(tools, key=lambda tool: str(getattr(tool, "name", tool)))
