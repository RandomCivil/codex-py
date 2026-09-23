"""Interpret MCP tool results consistently across tool-enabled execution modes."""

import re
from typing import Any, Mapping


_CODED_MCP_ERROR = re.compile(r"^\s*\[\d{4,}\]\s")


def tool_result_failed(value: Any) -> bool:
    """Return whether an MCP result explicitly represents a failed call.

    Atom reports some validation failures as a list of text content blocks, with
    a leading numeric diagnostic code (for example ``[1201] invalid patch``),
    instead of raising or returning the usual ``isError`` envelope.
    """
    if isinstance(value, Mapping):
        if value.get("isError") is True or value.get("error"):
            return True
        return isinstance(value.get("returncode"), int) and value["returncode"] != 0
    if isinstance(value, (list, tuple)):
        return any(
            isinstance(item, Mapping)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
            and _CODED_MCP_ERROR.match(item["text"]) is not None
            for item in value
        )
    return getattr(value, "isError", False) is True or (
        isinstance(getattr(value, "returncode", None), int) and value.returncode != 0
    )
