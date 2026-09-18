"""Shared validation for provider Structured-output modes."""

from typing import Literal, cast


ResponseFormat = Literal["json_schema", "json_object"]
RESPONSE_FORMATS = frozenset({"json_schema", "json_object"})


def require_response_format(value: object) -> ResponseFormat:
    """Return a supported mode or reject an invalid provider configuration."""
    if not isinstance(value, str) or value not in RESPONSE_FORMATS:
        raise ValueError("response_format must be json_schema or json_object")
    return cast(ResponseFormat, value)
