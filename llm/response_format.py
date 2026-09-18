"""Shared validation for provider Structured-output modes."""

from typing import Literal, cast


ResponseFormat = Literal["json_schema", "json_object"]
RESPONSE_FORMATS = frozenset({"json_schema", "json_object"})


def responses_text_format(response_format: ResponseFormat, *, name: str, schema: dict) -> dict:
    """Build the Responses API text format selected by configuration."""
    if response_format == "json_object":
        return {"type": "json_object"}
    if response_format == "json_schema":
        return {
            "type": "json_schema",
            "name": name,
            "strict": True,
            "schema": schema,
        }
    raise ValueError("response_format must be json_schema or json_object")


def chat_response_format(response_format: ResponseFormat, *, name: str, schema: dict) -> dict:
    """Build the Chat Completions/LangChain response format selected by configuration."""
    if response_format == "json_object":
        return {"type": "json_object"}
    if response_format == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "strict": True,
                "schema": schema,
            },
        }
    raise ValueError("response_format must be json_schema or json_object")


def require_response_format(value: object) -> ResponseFormat:
    """Return a supported mode or reject an invalid provider configuration."""
    if not isinstance(value, str) or value not in RESPONSE_FORMATS:
        raise ValueError("response_format must be json_schema or json_object")
    return cast(ResponseFormat, value)
