"""Shared protocols for models that return structured text responses."""

import json
from collections.abc import AsyncIterator
from typing import Any, Awaitable, Callable, Protocol, TypeVar


Validated = TypeVar("Validated")


async def validated_text(
    request: Callable[[str], Awaitable[str]],
    *,
    instructions: str,
    validate: Callable[[str], Validated],
    on_retry: Callable[[ValueError], None] | None = None,
) -> Validated:
    """Request structured text again once when local validation rejects it.

    Providers cannot reliably enforce every local invariant.  The repair request
    therefore preserves the original protocol instructions and gives the model
    both its rejected response and the precise local validation error.
    """
    output = await request(instructions)
    try:
        return validate(output)
    except ValueError as error:
        if on_retry is not None:
            on_retry(error)
        repair_instructions = validation_feedback_instructions(instructions, error, output)
        return validate(await request(repair_instructions))


def validation_feedback_instructions(
    instructions: str, error: ValueError, output: str
) -> str:
    """Add local validation feedback to a structured-output repair request."""
    return (
        f"{instructions}\n\n"
        "Your previous response was rejected by local validation. Return a complete "
        "replacement response that satisfies every instruction above; do not explain "
        "the correction or repeat this feedback.\n"
        f"Validation error: {error}\n"
        f"Previous response (JSON-encoded): {json.dumps(output, ensure_ascii=False)}"
    )


class TextStream(Protocol):
    def stream_text(
        self,
        input: str,
        *,
        instructions: str | None = None,
        tools: Any = None,
    ) -> AsyncIterator[str]: ...


class TextCompletion(Protocol):
    async def complete_text(
        self,
        input: str,
        *,
        instructions: str | None = None,
        tools: Any = None,
    ) -> str: ...


class TextModel(TextStream, TextCompletion, Protocol):
    """Model interface supporting either native streaming or completion calls."""
