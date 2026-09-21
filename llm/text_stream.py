"""Shared protocols for models that return structured text responses."""

from collections.abc import AsyncIterator
from typing import Any, Protocol


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
