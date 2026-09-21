"""Shared protocol for models that stream structured text responses."""

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
