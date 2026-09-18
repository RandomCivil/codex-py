"""Shared protocol for models that stream structured text responses."""

from collections.abc import AsyncIterator, Mapping
from typing import Any, Protocol


class TextStream(Protocol):
    def stream_text(
        self,
        input: str,
        *,
        instructions: str | None = None,
        tools: Any = None,
        text_format: Mapping[str, Any],
    ) -> AsyncIterator[str]: ...
