from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any, Callable, Mapping

from openai import AsyncOpenAI
from openai.types.responses import ResponseStreamEvent


class LLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        *,
        on_event: Callable[[ResponseStreamEvent], None] | None = None,
    ) -> None:
        self._model_name = model_name
        self._on_event = on_event
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
        )

    async def __aenter__(self) -> "LLM":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.close()

    async def stream_events(
        self,
        input: str | Sequence[dict[str, Any]],
        *,
        instructions: str | None = None,
        tools: Iterable[dict[str, Any]] | None = None,
        text_format: Mapping[str, Any],
    ) -> AsyncIterator[ResponseStreamEvent]:
        _require_json_schema(text_format)
        request: dict[str, Any] = {
            "model": self._model_name,
            "input": input,
            "text": {"format": dict(text_format)},
        }
        if instructions is not None:
            request["instructions"] = instructions
        if tools is not None:
            request["tools"] = tools

        async with self._client.responses.stream(**request) as stream:
            async for event in stream:
                if self._on_event is not None:
                    self._on_event(event)
                yield event

    async def stream_text(
        self,
        input: str | Sequence[dict[str, Any]],
        *,
        instructions: str | None = None,
        tools: Iterable[dict[str, Any]] | None = None,
        text_format: Mapping[str, Any],
    ) -> AsyncIterator[str]:
        async for event in self.stream_events(
            input,
            instructions=instructions,
            tools=tools,
            text_format=text_format,
        ):
            if event.type == "response.output_text.delta":
                yield event.delta


def _require_json_schema(text_format: Mapping[str, Any]) -> None:
    if (
        text_format.get("type") != "json_schema"
        or not isinstance(text_format.get("name"), str)
        or not text_format["name"]
        or not isinstance(text_format.get("schema"), Mapping)
        or text_format.get("strict") is not True
    ):
        raise ValueError("Responses API calls require a strict JSON Schema text format")
