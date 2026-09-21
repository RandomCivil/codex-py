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
        on_request: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self._model_name = model_name
        self._on_event = on_event
        self._on_request = on_request
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
    ) -> AsyncIterator[ResponseStreamEvent]:
        request: dict[str, Any] = {
            "model": self._model_name,
            "input": input,
        }
        if instructions is not None:
            request["instructions"] = instructions
        if tools is not None:
            request["tools"] = tools

        if self._on_request is not None:
            self._on_request(request)

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
    ) -> AsyncIterator[str]:
        async for event in self.stream_events(
            input,
            instructions=instructions,
            tools=tools,
        ):
            if event.type == "response.output_text.delta":
                yield event.delta
