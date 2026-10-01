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
        stream: bool = False,
        reasoning: Mapping[str, Any] | None = None,
        on_event: Callable[[ResponseStreamEvent], None] | None = None,
        on_stream_end: Callable[[], None] | None = None,
        on_request: Callable[[Mapping[str, Any]], None] | None = None,
        on_response: Callable[[Any], None] | None = None,
    ) -> None:
        self._model_name = model_name
        self._stream = stream
        self._reasoning = reasoning
        self._on_event = on_event
        self._on_stream_end = on_stream_end
        self._on_request = on_request
        self._on_response = on_response
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
        if self._reasoning is not None:
            request["reasoning"] = self._reasoning

        if self._on_request is not None:
            self._on_request(request)

        completed = False
        try:
            async with self._client.responses.stream(**request) as stream:
                async for event in stream:
                    if self._on_event is not None:
                        self._on_event(event)
                    completed = event.type == "response.completed"
                    yield event
        finally:
            if not completed and self._on_stream_end is not None:
                self._on_stream_end()

    async def complete_response(
        self,
        input: str | Sequence[dict[str, Any]],
        *,
        instructions: str | None = None,
        tools: Iterable[dict[str, Any]] | None = None,
    ) -> Any:
        """Return one complete Responses API response without opening a stream."""
        request: dict[str, Any] = {
            "model": self._model_name,
            "input": input,
        }
        if instructions is not None:
            request["instructions"] = instructions
        if tools is not None:
            request["tools"] = tools
        if self._reasoning is not None:
            request["reasoning"] = self._reasoning
        if self._on_request is not None:
            self._on_request(request)
        response = await self._client.responses.create(**request)
        if self._on_response is not None:
            self._on_response(response)
        return response

    async def request_response(
        self,
        input: str | Sequence[dict[str, Any]],
        *,
        instructions: str | None = None,
        tools: Iterable[dict[str, Any]] | None = None,
        on_event: Callable[[ResponseStreamEvent], None] | None = None,
        on_stream_end: Callable[[], None] | None = None,
    ) -> Any:
        """Return a complete response, using the configured transport mode."""
        if not self._stream:
            return await self.complete_response(input, instructions=instructions, tools=tools)

        request: dict[str, Any] = {"model": self._model_name, "input": input}
        if instructions is not None:
            request["instructions"] = instructions
        if tools is not None:
            request["tools"] = tools
        if self._reasoning is not None:
            request["reasoning"] = self._reasoning
        if self._on_request is not None:
            self._on_request(request)

        completed = False
        try:
            async with self._client.responses.stream(**request) as stream:
                async for event in stream:
                    if self._on_event is not None:
                        self._on_event(event)
                    if on_event is not None:
                        on_event(event)
                    if event.type == "response.completed":
                        completed = True
                        response = event.response
            if not completed:
                raise RuntimeError("Responses stream ended without a completed response")
        finally:
            if not completed and self._on_stream_end is not None:
                self._on_stream_end()
            if not completed and on_stream_end is not None:
                on_stream_end()
        if self._on_response is not None:
            self._on_response(response)
        return response

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

    async def complete_text(
        self,
        input: str | Sequence[dict[str, Any]],
        *,
        instructions: str | None = None,
        tools: Iterable[dict[str, Any]] | None = None,
    ) -> str:
        """Return the text from one non-streaming Responses API response."""
        response = await self.complete_response(
            input,
            instructions=instructions,
            tools=tools,
        )
        return getattr(response, "output_text", "") or ""
