import asyncio
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError

from llm import LLM

TEXT_FORMAT = {
    "type": "json_schema",
    "name": "test_output",
    "strict": True,
    "schema": {"type": "object", "additionalProperties": False, "properties": {}, "required": []},
}


def test_event_stream_rejects_non_schema_text_formats_before_calling_provider():
    async def consume():
        llm = LLM("https://provider.test", "secret", "model")
        return [event async for event in llm.stream_events("hello", text_format={"type": "text"})]

    with pytest.raises(ValueError, match="strict JSON Schema"):
        asyncio.run(consume())


def test_json_object_mode_uses_provider_json_object_format(monkeypatch):
    observed = {}

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class Responses:
        def stream(self, **kwargs):
            observed.update(kwargs)
            return Stream()

    class Client:
        def __init__(self, **kwargs):
            self.responses = Responses()

        async def close(self):
            pass

    monkeypatch.setattr("llm.llm.AsyncOpenAI", Client)

    async def consume():
        llm = LLM("https://provider.test", "secret", "model", response_format="json_object")
        return [event async for event in llm.stream_events("hello", text_format=TEXT_FORMAT)]

    assert asyncio.run(consume()) == []
    assert observed["text"] == {"format": {"type": "json_object"}}


def test_event_stream_reports_the_provider_request(monkeypatch):
    observed = {}

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class Responses:
        def stream(self, **kwargs):
            return Stream()

    class Client:
        def __init__(self, **kwargs):
            self.responses = Responses()

        async def close(self):
            pass

    monkeypatch.setattr("llm.llm.AsyncOpenAI", Client)

    async def consume():
        llm = LLM(
            "https://provider.test",
            "secret",
            "model",
            on_request=lambda request: observed.update(request),
        )
        return [event async for event in llm.stream_events("hello", instructions="Be concise")]

    assert asyncio.run(consume()) == []
    assert observed == {"model": "model", "input": "hello", "instructions": "Be concise"}


def test_text_stream_yields_only_text_deltas_in_provider_order(monkeypatch):
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="Hello"),
        SimpleNamespace(type="response.reasoning.delta", delta="private"),
        SimpleNamespace(type="response.output_text.delta", delta=" world"),
        SimpleNamespace(type="response.function_call_arguments.delta", delta='{"q"'),
        SimpleNamespace(type="response.completed", response=SimpleNamespace(usage={"total_tokens": 3})),
    ]

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            pass

        def __aiter__(self):
            return self._events()

        async def _events(self):
            for event in events:
                yield event

    class Responses:
        def stream(self, **kwargs):
            return Stream()

    class Client:
        def __init__(self, **kwargs):
            self.responses = Responses()

        async def close(self):
            pass

    monkeypatch.setattr("llm.llm.AsyncOpenAI", Client)

    async def consume():
        llm = LLM("https://provider.test", "secret", "model")
        return [delta async for delta in llm.stream_text("hello", text_format=TEXT_FORMAT)]

    assert asyncio.run(consume()) == ["Hello", " world"]


def test_text_stream_closes_provider_stream_when_consumer_exits_early(monkeypatch):
    observed = {"stream_closed": False}

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            observed["stream_closed"] = True

        def __aiter__(self):
            return self

        async def __anext__(self):
            return SimpleNamespace(type="response.output_text.delta", delta="first")

    class Responses:
        def stream(self, **kwargs):
            return Stream()

    class Client:
        def __init__(self, **kwargs):
            self.responses = Responses()

        async def close(self):
            pass

    monkeypatch.setattr("llm.llm.AsyncOpenAI", Client)

    async def consume_one():
        llm = LLM("https://provider.test", "secret", "model")
        stream = llm.stream_text("hello", text_format=TEXT_FORMAT)
        assert await anext(stream) == "first"
        await stream.aclose()

    asyncio.run(consume_one())
    assert observed["stream_closed"] is True


def test_event_stream_submits_string_request_and_preserves_sdk_events(monkeypatch):
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="Hello"),
        SimpleNamespace(type="response.reasoning.delta", delta="Thinking"),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            delta='{"location":"Shanghai"}',
        ),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(usage={"total_tokens": 3}),
        ),
    ]
    observed = {}

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            observed["closed"] = True

        def __aiter__(self):
            return self._events()

        async def _events(self):
            for event in events:
                yield event

    class Responses:
        def stream(self, **kwargs):
            observed["request"] = kwargs
            return Stream()

    class Client:
        def __init__(self, **kwargs):
            observed["client"] = kwargs
            self.responses = Responses()

        async def close(self):
            observed["client_closed"] = True

    monkeypatch.setattr("llm.llm.AsyncOpenAI", Client)

    async def consume():
        async with LLM("https://provider.test", "secret", "model") as llm:
            return [event async for event in llm.stream_events("hello", text_format=TEXT_FORMAT)]

    assert asyncio.run(consume()) == events
    assert observed["client"] == {
        "base_url": "https://provider.test",
        "api_key": "secret",
        "max_retries": 0,
    }
    assert observed["request"] == {
        "model": "model",
        "input": "hello",
        "text": {"format": TEXT_FORMAT},
    }
    assert observed["closed"] is True
    assert observed["client_closed"] is True


def test_event_stream_forwards_input_items_instructions_and_tools(monkeypatch):
    observed = {}

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class Responses:
        def stream(self, **kwargs):
            observed.update(kwargs)
            return Stream()

    class Client:
        def __init__(self, **kwargs):
            self.responses = Responses()

        async def close(self):
            pass

    monkeypatch.setattr("llm.llm.AsyncOpenAI", Client)
    items = [{"role": "user", "content": "continue"}]
    tools = [{"type": "function", "name": "lookup"}]

    async def consume():
        llm = LLM("https://provider.test", "secret", "model")
        return [
            event
            async for event in llm.stream_events(
                items,
                instructions="Be concise",
                tools=tools,
                text_format=TEXT_FORMAT,
            )
        ]

    assert asyncio.run(consume()) == []
    assert observed == {
        "model": "model",
        "input": items,
        "instructions": "Be concise",
        "tools": tools,
        "text": {"format": TEXT_FORMAT},
    }


def test_text_stream_can_request_plain_text_without_tools_or_structured_format(monkeypatch):
    observed = {}

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            pass

        def __aiter__(self):
            return self._events()

        async def _events(self):
            yield SimpleNamespace(type="response.output_text.delta", delta="Answer")

    class Responses:
        def stream(self, **kwargs):
            observed.update(kwargs)
            return Stream()

    class Client:
        def __init__(self, **kwargs):
            self.responses = Responses()

        async def close(self):
            pass

    monkeypatch.setattr("llm.llm.AsyncOpenAI", Client)

    async def consume():
        llm = LLM("https://provider.test", "secret", "model")
        return [chunk async for chunk in llm.stream_text("hello", tools=None)]

    assert asyncio.run(consume()) == ["Answer"]
    assert observed == {
        "model": "model",
        "input": "hello",
    }


def test_event_stream_closes_provider_stream_when_consumer_exits_early(monkeypatch):
    observed = {"stream_closed": False, "client_closed": False}

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            observed["stream_closed"] = True

        def __aiter__(self):
            return self

        async def __anext__(self):
            return "first event"

    class Responses:
        def stream(self, **kwargs):
            return Stream()

    class Client:
        def __init__(self, **kwargs):
            self.responses = Responses()

        async def close(self):
            observed["client_closed"] = True

    monkeypatch.setattr("llm.llm.AsyncOpenAI", Client)

    async def consume_one():
        llm = LLM("https://provider.test", "secret", "model")
        stream = llm.stream_events("hello", text_format=TEXT_FORMAT)
        assert await anext(stream) == "first event"
        await stream.aclose()
        await llm.close()

    asyncio.run(consume_one())
    assert observed == {"stream_closed": True, "client_closed": True}


def test_event_stream_propagates_provider_error_without_retry(monkeypatch):
    provider_error = APIConnectionError(
        message="provider failed",
        request=httpx.Request("POST", "https://provider.test/v1/responses"),
    )
    observed = {"requests": 0, "client": None}

    class Stream:
        async def __aenter__(self):
            raise provider_error

        async def __aexit__(self, exc_type, exc, traceback):
            pass

    class Responses:
        def stream(self, **kwargs):
            observed["requests"] += 1
            return Stream()

    class Client:
        def __init__(self, **kwargs):
            observed["client"] = kwargs
            self.responses = Responses()

        async def close(self):
            pass

    monkeypatch.setattr("llm.llm.AsyncOpenAI", Client)

    async def consume():
        llm = LLM("https://provider.test", "secret", "model")
        stream = llm.stream_events("hello", text_format=TEXT_FORMAT)
        try:
            await anext(stream)
        finally:
            await stream.aclose()

    try:
        asyncio.run(consume())
    except APIConnectionError as error:
        assert error is provider_error
    else:
        raise AssertionError("provider error was not propagated")

    assert observed["requests"] == 1
    assert observed["client"]["max_retries"] == 0
