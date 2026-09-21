import asyncio
from types import SimpleNamespace

import httpx
from openai import APIConnectionError

from llm import LLM


def test_event_stream_omits_provider_text_format_and_reports_request(monkeypatch):
    observed = {}

    class Stream:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def __aiter__(self): return self
        async def __anext__(self): raise StopAsyncIteration

    class Client:
        def __init__(self, **kwargs): self.responses = SimpleNamespace(stream=lambda **kwargs: observed.update(kwargs) or Stream())
        async def close(self): pass

    monkeypatch.setattr("llm.llm.AsyncOpenAI", Client)
    asyncio.run(_consume(LLM("https://provider.test", "secret", "model"), "hello", instructions="Be concise"))
    assert observed == {"model": "model", "input": "hello", "instructions": "Be concise"}
    assert "text" not in observed


def test_text_stream_yields_only_text_deltas_in_order(monkeypatch):
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="Hello"),
        SimpleNamespace(type="response.reasoning.delta", delta="private"),
        SimpleNamespace(type="response.output_text.delta", delta=" world"),
    ]

    class Stream:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def __aiter__(self): return self._events()
        async def _events(self):
            for event in events: yield event

    class Client:
        def __init__(self, **kwargs): self.responses = SimpleNamespace(stream=lambda **kwargs: Stream())
        async def close(self): pass

    monkeypatch.setattr("llm.llm.AsyncOpenAI", Client)
    async def consume():
        llm = LLM("https://provider.test", "secret", "model")
        return [item async for item in llm.stream_text("hello")]
    assert asyncio.run(consume()) == ["Hello", " world"]


def test_complete_text_uses_non_streaming_responses_api_and_reports_request(monkeypatch):
    observed = {}

    class Client:
        def __init__(self, **kwargs):
            self.responses = SimpleNamespace(
                create=lambda **kwargs: observed.update(kwargs) or _response("Complete response")
            )
        async def close(self): pass

    monkeypatch.setattr("llm.llm.AsyncOpenAI", Client)

    assert asyncio.run(
        LLM("https://provider.test", "secret", "model").complete_text(
            "hello", instructions="Be concise"
        )
    ) == "Complete response"
    assert observed == {"model": "model", "input": "hello", "instructions": "Be concise"}


def test_event_stream_forwards_tools_without_text_format(monkeypatch):
    observed = {}

    class Stream:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def __aiter__(self): return self
        async def __anext__(self): raise StopAsyncIteration

    class Client:
        def __init__(self, **kwargs): self.responses = SimpleNamespace(stream=lambda **kwargs: observed.update(kwargs) or Stream())
        async def close(self): pass

    monkeypatch.setattr("llm.llm.AsyncOpenAI", Client)
    tools = [{"type": "function", "name": "lookup"}]
    asyncio.run(_consume(LLM("https://provider.test", "secret", "model"), [{"role": "user"}], tools=tools))
    assert observed == {"model": "model", "input": [{"role": "user"}], "tools": tools}


def test_event_stream_propagates_provider_error_without_retry(monkeypatch):
    error = APIConnectionError(message="provider failed", request=httpx.Request("POST", "https://provider.test"))

    class Stream:
        async def __aenter__(self): raise error
        async def __aexit__(self, *args): pass

    class Client:
        def __init__(self, **kwargs): self.responses = SimpleNamespace(stream=lambda **kwargs: Stream())
        async def close(self): pass

    monkeypatch.setattr("llm.llm.AsyncOpenAI", Client)
    try:
        asyncio.run(_consume(LLM("https://provider.test", "secret", "model"), "hello"))
    except APIConnectionError as actual:
        assert actual is error
    else:
        raise AssertionError("provider error was not propagated")


async def _consume(llm, input, **kwargs):
    return [event async for event in llm.stream_events(input, **kwargs)]


async def _response(output_text):
    return SimpleNamespace(output_text=output_text)
