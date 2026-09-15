# Codex Python

This context coordinates language-model interactions for the Codex Python agent.

## Language

**OpenAI-compatible provider**:
A service exposing the OpenAI Responses API through a caller-supplied base URL, API key, and model name.
_Avoid_: OpenAI provider, model provider

**Text stream**:
An asynchronous sequence containing only successive generated text deltas; it ends after the model response completes.
_Avoid_: response stream, event stream

**Event stream**:
An asynchronous sequence of OpenAI SDK `ResponseStreamEvent` values, including text, tool-call, reasoning, and completion usage information.
_Avoid_: raw stream, normalized event stream

**Model request**:
The bounded request submitted to an OpenAI-compatible provider: input plus optional instructions and tool definitions.
_Avoid_: provider request, completion request

**Tool execution**:
The upper-layer responsibility for acting on a model-issued function call and deciding whether to make a follow-up model request.
_Avoid_: automatic tool loop, LLM tool execution
