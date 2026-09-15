# Async OpenAI-Compatible Responses Streaming

Status: ready-for-agent

## Problem Statement

The agent has no language-model adapter yet. It needs one asynchronous integration point that can connect to an OpenAI-compatible provider using caller-supplied credentials and model configuration, while safely exposing streamed model output to the layers that render text or orchestrate tools.

## Solution

Provide an `LLM` abstraction backed by the official OpenAI Python SDK's asynchronous Responses API client. It will own a reusable asynchronous client session and offer two explicit asynchronous stream contracts: a Text stream for consumers that only need generated text deltas, and an Event stream for consumers that must observe SDK response events such as tool calls, reasoning, and final usage.

## User Stories

1. As an agent developer, I want to configure an OpenAI-compatible provider with a base URL, API key, and model name, so that I can use hosted or self-hosted compatible services.
2. As an agent developer, I want an asynchronous LLM interface, so that model streaming does not block unrelated coroutine work.
3. As a text-rendering layer, I want a Text stream of only newly generated text deltas, so that I can render incremental output without interpreting provider events.
4. As a text-rendering layer, I want the Text stream to finish normally when the response completes, so that I can finalize rendering deterministically.
5. As an orchestration layer, I want an Event stream of OpenAI SDK `ResponseStreamEvent` values, so that I can distinguish text, reasoning, tool-call, completion, and usage events.
6. As an orchestration layer, I want Event stream values to retain the SDK's response semantics, so that no tool arguments or provider-supported event fields are lost in wrapper normalization.
7. As an orchestration layer, I want to submit a simple string as a Model request, so that ordinary prompt-and-stream interactions are concise.
8. As an orchestration layer, I want to submit a Responses input-item sequence as a Model request, so that I can construct multi-turn context and follow-up requests after tool execution.
9. As an orchestration layer, I want to supply optional instructions, so that response behavior can be scoped per request.
10. As an orchestration layer, I want to supply optional tool definitions, so that the provider can emit function-call events.
11. As a tool orchestrator, I want function-call events to be exposed but never executed by the LLM abstraction, so that only the upper layer controls permissions and side effects.
12. As a tool orchestrator, I want to decide whether and how to submit tool results in a later Model request, so that multi-turn control flow remains explicit.
13. As an application owner, I want a reusable asynchronous client session, so that connection resources can be reused across requests.
14. As an application owner, I want to use the LLM abstraction as an asynchronous context manager, so that its transport resources are closed predictably.
15. As an application owner, I want an explicit asynchronous close operation, so that applications with their own lifecycle management can release resources without a context manager.
16. As an application owner, I want an early-exited or cancelled stream to release its stream resources, so that abandoned consumers do not leak connections.
17. As an error-handling layer, I want provider and transport errors to surface as OpenAI SDK errors, so that I retain their actionable status and diagnostic information.
18. As a tool orchestrator, I do not want hidden retries, so that partially displayed output and potentially side-effecting tool workflows cannot be duplicated.
19. As an application owner, I want SDK-default timeout behavior initially, so that the initial adapter has a clear, provider-aligned timeout policy.

## Implementation Decisions

- The integration uses the official OpenAI Python SDK asynchronous client and its Responses API against an OpenAI-compatible provider.
- The public abstraction is named `LLM` and receives `base_url`, `api_key`, and `model_name` at construction.
- `LLM` exposes separate `stream_text` and `stream_events` methods rather than a flag that changes one method's yielded type.
- `stream_text` yields only successive text deltas as strings and ends normally after response completion.
- `stream_events` yields the SDK's `ResponseStreamEvent` objects directly. It does not introduce a project-specific event model.
- Both streaming methods accept a Model request whose input is either a string or a Responses input-item sequence, plus optional instructions and tool definitions. Other low-level Responses API parameters are intentionally not forwarded.
- Tool execution, tool-result submission, and the decision to continue a multi-turn interaction belong to the upper layer.
- `LLM` owns a reusable asynchronous SDK client, supports asynchronous context management, and provides an explicit asynchronous close operation.
- The adapter makes no automatic retry attempt and leaves timeouts at SDK defaults. SDK errors propagate to the caller.
- The project dependency set must include the official OpenAI Python SDK with support for the asynchronous Responses streaming API.

## Testing Decisions

- Test observable behavior at the `LLM` public-interface seam, using a controlled SDK/transport double rather than a live model provider.
- Verify that an Event stream preserves and yields each supplied SDK event in order, including text, reasoning, function-call, and completion/usage events.
- Verify that a Text stream yields only text-delta payloads, in order, and ignores non-text events.
- Verify both supported Model request input forms and forwarding of optional instructions and tools.
- Verify that provider configuration is applied to client construction and that the selected model is used for each request.
- Verify context-manager and explicit-close lifecycle behavior, including stream cleanup after early consumer exit where the SDK seam can observe it.
- Verify SDK exceptions propagate unchanged and that no retry request is initiated.
- There is no existing LLM test prior art in the repository; these tests establish the first public-adapter test seam.

## Out of Scope

- Executing model-requested tools, submitting tool outputs, or automatically running a multi-turn tool loop.
- A provider-neutral event schema or translation layer.
- Passing arbitrary Responses API parameters through the abstraction.
- Automatic retries, custom timeout configuration, backoff, rate-limit handling, or circuit breaking.
- Synchronous model calls, non-streaming convenience calls, prompt templating, persistence, and user-interface rendering.
- Integration tests against a live provider or real credentials.

## Further Notes

The terms OpenAI-compatible provider, Text stream, Event stream, Model request, and Tool execution follow the repository's domain glossary. No ADR currently exists for this area. A real implementation should verify the installed SDK's exact asynchronous Responses streaming entry points and event type imports against official OpenAI documentation before coding.
