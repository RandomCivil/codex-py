# Tool-capable ephemeral modes use Responses API streaming

**Status: accepted.** Tool-agent mode and ReAct mode send every model request, including protocol repair and Completion judge requests, through the shared OpenAI Responses API adapter using streaming responses. This replaces their LangChain Chat Completions path while retaining host-owned Tool execution and their existing mode semantics. One timing and usage record is emitted for each provider request: timing uses the first and last text-delta arrival times, while provider usage is recorded without estimation. This makes tool-capable ephemeral executions observable through the same provider-native event boundary as the Plan-step Executor.

## Consequences

- `first_token_ms` is measured from request start to the first text delta and `stream_duration_ms` from request start to completion.
- For more than one output token with text deltas, `time_per_output_token_ms` is `(last_text_delta - first_text_delta) / (output_tokens - 1)` and `tokens_per_second` is its reciprocal in seconds. They are `null` when those values cannot be computed.
- Failed requests emit a timing record with `request_status=failed`; only provider-supplied usage is logged.
