# Responses API Tool Execution Observability

Status: ready-for-agent

## Problem Statement

Tool-agent mode and ReAct mode currently use a different model transport from the project's Responses API adapter. Their tool loops therefore do not receive the provider-native streaming event boundary that already supplies reliable LLM usage and timing tracing elsewhere. Operators cannot consistently attribute the latency and token usage of tool-agent requests, ReAct tool rounds, protocol repairs, or Completion judgments.

## Solution

Route every Model request owned by Tool-agent mode and ReAct mode through the shared OpenAI-compatible Responses API adapter as a streamed Event stream. Preserve each mode's existing host-owned Tool execution semantics while converting native function calls and tool results at the execution-mode boundary. Emit one independently attributed LLM timing record and one provider-backed usage record per request, including requests that fail.

## User Stories

1. As an operator, I want Tool-agent mode to use the OpenAI-compatible Responses API, so that its model traffic follows the project's common provider integration.
2. As an operator, I want ReAct mode to use the OpenAI-compatible Responses API for every tool round, so that iterative tool work has provider-native observability.
3. As an operator, I want Tool-agent protocol-repair requests to use the same API and logs as ordinary Tool-agent requests, so that repair cost is visible.
4. As an operator, I want ReAct Completion judge and judge-repair requests to use the same API and logs as ReAct tool rounds, so that completion validation cost is visible.
5. As a user of Tool-agent mode, I want its one-request and first-tool-call-only behavior preserved, so that changing provider transport does not change execution semantics.
6. As a user of ReAct mode, I want its multi-round, host-owned Tool execution preserved, so that changing provider transport does not change the tool-loop contract.
7. As a tool-runtime owner, I want the host to continue deciding which MCP tools run and when follow-up requests occur, so that model transport does not gain side-effect authority.
8. As an operator, I want every provider request to emit a separately attributable timing record, so that multiple ReAct rounds are not hidden in an aggregate duration.
9. As an operator, I want timing records to identify the existing model component, so that `react`, `tool_agent`, and `completion_judge` costs can be separated.
10. As an operator, I want `first_token_ms` measured to the first text delta, so that initial generation delay is visible.
11. As an operator, I want `stream_duration_ms` measured from request start to provider completion, so that total provider request latency is visible without including Tool execution time.
12. As an operator, I want generation-speed metrics based on the interval between the first and last text deltas, so that first-response wait does not distort steady-state generation speed.
13. As an operator, I want `time_per_output_token_ms` and `tokens_per_second` to be null when output is absent, has one or fewer tokens, or lacks text deltas, so that logs never contain fabricated, infinite, or misleading values.
14. As an operator, I want usage logs to use the provider's input, output, reasoning, cached, and total token fields when supplied, so that billing and capacity data remain authoritative.
15. As an operator, I want missing provider usage fields to remain null rather than estimated, so that observability does not misrepresent provider data.
16. As an operator, I want failed or interrupted model requests to emit a timing record marked `request_status=failed`, so that outages and partial streams are diagnosable.
17. As an operator, I do not want failed requests to invent usage data, so that metrics retain a clear provenance boundary.
18. As a CLI caller, I want LLM timing and usage to remain in the application trace stream, so that the final stdout result remains machine-readable and unpolluted.
19. As a provider integrator, I want tool-call and tool-result continuation data represented through Responses-native input/output semantics, so that OpenAI-compatible providers receive a valid multi-turn request sequence.
20. As a maintainer, I want controlled tests at the execution-mode seam, so that the migration can be verified without live provider credentials.

## Implementation Decisions

- Tool-agent mode and ReAct mode use the shared `LLM` Responses API adapter rather than a LangChain Chat Completions model for all requests they own.
- The adapter is consumed as a streamed Event stream for these modes. Responses function-call output is adapted into each mode's existing host-side tool-call representation; tool results are supplied in a valid subsequent Responses request.
- Tool-agent mode remains limited to one model turn and at most the first requested Tool execution, except for its existing local Line Protocol repair behavior. ReAct retains its existing iterative tool rounds and Completion judgment behavior.
- A trace request is started for every provider request and retains the current safe request-family attribution. Each request produces independent timing and usage output, including tool rounds, validation repairs, Completion judge requests, and judge repairs.
- `first_token_ms` is request start to first `response.output_text.delta`; `stream_duration_ms` is request start to `response.completed`.
- If provider `output_tokens > 1` and first/last text deltas exist, `time_per_output_token_ms` is `(last_text_delta_at - first_text_delta_at) / (output_tokens - 1)` and `tokens_per_second` is `(output_tokens - 1) / (last_text_delta_at - first_text_delta_at)`. Otherwise both fields are null.
- Timing excludes Tool execution time because it is bounded to a single provider request. A request that fails before completion emits a timing record with `request_status=failed` and null values where unavailable.
- Usage values come only from the provider completion response. The existing usage fields remain logged when present; unavailable fields are null and no client-side token estimation is added.
- Timing and usage use the existing application trace destination and log-level policy. They are not inserted into the user answer or CLI stdout result.

## Testing Decisions

- Test at the existing public Tool-agent and ReAct execution-mode seams using controlled Responses event-stream and Tool-runtime doubles; this is the highest seam that verifies transport selection, native function-call continuation, host-owned tool execution, and request-level tracing together.
- Verify Tool-agent's no-tool answer, first-tool-call-only behavior, and Line Protocol repair behavior remain externally unchanged after the transport migration.
- Verify ReAct's multiple tool rounds, native tool-result continuation, completion proposal, Completion judge, and judge repair behavior remain externally unchanged after the transport migration.
- Verify every provider request generates independently attributed usage and timing logs, including repair and Completion judge requests.
- Verify timing values against a controlled monotonic clock: first-text-delta latency, completion duration, the first-to-last-delta formulas, and null behavior for zero/one output tokens or no text deltas.
- Verify failed and interrupted streams produce `request_status=failed` timing records without fabricated usage.
- Verify trace output stays on the application trace stream and final CLI stdout remains unchanged.
- Reuse the existing Responses adapter event-stream tests and trace formatting tests as prior art, extending their controlled event and timing fixtures rather than adding live-provider tests.

## Out of Scope

- Changing Direct mode, Plan-execute mode, provider credentials, provider selection, pricing calculations, or token estimation.
- Changing MCP Tool permissions, Tool runtime ownership, ReAct round limits, Completion-judge decision rules, or Tool-agent's first-call-only contract.
- Adding metrics export, dashboards, persistent telemetry storage, distributed tracing, retries, or timeout-policy changes.
- Logging prompt contents, tool-result payloads, API keys, or other sensitive request data beyond the existing safe request-family attribution.

## Further Notes

This decision extends the Responses API boundary already established for the Plan-step Executor to the two tool-capable Ephemeral execution modes. It is recorded in ADR-0024. The test seam is the existing execution-mode interface with controlled Responses Event streams and Tool runtime doubles; it was selected to verify externally observable behavior without a live OpenAI-compatible provider.
