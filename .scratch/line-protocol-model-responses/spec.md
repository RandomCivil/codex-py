# Line Protocol model responses

Status: ready-for-agent

## Problem Statement

The agent currently asks OpenAI-compatible providers to produce structured responses through provider-specific `json_schema` and `json_object` settings. That makes structured contracts dependent on provider capabilities and leaves different request paths with different output-format mechanics. The system needs one provider-neutral, locally validated response representation for every language-model reply that is not a native tool call.

## Solution

Replace all `json_schema` and `json_object` output configuration with one strict UTF-8 Line Protocol. A response is a single typed block with explicit `BEGIN <TYPE>` and `END <TYPE>` delimiters. Local codecs parse the protocol into the existing component contracts, which retain final validation. Provider requests no longer include OpenAI Responses `text.format` or LangChain `response_format`; prompts describe the required Line Protocol instead.

Native function tool calls remain provider-native and use their existing tool definitions and parameter schemas. A reply must be either native tool calls or a complete Line Protocol block, never both. ReAct and the Plan-step Executor retain their separate completion request: a tool-selection reply with no native tool call is an empty `NO_TOOL` block, after which the host requests `GOAL_COMPLETION` or `STEP_COMPLETION`.

## User Stories

1. As an operator, I want every non-tool model response to use one provider-neutral Line Protocol, so compatible providers need not implement JSON Schema or JSON-object output modes.
2. As an application integrator, I want each protocol response parsed locally and validated against the existing Plan, task-analysis, completion, and Observation contracts, so format migration does not weaken invariants.
3. As an application integrator, I want direct and no-tool tool-agent answers decoded into the existing `Execution answer` text, so callers do not consume protocol framing.
4. As an application integrator, I want ReAct and Plan-step execution to retain their current separate completion request, so their established tool-selection and completion boundary is preserved.
5. As an operator, I want YAML containing the removed `response_format` field rejected before model or Tool execution, so stale configuration cannot appear to work.
6. As a maintainer, I want a stable, explicit codec registry for every response type, so field names, nested blocks, and arrays do not rely on English pluralization or provider behavior.
7. As an application owner, I want malformed, incomplete, unexpected, or mixed tool/protocol output to fail closed, so an ambiguous model reply never reaches domain logic.

## Protocol Contract

### Common grammar

- A response contains exactly one top-level block. Its first and last lines are matching `BEGIN <TYPE>` and `END <TYPE>` lines.
- `<TYPE>` and every field-path segment match `[A-Z][A-Z0-9_]*`.
- A scalar line is exactly `PATH=JSON_LITERAL`: no whitespace surrounds `=`, and the right side is one JSON literal. Strings therefore use JSON quotes and escapes, for example `QUERY="iphone 17"`; numbers, booleans, and `null` retain JSON spelling.
- Dot-separated paths encode nested object fields. Repeated declared scalar paths encode scalar arrays.
- A declared object array is represented by repeated, complete nested blocks. Fields within each child block are mapped explicitly by its parent codec; array elements never depend on positional zipping of repeated fields.
- CRLF is normalized to LF. Blank lines, comments, text outside a block, undeclared fields, invalid nesting, repeated non-array fields, invalid JSON literals, and a mismatched boundary type are errors.
- The codec registry is authoritative for permitted block types, fields, scalar-array paths, object-array child blocks, and their target contract fields. It does not infer plural forms.
- A complete parsed document must still pass the component's existing Pydantic or domain validation. A protocol or contract validation failure fails that model call; no repair, coercion, or retry is attempted for invalid output.

### Response types

| Type | Payload | Host handling |
| --- | --- | --- |
| `PLAN` | `REVISION`, `GOAL`, and repeated `STEP` blocks containing `ID`, `INTENT`, `COMPLETION_CRITERION` | Decode and validate as `Plan`. |
| `TASK_ANALYSIS` | Existing task-analysis fields rendered as uppercase snake-case scalar paths | Decode and validate as `TaskAnalysis`. |
| `STEP_COMPLETION` | `COMPLETED`, `COMPLETION_CRITERION_MET`, `RESULT`, plus repeated `FILES_READ`, `FILES_MODIFIED`, and `OBSERVATIONS` fields | Decode and validate as the existing Plan-step completion receipt. |
| `OBSERVATION` | `ROUND`, optional source-round fields, and repeated `EVIDENCE` blocks. Each evidence block declares its category, `TEXT`, and repeated tool-call IDs. | Decode and validate as `Observation`; retain the current confirmed-facts, reported-errors, and model-inferences semantics. |
| `GOAL_COMPLETION` | `ANSWER` and `GOAL_SATISFIED=true` | Decode and validate ReAct completion before returning its answer. |
| `ANSWER` | Exactly one nonempty `TEXT` field | Decode `TEXT` into an `Execution answer`. |
| `NO_TOOL` | No fields or child blocks | Valid only as the no-tool result of a ReAct or Plan-step tool-selection request; triggers its existing separate completion request. |

## Implementation Decisions

- Add a shared Line Protocol parser, renderer/prompt contract support, and a local codec registry. Keep it independent of provider clients and domain components except for explicit codec adapters.
- Remove `llm.response_format` and all `ResponseFormat` arguments, fields, imports, helper functions, request construction, tests, and documentation. There is no replacement configuration switch.
- Remove `response_format` from component provider configuration and YAML inheritance. A supplied legacy field is an unknown-field configuration error that explains it was removed because non-tool responses use Line Protocol.
- Remove Structured-output mode from non-secret configuration snapshots and fingerprints. Endpoint and model-name drift remain meaningful resume drift; API-key rotation remains excluded as today.
- The OpenAI Responses `LLM` request path must omit `text.format`. Planner, task analyzer, and direct mode prompt for their required response type and pass no output-format request option.
- The LangChain `ChatOpenAI` paths must omit `response_format` and no longer bind JSON response schemas. Native `tools` bindings and MCP tool schemas continue unchanged.
- Planner produces `PLAN`; the task analyzer produces `TASK_ANALYSIS`; runtime-context Observation and merge calls produce `OBSERVATION`; direct mode and tool-agent with no tool call produce `ANSWER`.
- ReAct and the Plan-step Executor keep their current extra completion invocation. Their tool-selection prompts require either native tool calls or `NO_TOOL`; the completion prompts require `GOAL_COMPLETION` and `STEP_COMPLETION`, respectively. A response containing both protocol text and tool calls is invalid.
- Stream raw text deltas internally until a complete block is available. Do not report a successful model response or terminal execution answer until parsing and contract validation succeed. Trace output may retain protocol text; public `Execution answer` values expose decoded answer text only.
- Preserve all existing Plan, Plan-step, Agent, Tool execution, tool-call batch, Durable State, Observation fallback, and recovery semantics except where they consume the former JSON response text.

## Testing Decisions

- Add focused parser and renderer tests for all scalar JSON literals, escaping, CRLF normalization, block matching, nesting, scalar arrays, object arrays, and all specified rejection cases.
- Test every registered response type against valid examples and malformed input; prove codecs do not infer unregistered field names or array plurality and that existing domain validation still rejects semantically invalid values.
- Update Planner, task analyzer, runtime-context, Executor, ReAct, direct, and tool-agent tests to capture prompts and provider request options. Assert the expected Line Protocol type is prompted, neither `text.format` nor `response_format` is sent, and decoded domain results retain prior behavior.
- Test `NO_TOOL` for ReAct and the Plan-step Executor: it alone causes the separate completion request; arbitrary no-tool prose, protocol text mixed with tool calls, and any invalid `NO_TOOL` payload fail.
- Test direct and no-tool tool-agent responses so `ANSWER.TEXT` becomes the public `Execution answer`; invalid or incomplete protocol fails rather than returning raw framing.
- Update configuration, CLI, registry, and resume tests: configurations without `response_format` succeed; legacy values are rejected before model/Tool work; snapshots and fingerprints contain no output-mode field; endpoint/model drift and API-key rotation preserve their existing behavior.
- Retire JSON Schema/JSON-object-specific request-shape assertions and replace them with provider-neutral request and local-validation assertions. No test requires a live provider or MCP host.

## Out of Scope

- Changing native MCP tool schemas, tool selection semantics, function-call argument validation, Tool execution, Tool-call batch behavior, or Tool runtime authority.
- Changing the two-stage ReAct or Plan-step completion flow, durable run/recovery rules, Plan semantics, or task-routing policy.
- Supporting JSON output modes, provider-side schema enforcement, silent migration of legacy YAML, permissive protocol parsing, automatic output repair, or invalid-output retries.
- Exposing Line Protocol framing as the public `Execution answer` payload.

## Further Notes

This specification implements [ADR-0017](../../docs/adr/0017-line-protocol-model-responses.md) and uses the terms defined in the root [CONTEXT.md](../../CONTEXT.md). It amends the structured-output portions of ADR-0006 and ADR-0007: the old provider-specific modes are removed, while their local contract validation and native-tool boundaries remain.
