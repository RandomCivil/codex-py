# Line Protocol model responses

Status: ready-for-agent

**Amendment:** The ReAct response and judge contracts below are superseded by [ReAct completion judgment at a proposed terminal response](../react-completion-criteria/spec.md). Plan-step Executor behavior is unchanged.

## Problem Statement

The agent currently asks OpenAI-compatible providers to produce structured responses through provider-specific `json_schema` and `json_object` settings. That makes structured contracts dependent on provider capabilities and leaves different request paths with different output-format mechanics. The system needs one provider-neutral, locally validated response representation for every language-model reply that is not a native tool call.

## Solution

Replace all `json_schema` and `json_object` output configuration with one strict UTF-8 Line Protocol. A response is a single typed block with explicit `BEGIN <TYPE>` and `END <TYPE>` delimiters. Local codecs parse the protocol into the existing component contracts, which retain final validation. Provider requests no longer include OpenAI Responses `text.format` or LangChain `response_format`; prompts describe the required Line Protocol instead.

Native function tool calls remain provider-native and use their existing tool definitions and parameter schemas. ReAct accepts ordinary reasoning text with Tool calls but rejects a Tool-call response that also attempts a `REACT_DECISION` block before executing any calls. A no-tool ReAct response is a strict `REACT_DECISION` block; only `STATUS="completed"` triggers the separate Completion judge. The Plan-step Executor retains its own judgment and handoff contract.

## User Stories

1. As an operator, I want every non-tool model response to use one provider-neutral Line Protocol, so compatible providers need not implement JSON Schema or JSON-object output modes.
2. As an application integrator, I want each protocol response parsed locally and validated against the existing Plan, task-analysis, completion, and Observation contracts, so format migration does not weaken invariants.
3. As an application integrator, I want direct and no-tool tool-agent answers decoded into the existing `Execution answer` text, so callers do not consume protocol framing.
4. As an application integrator, I want ReAct's no-tool terminal proposal and Plan-step execution's separate completion boundary to use explicit, locally validated contracts.
5. As an operator, I want YAML containing the removed `response_format` field rejected before model or Tool execution, so stale configuration cannot appear to work.
6. As a maintainer, I want a stable, explicit codec registry for every response type, so field names, nested blocks, and arrays do not rely on English pluralization or provider behavior.
7. As an application owner, I want malformed, incomplete, unexpected, or mixed tool/protocol output rejected before domain logic or Tool execution; ReAct may request a bounded correction under its own spec.

## Protocol Contract

### Common grammar

- A response contains exactly one top-level block. Its first and last lines are matching `BEGIN <TYPE>` and `END <TYPE>` lines.
- `<TYPE>` and every field-path segment match `[A-Z][A-Z0-9_]*`.
- A scalar line is exactly `PATH=JSON_LITERAL`: no whitespace surrounds `=`, and the right side is one JSON literal. Strings therefore use JSON quotes and escapes, for example `QUERY="iphone 17"`; numbers, booleans, and `null` retain JSON spelling.
- Dot-separated paths encode nested object fields. Repeated declared scalar paths encode scalar arrays.
- A declared object array is represented by repeated, complete nested blocks. Fields within each child block are mapped explicitly by its parent codec; array elements never depend on positional zipping of repeated fields.
- CRLF is normalized to LF. Blank lines, comments, text outside a block, undeclared fields, invalid nesting, repeated non-array fields, invalid JSON literals, and a mismatched boundary type are errors.
- The codec registry is authoritative for permitted block types, fields, scalar-array paths, object-array child blocks, and their target contract fields. It does not infer plural forms.
- A complete parsed document must still pass the component's existing Pydantic or domain validation. ReAct handles an invalid operational decision with bounded feedback and one further budgeted round; its judge permits one tool-free repair request. Other components retain their own invalid-output behavior.

### Response types

| Type | Payload | Host handling |
| --- | --- | --- |
| `PLAN` | `REVISION`, `GOAL`, and repeated `STEP` blocks containing `ID`, `INTENT`, `COMPLETION_CRITERION` | Decode and validate as `Plan`. |
| `TASK_ANALYSIS` | Existing task-analysis fields rendered as uppercase snake-case scalar paths | Decode and validate as `TaskAnalysis`. |
| `STEP_COMPLETION` | `COMPLETED`, `COMPLETION_CRITERION_MET`, `RESULT`, plus repeated `FILES_READ`, `FILES_MODIFIED`, and `OBSERVATIONS` fields | Decode and validate as the existing Plan-step completion receipt. |
| `OBSERVATION` | `ROUND`, optional source-round fields, and repeated `EVIDENCE` blocks. Each evidence block declares its category, `TEXT`, and repeated tool-call IDs. | Decode and validate as `Observation`; retain the current confirmed-facts, reported-errors, and model-inferences semantics. |
| `GOAL_COMPLETION` | Legacy ReAct completion receipt | Superseded for ReAct by `REACT_DECISION` plus Completion judge. |
| `REACT_DECISION` | `STATUS` is `completed`, `failed`, or `need_tool`; `completed` requires nonempty `ANSWER`, `failed` requires nonempty `ERROR`, and `need_tool` has neither | Decode a no-tool ReAct decision; only `completed` requests judgment. |
| `COMPLETION_PROGRESS` | `ALL_COMPLETED` and exactly one numbered `CRITERION_VERDICT` block per supplied criterion, each with `VERIFIED` plus either `EVIDENCE` or `GAP` | Judge the current state of every criterion; a positive verdict releases ReAct's candidate answer. |
| `GOAL_JUDGMENT` | `COMPLETED` plus `EVIDENCE` when true or `GAP` when false | Judge a ReAct invocation without supplied criteria. |
| `ANSWER` | Exactly one nonempty `TEXT` field | Decode `TEXT` into an `Execution answer`. |
| `NO_TOOL` | Legacy no-tool marker | Superseded for ReAct; Plan-step execution follows its own current contract. |

## Implementation Decisions

- Add a shared Line Protocol parser, renderer/prompt contract support, and a local codec registry. Keep it independent of provider clients and domain components except for explicit codec adapters.
- Remove `llm.response_format` and all `ResponseFormat` arguments, fields, imports, helper functions, request construction, tests, and documentation. There is no replacement configuration switch.
- Remove `response_format` from component provider configuration and YAML inheritance. A supplied legacy field is an unknown-field configuration error that explains it was removed because non-tool responses use Line Protocol.
- Remove Structured-output mode from non-secret configuration snapshots and fingerprints. Endpoint and model-name drift remain meaningful resume drift; API-key rotation remains excluded as today.
- The OpenAI Responses `LLM` request path must omit `text.format`. Planner, task analyzer, and direct mode prompt for their required response type and pass no output-format request option.
- The LangChain `ChatOpenAI` paths must omit `response_format` and no longer bind JSON response schemas. Native `tools` bindings and MCP tool schemas continue unchanged.
- Planner produces `PLAN`; the task analyzer produces `TASK_ANALYSIS`; runtime-context Observation and merge calls produce `OBSERVATION`; direct mode and tool-agent with no tool call produce `ANSWER`.
- ReAct's no-tool response uses `REACT_DECISION`. `completed` requests a tool-free Completion judge; `failed` and `need_tool` do not. Tool calls with ordinary reasoning text remain valid; Tool calls with a `REACT_DECISION` block are rejected before execution. Plan-step execution follows its separate current judgment and handoff contract.
- Stream raw text deltas internally until a complete block is available. Do not report a successful terminal Execution answer until parsing and contract validation succeed. Trace output may retain protocol text; public `Execution answer` values expose decoded answer text only.
- Preserve all existing Plan, Plan-step, Agent, Tool execution, tool-call batch, Durable State, Observation fallback, and recovery semantics except where they consume the former JSON response text.

## Testing Decisions

- Add focused parser and renderer tests for all scalar JSON literals, escaping, CRLF normalization, block matching, nesting, scalar arrays, object arrays, and all specified rejection cases.
- Test every registered response type against valid examples and malformed input; prove codecs do not infer unregistered field names or array plurality and that existing domain validation still rejects semantically invalid values.
- Update Planner, task analyzer, runtime-context, Executor, ReAct, direct, and tool-agent tests to capture prompts and provider request options. Assert the expected Line Protocol type is prompted, neither `text.format` nor `response_format` is sent, and decoded domain results retain prior behavior.
- Test the ReAct `REACT_DECISION` statuses, required and forbidden fields, contradictory Tool calls, and bounded correction. Keep Plan-step tests on its separate contract.
- Test direct and no-tool tool-agent responses so `ANSWER.TEXT` becomes the public `Execution answer`; invalid or incomplete protocol fails rather than returning raw framing.
- Update configuration, CLI, registry, and resume tests: configurations without `response_format` succeed; legacy values are rejected before model/Tool work; snapshots and fingerprints contain no output-mode field; endpoint/model drift and API-key rotation preserve their existing behavior.
- Retire JSON Schema/JSON-object-specific request-shape assertions and replace them with provider-neutral request and local-validation assertions. No test requires a live provider or MCP host.

## Out of Scope

- Changing native MCP tool schemas, tool selection semantics, function-call argument validation, Tool execution, Tool-call batch behavior, or Tool runtime authority.
- Changing Plan-step completion flow, durable run/recovery rules, Plan semantics, or task-routing policy.
- Supporting JSON output modes, provider-side schema enforcement, silent migration of legacy YAML, or permissive protocol parsing. ReAct's bounded operational correction and judge repair are explicit exceptions to the earlier no-retry rule.
- Exposing Line Protocol framing as the public `Execution answer` payload.

## Further Notes

This specification implements [ADR-0017](../../docs/adr/0017-line-protocol-model-responses.md) and uses the terms defined in the root [CONTEXT.md](../../CONTEXT.md). It amends the structured-output portions of ADR-0006 and ADR-0007: the old provider-specific modes are removed, while their local contract validation and native-tool boundaries remain.
