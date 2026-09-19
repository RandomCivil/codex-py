# Prompt-cache-efficient Model requests

Status: ready-for-agent

## Problem Statement

As an operator of the Codex Python agent, I cannot tell whether repeated Model requests reuse an OpenAI-compatible provider's input-token cache, nor can I rely on semantically identical MCP tool sets producing identical Model request prefixes. Tool-capable Execution modes make several requests in one invocation, and the Runtime-context component makes additional structured requests after Tool rounds. Small request-shape drift in their fixed instructions, message ordering, or tool-definition ordering can turn reusable input into uncached input without changing the Agent's observable outcome.

## Solution

The agent will make cache-relevant portions of every optimizable Model request deterministic, minimal for its Structured-output mode, and observable. Tool-capable Execution modes will expose an MCP tool set in a stable name order before binding it to a Tool-calling model. Planner, Task Analyzer, ReAct mode, the Plan-step Executor, and the Runtime-context component will retain only the fixed guidance needed beyond their configured Structured-output contract, with dynamic state and evidence following it. Run tracing will attribute input-token and cache-read usage to the language-model component and a safe, deterministic request-shape identifier, allowing operators to compare cache effectiveness without recording secrets or changing Model request semantics.

## User Stories

1. As an operator, I want each Model request's usage trace attributed to its language-model component, so that I can identify which component consumes uncached input.
2. As an operator, I want cache-read token counts associated with a stable request-shape identifier, so that I can compare repeated request families without exposing prompt contents or credentials.
3. As an operator, I want missing provider cache metrics represented clearly, so that providers which do not report cache reads remain usable.
4. As an operator, I want the existing error-level trace to retain token-accounting visibility, so that cache efficiency can be monitored with reduced diagnostic output.
5. As an operator, I want Planner and Task Analyzer requests to keep their fixed instructions separate from the goal or Agent state, so that repeated calls retain the provider-cacheable prefix already supported by the Model request boundary.
6. As an operator, I want Task Analyzer's fixed instructions to avoid redundant contract text when strict structured output already supplies the same constraint, so that cache misses cost fewer input tokens.
7. As an operator, I want ReAct mode to preserve its fixed execution guidance and goal before changing Runtime context, so that successive Tool rounds share the longest valid prefix.
8. As an operator, I want the Plan-step Executor to preserve its fixed guidance and Durable State before changing Raw tool results, so that successive Tool rounds share the longest valid prefix.
9. As an operator, I want the Runtime-context component's Observation request to put its fixed contract before the settled Tool round, so that Observation generation requests can reuse their shared prefix.
10. As an operator, I want the Runtime-context component's merge request to put its fixed merge contract before the dynamic Observations, so that repeated compaction requests can reuse their shared prefix.
11. As an operator, I want dynamic round numbers, Tool-call IDs, Tool arguments, results, errors, and recovery evidence to remain outside fixed instruction content, so that they do not invalidate an earlier reusable prefix.
12. As a host integrator, I want an identical allowed MCP tool set to be bound in a deterministic order, so that incidental server enumeration order does not change the Tool-calling model request.
13. As a host integrator, I want the configured MCP allowlist to continue to determine which tools are exposed, so that cache optimization does not widen Tool execution authority.
14. As a user of ReAct mode, I want its Tool execution behavior and final structured goal-completion contract to remain unchanged, so that cache work does not alter completed or failed Execution answers.
15. As a user of Plan-execute mode, I want Plan-step completion receipts and Step context to remain unchanged, so that cache work does not alter durable Agent coordination.
16. As a user recovering an interrupted Step execution, I want reconciliation guidance and the rule excluding unconfirmed prior Tool output to remain intact, so that cache work does not weaken recovery safety.
17. As a maintainer, I want tests to verify request-family stability through public model and trace seams, so that implementation details can evolve without losing the cache contract.
18. As a maintainer, I want no extra static prompt added to one-shot Direct or Tool-agent modes solely for caching, so that a speculative cache opportunity does not increase ordinary request cost.
19. As an operator, I want Planner instructions to be as short as the configured Structured-output mode safely permits, so that Plan generation has a smaller fixed input cost without weakening local Plan validation.
20. As an operator, I want ReAct's tool-loop guidance and terminal goal-completion request to avoid repeating guarantees already supplied by the terminal Structured-output contract, so that repeated Tool rounds and finalization requests minimize fixed input.
21. As an operator, I want Plan-step Executor tool-loop guidance and completion-receipt instructions to avoid repeating guarantees already supplied by the configured receipt contract, so that a Plan step's repeated Model requests minimize fixed input.
22. As an operator, I want the `json_object` compatibility path to retain every textual constraint that is not enforced by that mode, so that prompt reduction never trades away compatibility or local validation.

## Implementation Decisions

- Preserve the current Model request principle: fixed instructions, fixed Structured-output mode, and fixed tool definitions precede per-goal, per-round, or per-result content. Dynamic values must not be interpolated into a fixed instruction.
- Preserve the current Planner and Task Analyzer Model request boundary, where static instructions are submitted separately from the input. Make fixed instructions Structured-output-mode aware: `json_schema` requests may omit constraints enforced by their strict schema; `json_object` requests retain textual field, shape, and JSON-only constraints required for compatibility and local validation.
- Apply the same Structured-output-mode-aware prompt reduction to ReAct's tool-loop/final goal-completion requests, the Plan-step Executor's operational/completion-receipt requests, and the Runtime-context component's Observation/merge requests. Retain all guidance needed for unstructured tool selection, evidence-based completion, recovery reconciliation, and `json_object` compatibility.
- Keep ReAct mode and the Plan-step Executor on the accepted layered Runtime context policy. Durable State remains before Observations and recent Raw tool results; Raw tool results remain transient and are never made Step context or Checkpoint data.
- Give the Runtime-context component separate fixed message contracts for creating an Observation and merging Observations. Send only the applicable dynamic source payload in a following message. The existing schema, source-ID preservation, local validation, and Structured-output mode remain the authority for the response contract.
- Sort the filtered MCP tool set by canonical tool name immediately before binding it to Tool-calling models in each tool-enabled Execution mode. Do not modify tool objects, descriptions, schemas, or the effective allowlist.
- Extend tracing at the existing Model-request and usage boundary with an explicit component label and a deterministic digest of cache-relevant static request shape. The digest must not include credentials, goal text, Agent state, Tool arguments, Tool results, Raw tool results, or response content.
- Emit the new attribution alongside existing input, output, total, cached, and reasoning token fields. Continue to accept both Responses-style `cached_tokens` and LangChain/provider-style `cache_read` metadata.
- Treat cache behavior as provider-reported telemetry rather than a correctness condition. An OpenAI-compatible provider may have its own cache threshold, TTL, model-specific scope, or omit cache telemetry entirely.
- Do not introduce provider-specific cache-control parameters, cross-provider tokenizer assumptions, server-side cache persistence, or reuse of prior response identifiers in this feature.
- This specification is consistent with ADR-0010: it retains the sole Runtime context window, the separation of Raw tool results from Durable State, and no-tool structured Observation/merge requests.

## Testing Decisions

- Test at the existing Model request/trace seam rather than asserting private string layout. A good test observes the messages, bound tools, or trace records provided to a controlled model and verifies the visible request family and telemetry attribution.
- Add controlled Runtime-context component tests proving that both Observation creation and Observation merging place their invariant contract before their dynamic payload while preserving schema binding, parsed Observations, expected rounds, and source Tool-call IDs.
- Add controlled ReAct and Plan-step Executor tests with a deliberately differently ordered MCP tool enumeration. Verify the model receives the same canonical tool order and that Tool execution semantics remain unchanged.
- Add trace tests for component attribution, deterministic request-shape identifiers, Responses-style cache usage, LangChain-style cache usage, missing cache metrics, and error-level trace behavior.
- Retain existing Planner and Task Analyzer tests that exercise their public Text stream seam and strict local validation. Add only behavior-level coverage needed to show static instructions and dynamic input remain separate after prompt tightening.
- Add controlled ReAct and Plan-step Executor tests covering both Structured-output modes: strict-schema requests must use the reduced fixed guidance while JSON-object requests retain the necessary textual contract; both must preserve tool-loop, completion, receipt, and recovery outcomes.
- Retain existing execution and durable-execution tests as regression coverage for final structured responses, completion receipts, Step context, recovery behavior, and allowlist enforcement.
- Use the existing controlled Model, Text stream, MCP client, and trace doubles as test prior art. No live OpenAI-compatible provider request is required.

## Out of Scope

- Changing the selected OpenAI-compatible provider, model, pricing, cache TTL, cache threshold, or provider-side cache configuration.
- Guaranteeing a cache hit or treating absent cache usage as an Agent, Planner, Executor, Runtime-context component, or Execution mode failure.
- Changing Agent state, Checkpoints, Raw tool-result retention, Step context, recovery decisions, Tool execution permissions, or MCP tool schemas.
- Adding static prompts to Direct mode or Tool-agent mode purely to manufacture a cacheable prefix. These one-shot modes have no fixed business prompt to safely reduce.
- Redesigning execution-mode routing, Plan generation, structured-output schemas, or terminal result contracts.

## Further Notes

The primary validation seam is the existing trace plus controlled Model-request boundary: it captures request-family identity and provider usage while keeping cache telemetry non-semantic. The implementation should validate improvement with repeated tool-capable sample runs and compare provider-reported cached input tokens by component and request-shape identifier. Existing run traces already surface cached-token metadata; this feature makes that data attributable and comparable.
