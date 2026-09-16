# ToolNode Plan–Execute Executor

Status: ready-for-agent

## Problem Statement

The Codex Python agent can derive an immutable, ordered Plan, but it cannot perform an individual Plan step. An agent developer needs a bounded Executor that can use the local Atom MCP tools under explicit host authority, produce an auditable Step execution result, and preserve the separation between planning, tool execution, and future replanning.

## Solution

Provide an asynchronous Executor for one identified Plan step. It accepts Agent state, a Plan revision, and a Plan-step ID; uses an OpenAI-compatible LangChain tool-calling model with `langchain_mcp_adapters` and a LangGraph `ToolNode`; and returns a completed or failed Step execution without changing Agent state or a Plan. The Executor owns an Atom MCP stdio session for its context-manager lifetime, limits a step to fifty model/tool rounds, and enforces explicit success and failure rules.

## User Stories

1. As an Agent developer, I want to execute one Plan step by revision and Plan-step ID, so that execution cannot be associated with the wrong immutable Plan revision.
2. As an Agent developer, I want the Executor to read Agent state without modifying it, so that the Agent remains responsible for durable state evolution.
3. As an Agent developer, I want the Executor to return a Step execution value, so that the Agent can explicitly record the outcome through its existing state seam.
4. As an Executor consumer, I want invalid revision or Plan-step references to fail explicitly, so that invalid orchestration never causes tools to run.
5. As an Executor consumer, I want the model to receive the goal, current Plan step, Plan history, Step executions, and optional memory summary, so that tool choices have the complete serializable execution context.
6. As an Agent developer, I want an OpenAI-compatible tool-calling model configuration, so that Executor deployments can use caller-supplied base URL, API key, and model name.
7. As a test author, I want to inject a LangChain chat model, so that Executor behavior can be tested without provider credentials.
8. As a tool owner, I want MCP tools loaded through `langchain_mcp_adapters`, so that the Executor consumes the Atom MCP server through its supported protocol integration.
9. As a tool owner, I want tool invocation to pass through LangGraph `ToolNode`, so that model-issued tool calls have a single explicit execution boundary.
10. As an Agent developer, I want the model/tool workflow to be a bounded `model → ToolNode → model` graph, so that the loop is observable and cannot run indefinitely.
11. As an application owner, I want at most fifty tool rounds per Step execution by default, so that a malformed or indecisive model cannot produce unbounded side effects.
12. As an application owner, I want the round limit to be configurable as a positive integer, so that hosts can set a stricter or looser operational budget deliberately.
13. As an Agent developer, I want a completed Step execution to require at least one successful tool call, so that the Executor cannot claim tool-based work merely from natural-language output.
14. As an Agent developer, I want the model's final answer to be a forced strict completion-tool call declaring completion and a concise result, so that completion is machine-validatable rather than inferred from prose.
15. As an Agent developer, I want the result to explain how the completion criterion was met, so that the recorded Execution result is useful to the future Agent and Planner.
16. As an Agent developer, I want a final model response without a valid forced completion-tool call to fail the Step execution, so that unsupported conversational exits cannot be recorded as success.
17. As an Agent developer, I want an MCP tool error to fail the Step execution immediately, so that the Executor does not silently recover or retry a possibly side-effecting action.
18. As an Agent developer, I want Atom `exec` with a nonzero exit code to fail the Step execution immediately, so that process failure is not counted as a successful tool call.
19. As an Agent developer, I want model errors and an exhausted round budget to produce failed Step executions, so that the surrounding Agent can decide whether to replan.
20. As an application owner, I want the Executor to be an asynchronous context manager, so that its MCP stdio process and session close predictably.
21. As an application owner, I want one Executor instance to reuse its MCP session across Plan steps, so that sequential work does not restart Atom MCP per step.
22. As an application owner, I want a default Atom MCP development configuration, so that local use works with the sibling Atom MCP workspace.
23. As an application owner, I want the Atom command, arguments, working directory, environment additions, and tool allowlist to be configurable, so that deployments can change integration details without editing Executor logic.
24. As a security-conscious host, I want all Atom environment configuration inherited by default and only explicitly supplied values supplemented, so that Executor code never embeds sandbox, workspace, database, or credential values.
25. As a security-conscious host, I want an optional tool allowlist, so that deployments can reduce the MCP tool set made available to the model.
26. As a maintainer, I want Executor outcomes to remain existing serializable Step execution data, so that no LangChain message or tool-trace type leaks into the domain model.
27. As a maintainer, I want the Executor to preserve ADR-0001's Planner/Executor/Agent boundary, so that tool authority, failure history, and future replanning remain auditable.

## Implementation Decisions

- Add the public Executor abstraction alongside the Planner. Its asynchronous operation accepts an Agent state, a Plan revision, and a Plan-step ID, and returns a Step execution value. It does not call `AgentState.with_step_execution`, mutate Agent state, mutate a Plan, invoke the Planner, or run a later Plan step.
- Resolve the requested Plan step from Agent state before creating a model/tool loop. A missing revision, missing step ID, or invalid state association is an explicit failed execution and must not cause an MCP connection or tool invocation.
- Add LangChain's OpenAI-compatible chat-model adapter and `langchain_mcp_adapters` dependencies. The production model is configured with the same base URL, API key, and model-name concepts as the established OpenAI-compatible provider; dependency injection of a chat model remains available for tests.
- Represent Atom MCP as an explicit stdio-server configuration. The development default invokes `poetry run atom-mcp` with `/home/xzp/workspace/atom-mcp` as its working directory. Command, arguments, working directory, supplemental environment values, and an optional tool allowlist are constructor configuration; the process environment is inherited and no Atom credentials or policy values are embedded.
- The Executor owns MCP connection and cleanup through asynchronous context management. Initializing the context loads the MCP tool set once, and closing it tears down its session and subprocess resources. Calls before initialization or after closure fail explicitly.
- Build the execution loop with a LangGraph StateGraph whose model node emits tool-capable messages and whose ToolNode executes requested MCP tools. Conditional routing loops to the model only while a tool call remains valid and within the configured budget.
- The model instruction identifies the goal, selected Plan step, its intent and completion criterion, the complete serializable Agent state, and the requirement to use MCP tools. After the MCP loop ends, a separate binding exposes only a local `report_step_completion` tool with a strict schema, a forced tool choice, and parallel tool calls disabled. Its only successful final response is that tool call with `completed: true`, `completion_criterion_met: true`, and a non-empty string `result` that explains the work. This isolates structured completion from MCP tools whose schemas are not necessarily strict-output compatible.
- A completed execution requires at least one successful ToolNode invocation. A no-tool response, malformed/extra-field completion-tool call, false completion flag, missing/non-text result, model exception, or exhausted fifty-round default budget yields `StepExecution(status="failed", error=...)`.
- Any MCP tool error immediately yields a failed Step execution without returning that error to the model for recovery. A result from Atom's `exec` tool whose structured exit code is nonzero likewise fails immediately. Neither case counts as a successful tool invocation.
- A successful execution returns `StepExecution(status="completed", result=...)`; its result is the concise final model explanation only. A failure returns a diagnostic `error` and does not place raw LangChain messages or MCP tool traces into Agent state.
- The calling Agent remains responsible for recording an outcome through the existing immutable `AgentState.with_step_execution` seam and, later, for evaluating failures and asking the Planner for a new Plan revision.
- The design follows ADR-0001 and ADR-0002: Plan revisions remain immutable, an Executor makes one bounded Step execution attempt, and planning/replanning remains outside the Executor.

## Testing Decisions

- Test the feature at the single high-level `Executor.execute(state, revision, step_id)` seam. Good tests assert observable returned Step execution data and visible MCP/model interactions, not private prompt text, node implementation details, or LangGraph internal state.
- Use an injected controlled LangChain chat-model double and a local controlled stdio MCP test server or equivalent controlled MCP integration. Tests must exercise the adapter-loaded tools and ToolNode loop without live model credentials or a real Atom sandbox/MySQL configuration.
- Verify correct Plan-step resolution, full Agent-state context, a successful at-least-one-tool execution, and a final result that satisfies the forced strict completion-tool contract.
- Verify failures for an unknown revision/step, no tool call, malformed or invalid completion-tool call, model failure, MCP error, Atom `exec` nonzero exit code, and exhausted tool-round budget. Verify those outcomes do not mutate the input Agent state or Plan.
- Verify MCP lifecycle through context management, tool-set reuse across multiple executions in one Executor instance, configuration forwarding for Atom stdio settings, environment supplementation, and allowlist filtering.
- Existing Planner tests provide the prior art for controlled asynchronous public seams, immutable state assertions, and credential-free tests. Existing Atom MCP tests provide the expected structured `exec` exit-code behavior; they are not a required integration environment for this feature.

## Out of Scope

- The top-level Agent loop, recording returned Step executions, evaluating results, Plan revision creation, and blocked-result production.
- Planner changes, Plan schema changes, mutable Plan execution state, persistence, process recovery, and concurrent execution.
- Retry, recovery, fallback tools, or model repair after any MCP error, nonzero Atom `exec` exit code, or failed Step execution.
- Parallel/dependency-graph Plan execution, multi-step execution in one call, and tool calls prescribed by a Plan step.
- A live provider test, a live Atom sandbox/MySQL integration test, credential provisioning, or changes to Atom MCP's security policy.
- Persisting complete model messages, MCP tool traces, or LangGraph state in Agent state.

## Further Notes

The project glossary's Agent, Agent state, Plan, Plan revision, Plan step, Step execution, Tool execution, Executor, MCP tool set, Execution result, and Tool-calling model terms apply. This feature is the execution half deliberately deferred by the structured Plan–Execute Planner specification and records the selected integration boundary in ADR-0002.
