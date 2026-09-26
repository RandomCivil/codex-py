# Composable execution modes

Status: ready-for-agent

**Amendment:** ReAct completion behavior below is superseded by [ReAct completion judgment at a proposed terminal response](../react-completion-criteria/spec.md). The other Execution modes retain their contracts.

## Problem Statement

The agent currently exposes one Plan–Execute path whose `Executor` is deliberately limited to one Plan step. A caller that has characterized a task needs to assemble a suitable whole-task execution path without redefining that Plan-step boundary: a direct language-model answer, a single tool attempt, an iterative ReAct loop, or the existing durable Plan–Execute workflow.

## Solution

Introduce a common whole-task Execution mode abstraction with an explicit factory selector for `direct`, `tool_agent`, `react`, and `plan_execute`. Every selected mode returns an Execution answer. The existing Planner, Plan-step Executor, Agent, and DurableAgent retain their current responsibilities inside `plan_execute`; the three new modes are ephemeral. The future task analyzer and deterministic router may select the factory mode later, but neither is part of this feature.

## User Stories

1. As an application integrator, I want to select an Execution mode explicitly, so that a later task-analysis Router can assemble an appropriate runner without coupling to its implementation.
2. As an application integrator, I want every mode to return one Execution answer, so that callers consume a stable whole-task result regardless of the selected mode.
3. As an application integrator, I want an Execution answer to distinguish completed from failed work and include a safe error on failure, so that callers do not infer success from partial output.
4. As a maintainer, I want `Executor` to remain the component for one Plan step only, so that the established Planner/Executor/Agent boundary remains understandable and auditable.
5. As an application integrator, I want direct mode to make one language-model call with no exposed tools, so that simple generation tasks have the lowest-cost execution path.
6. As an application integrator, I want a nonempty direct-mode model response to complete the Execution answer, so that direct mode does not require an artificial Plan or completion receipt.
7. As an application integrator, I want a direct-mode provider or empty-response failure to produce a failed Execution answer, so that callers receive an explicit terminal outcome.
8. As an application integrator, I want tool-agent mode to make one language-model request and execute at most one tool invocation, so that its latency, tool authority, and cost are bounded.
9. As an application integrator, I want tool-agent mode to return the model text when no tool is selected, so that an unnecessary tool selection does not turn an otherwise valid answer into failure.
10. As an application integrator, I want tool-agent mode to execute only the first requested tool when a model response requests several, so that the mode honors its single-invocation contract.
11. As an application integrator, I want ignored extra tool requests recorded in trace output, so that audit records explain why a requested operation did not occur.
12. As an application integrator, I want a successful tool-agent result deterministically rendered as the final answer without a second model call, so that the mode's single-model-call limit remains enforceable.
13. As an application integrator, I want text tool results preserved and structured or multi-content results stably serialized, so that tool-agent answers remain reproducible and presentable.
14. As an application integrator, I want a tool error, invalid arguments, nonzero execution result, or unrenderable result to fail tool-agent mode, so that a tool failure is never presented as a completed answer.
15. As an application integrator, I want ReAct mode to iterate model requests and Tool executions without creating a Planner or Plan, so that dynamically discovered paths can be followed without whole-goal planning.
16. As an application integrator, I want ReAct mode to reuse the established concurrent tool-call batch behavior, so that independent calls from one model response settle together and preserve request order.
17. As an application integrator, I want ReAct tool errors returned to the model as error context, so that it may select a corrective next action within its remaining budget.
18. As an application owner, I want ReAct mode to default to a bounded fifty-round budget, so that an indecisive model cannot execute tools indefinitely.
19. As an application owner, I want ReAct to return completed only after a no-tool `REACT_DECISION` with `STATUS="completed"` receives a valid positive Completion judge verdict.
20. As an application owner, I want ReAct to fail when it declares `failed`, exhausts its round budget, or its model, judge, or Tool runtime fails unrecoverably.
21. As a security-conscious host, I want tool-agent and ReAct to share the constrained Tool runtime, so that both observe the MCP host configuration, allowlist, forced working directory, and session lifecycle consistently.
22. As a security-conscious host, I want one MCP session per tool-enabled mode invocation, so that temporary tool authority and subprocess resources are bounded and predictably cleaned up.
23. As an operator, I want `direct`, `tool_agent`, and `react` provider overrides in YAML alongside the existing Planner and Executor overrides, so that each mode can use an appropriate compatible provider and model.
24. As an operator, I want the new overrides to inherit shared `model` values, so that common endpoint, credential, and model values are declared once.
25. As an operator, I want only ReAct to require a Structured-output mode, so that Direct and tool-agent configuration does not imply an unused JSON contract.
26. As an operator, I want existing CLI commands to retain their current mode and durable-run behavior, so that adding programmatic mode composition does not silently change production invocation contracts.
27. As an application integrator, I want plan_execute mode to preserve the existing durable Planner/Executor run and recovery behavior, so that Plan revisions, checkpoints, and recovery semantics remain intact.
28. As an application integrator, I want a completed plan_execute Execution answer to use the last completed Plan-step handoff, so that whole-task output does not require a new final model request.
29. As an application integrator, I want a blocked or failed plan_execute run translated into a failed Execution answer, so that the common contract has only completed and failed terminal statuses.
30. As an application integrator, I want failed modes never to automatically switch to another mode, so that the caller or future Router explicitly controls cost, authority, and durability changes.
31. As a maintainer, I want existing direct-injection seams for models and durable Plan–Execute dependencies retained, so that unit tests and Python embeddings do not require YAML or CLI use.
32. As a future Router author, I want task analysis and routing deliberately absent from this feature, so that task characterization stays descriptive and mode-selection policy can be designed independently.

## Implementation Decisions

- Define Execution mode as the whole-task abstraction and preserve the existing Executor as the Plan-step Executor exclusively. The four selectable values are `direct`, `tool_agent`, `react`, and `plan_execute`.
- Provide an explicit factory at the single public composition seam. It accepts a caller-selected mode and the dependencies appropriate to that mode, then supplies a runner with a whole-task goal input and an Execution answer result. It does not implement task analysis, thresholds, or routing policy.
- Define Execution answer as a stable whole-task value containing an answer, a terminal `completed` or `failed` status, and a safe failure error where applicable. No mode performs automatic fallback or escalation.
- Direct mode makes exactly one tool-free language-model request. A nonempty text response completes; an empty response or invocation error fails.
- Tool-agent mode makes exactly one model request. With no tool request, nonempty model text is the completed answer. With tool requests, it executes the first request only, traces every later request as ignored because of the single-call limit, and never returns to the model.
- Tool-agent tool results are converted by one shared deterministic Tool-result renderer. Text remains text; structured and multi-content values use a stable presentation form; values that cannot be safely presented fail. A tool exception, invalid arguments, nonzero execution result, or renderer failure produces a failed Execution answer rather than a model-text fallback.
- ReAct mode uses the established model-to-tool-batch loop and a configurable positive round budget whose default is fifty. A batch preserves existing concurrent execution, request-order results, tool-error return-to-model behavior, allowlist enforcement, and forced working-directory behavior.
- Every no-tool ReAct response uses strict `REACT_DECISION` status. Only `completed` with a candidate answer requests the separate Completion judge; a positive verdict returns that candidate answer. `failed` terminates with an error; `need_tool` continues within budget. A Tool-call response containing a status declaration executes no Tools and receives a protocol-error continuation. ReAct does not create a Plan, invoke a Planner, create checkpoints, or resume an interrupted loop.
- Extract one shared Tool runtime for tool-agent and ReAct. It owns the host-configured MCP connection, loaded tool set, allowlist, working-directory enforcement, trace integration, and one session for the selected mode invocation. `plan_execute` continues to use equivalent existing constraints without changing its externally visible behavior.
- Retain plan_execute as an adapter over the current Planner/Plan-step Executor/Agent and durable run behavior. On completion, its Execution answer uses the last completed Plan-step Execution result; on blocked or failed runs, it returns a failed Execution answer. It performs no additional answer-synthesis model request.
- Extend YAML component configuration with optional `direct`, `tool_agent`, and `react` provider overrides at the same level as `planner` and `executor`. All component overrides inherit from shared `model` values. Direct and tool-agent require effective endpoint, credential, and model values; ReAct additionally requires the existing Structured-output mode. Do not add a mode option to the CLI in this feature.
- Preserve the current CLI, MySQL Checkpoint, configuration fingerprint, Plan-revision, recovery, and Tool-call-batch contracts for plan_execute. The newly introduced modes are explicitly ephemeral and are not registered as durable Agent runs.
- Keep the glossary and ADR-0007 as the source of terminology and architectural rationale. The implementation must not rename the Plan-step Executor into a generic runner.

## Testing Decisions

- Add tests at one new highest-level seam: explicit factory selection followed by runner execution of a goal and observation of the returned Execution answer. Good tests assert mode-visible behavior, tool effects, terminal answers, and terminal statuses; they do not assert private graph topology, prompt wording, or internal helper calls.
- Use controlled injectable model and tool-runtime doubles at that seam. Do not require live model credentials, an actual Atom MCP workspace, MySQL, or CLI invocation for the new mode tests.
- Verify Direct's one tool-free request, nonempty-text completion, and empty/error failure.
- Verify tool-agent's zero-tool text answer, first-call-only execution, trace record for ignored later calls, deterministic text and structured rendering, and failed outcomes for tool, validation, nonzero-result, and renderer errors.
- Verify ReAct's multi-round correction after returned Tool errors, concurrent ordered batch behavior, default and configured round budgets, all three no-tool statuses, Completion judge approval and rejection, rejection of contradictory status-and-Tool responses, and failure on exhaustion.
- Verify the shared Tool runtime applies allowlist and forced-working-directory restrictions and opens/closes one session per ephemeral tool-enabled invocation.
- Verify plan_execute maps existing completed and blocked/failed durable outcomes into the common Execution answer without altering the existing Agent, DurableAgent, Executor, Checkpoint, or recovery tests.
- Extend the existing configuration-boundary tests for shared-default inheritance, new component overrides, required ReAct Structured-output configuration, rejected invalid fields, and backwards-compatible existing Planner/Executor configuration.
- Existing Executor tests are the prior art for controlled model/tool interactions, concurrent Tool-call batches, tool error propagation, nonzero execution-result handling, allowlists, forced working directories, and session lifecycle. Existing configuration tests are the prior art for YAML inheritance and validation. Existing Agent and DurableAgent tests remain the prior art for plan_execute behavior.

## Out of Scope

- Implementing task analysis, parsing task-analysis output, deterministic routing rules, routing thresholds, or automatic mode selection.
- Adding a CLI `--mode` selector or changing the behavior of existing `run`, `resume`, or `migrate` commands.
- Making direct, tool-agent, or ReAct durable, resumable, checkpointed, lease-managed, or recoverable.
- Changing Plan, Plan revision, Plan step, Step execution, Planner, Plan-step Executor, Agent, DurableAgent, MySQL persistence, recovery decisions, or existing plan_execute tool-call batch semantics.
- Automatic fallback or escalation between modes after a failure.
- A second language-model call to summarize tool-agent output or to synthesize a final plan_execute answer.
- MCP-host policy changes, new tools, changes to Atom MCP, live provider tests, or live MySQL/MCP integration requirements.

## Further Notes

This spec follows the glossary definitions of Execution mode, Execution answer, Ephemeral execution, Tool runtime, and Tool-result renderer, and records the architectural boundary in ADR-0007. It must also preserve ADR-0001's planning/execution separation, ADR-0002's Plan-step Executor boundary, ADR-0003 and ADR-0004's durable recovery rules, ADR-0005's batch behavior, and ADR-0006's explicit YAML configuration authority. Existing unrelated edits to the current Executor and its tests are not part of this feature unless they are independently incorporated through the established plan_execute compatibility work.
