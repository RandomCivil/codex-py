# Task analysis and deterministic execution-mode routing

Status: ready-for-agent

## Problem Statement

An application currently has to choose an Execution mode before it can run a goal, even though the project already supports direct, tool-agent, ReAct, and Plan–execute modes. Callers need one goal-level entry point that first characterizes the task, then applies a predictable routing policy, and finally returns both the execution outcome and the decision that led to it.

## Solution

Introduce a Task Analyzer that obtains a strict structured description of a goal using the supplied task-analysis prompt. A Task Router runs that analysis before any Execution mode work, applies the agreed deterministic routing policy, and invokes the matching existing Execution mode through the existing mode-factory seam. Its result retains the Execution answer, selected Execution mode, analysis, and a safe analysis error when analysis falls back. An unavailable, malformed, or failed analysis conservatively selects Plan–execute mode.

## User Stories

1. As an application integrator, I want to submit one goal to a task-routing entry point, so that mode selection happens before work begins.
2. As an application integrator, I want Task Analyzer to describe task characteristics rather than select an architecture, so that routing policy remains deterministic and reviewable.
3. As an application integrator, I want the analyzer to use the supplied task-analysis prompt, so that task complexity is interpreted consistently.
4. As an application integrator, I want a valid structured task analysis, so that routing is based on typed characteristics rather than free-form model prose.
5. As an application integrator, I want a goal requiring no tools routed to direct mode, so that simple work avoids tool and durable-run overhead.
6. As an application integrator, I want a tool-requiring goal with at most two expected steps and path uncertainty below 0.3 routed to tool-agent mode, so that bounded tool work stays bounded.
7. As an application integrator, I want a long-horizon goal routed to Plan–execute mode, so that sustained state tracking uses the durable workflow.
8. As an application integrator, I want a goal with at least three open subgoals routed to Plan–execute mode, so that multiple tracked concerns receive a Plan.
9. As an application integrator, I want a goal with need_replanning greater than 0.7 routed to Plan–execute mode, so that likely approach changes use the planning boundary.
10. As an application integrator, I want all other tool-requiring goals routed to ReAct mode, so that dynamic investigation can react to observations without an unnecessary Plan.
11. As an application integrator, I want the policy evaluated in the stated order, so that the no-tools and short-certain-tool cases retain their intended precedence.
12. As an application integrator, I want an analyzer failure to select Plan–execute mode, so that uncertainty never incorrectly grants a lightweight execution path.
13. As an application integrator, I want fallback routing to retain a safe analysis-error value, so that I can distinguish a deliberate Plan–execute selection from a conservative fallback.
14. As an application integrator, I want the selected Execution mode exposed with the final result, so that I can audit the routing choice.
15. As an application integrator, I want the successful Task Analysis exposed with the final result, so that I can inspect the characteristics that caused the choice.
16. As an application integrator, I want the returned whole-task outcome to preserve the existing Execution answer semantics, so that current consumers can still distinguish completed and failed work.
17. As a maintainer, I want the Plan-step Executor to remain exclusive to Plan–execute mode, so that routing does not blur the established Agent and Executor boundary.
18. As a maintainer, I want no automatic escalation after a selected Execution mode fails, so that failures remain attributable to the selected mode rather than hidden secondary work.
19. As a test author, I want controlled analysis output and a controlled mode factory at the routing seam, so that policy behavior can be tested without live provider, MCP, or MySQL dependencies.

## Implementation Decisions

- Use the glossary terms Task Analyzer, Execution mode, Execution answer, Agent, and Plan-step Executor consistently. The Task Analyzer is descriptive only; it must not recommend a mode in its response.
- Represent the analyzer response as a strict, locally validated task-characteristics value. It contains exactly the prompt-defined task type, bounded numeric scores, tool requirement, expected-step and subgoal counts, horizon, risk, parallelism, and concise classification rationale.
- Request structured output using the project’s established Structured-output mode convention. `json_schema` uses the strict task-analysis schema; `json_object` still undergoes identical local validation.
- Add one highest-level Task Router seam with a goal input, an injected Task Analyzer, and an injected Execution-mode factory. It performs exactly: analyze goal, route characteristics, create selected mode, run goal, return routed outcome.
- Apply this exact ordered policy:
  1. `needs_tools == false` selects `direct`.
  2. `expected_steps <= 2` and `path_uncertainty < 0.3` select `tool_agent`.
  3. `expected_horizon == long`, `open_subgoals >= 3`, or `need_replanning > 0.7` selects `plan_execute`.
  4. Every remaining task selects `react`.
- Catch analyzer invocation and validation failures at the Task Router boundary. Select `plan_execute`, omit unavailable analysis, and return a safe diagnostic string; do not retry analysis or change the selected mode after execution begins.
- Return a routed outcome that wraps the existing Execution answer and additionally exposes `execution_mode`, optional analysis, and optional analysis error. It must not redefine completed/failed Execution answer semantics.
- Preserve the existing explicit Execution-mode factory and reuse it rather than creating mode-specific executor interfaces. Direct, tool-agent, and ReAct remain Ephemeral executions; only Plan–execute keeps its existing durable Agent-run behavior.
- This feature is a programmatic composition boundary. Existing CLI run/resume behavior, including durable run IDs and recovery, remains unchanged unless specified by a later CLI-focused change.
- Record the routing policy as a new ADR because it is an ordered, externally observable policy that extends ADR-0007’s deliberately deferred task-analysis routing decision.

## Testing Decisions

- Test at the new Task Router seam: supply a goal, controlled structured analyzer output, and a recording mode factory; assert only the selected mode, goal passed to that mode, and routed outcome. Do not test prompt implementation details or private parser helpers.
- Add focused Task Analyzer contract tests using the same controlled text-stream doubles used by Planner tests. Verify schema/json-object requests, successful parsing, every required field, rejected extra or missing fields, invalid categories, out-of-range scores, invalid counts, invalid boolean values, empty rationale, malformed JSON, and provider failure propagation.
- Cover each routing branch, including threshold boundaries: no tools; two versus three expected steps; `0.299…` versus `0.3` uncertainty; long horizon; three subgoals; `0.7` versus greater-than-`0.7` replanning; and the ReAct default branch.
- Verify the policy precedence with combined characteristics, especially no-tools tasks that otherwise look long or replanning-heavy, and short/certain tool tasks that also satisfy a later Plan–execute condition.
- Verify analyzer exceptions and malformed output choose Plan–execute once, preserve a safe analysis error, and do not attempt another mode.
- Verify the selected mode’s completed and failed Execution answers are returned unchanged inside the routed outcome; no automatic retry, fallback, or escalation occurs after the mode is selected.
- Existing Planner structured-output tests are prior art for injectable text streams, JSON-schema/json-object behavior, and local validation. Existing Execution-mode tests are prior art for factory selection and common Execution answer behavior.

## Out of Scope

- Changing the four existing Execution mode implementations, their tool authority, or their terminal semantics.
- Renaming or generalizing the Plan-step Executor into a common execution abstraction.
- Automatic retries, mode escalation, or re-analysis after a selected mode begins or fails.
- Making direct, tool-agent, or ReAct durable, resumable, checkpointed, or eligible for Agent-run recovery.
- Adding a CLI mode selector, changing existing CLI run/resume/migrate contracts, or persisting analysis metadata in the run registry.
- Changing Planner, Plan, Plan revisions, Plan steps, Step execution recovery, Tool-call batches, MCP configuration, or model-provider configuration beyond what is necessary to instantiate the Task Analyzer at the programmatic seam.

## Further Notes

This specification implements the future deterministic router anticipated by ADR-0007 while preserving ADR-0001’s Plan–execute responsibilities, ADR-0002’s Plan-step Executor boundary, ADR-0005’s tool-call batching, and ADR-0006’s explicit provider configuration convention. The classification rationale is an auditable concise summary, not chain-of-thought or an architecture recommendation.
