Status: ready-for-agent

# Plan–Execute Agent Loop

## Problem Statement

The project has independent Planner and Executor components, but no top-level Agent that turns a goal into a complete, auditable Plan–Execute run. Callers must currently coordinate planning, sequential Step executions, failure recording, replanning, the three-revision budget, and the Executor's MCP session themselves. That duplicates policy outside the domain boundary and makes reliable resume behavior unavailable.

## Solution

Provide an Agent that owns the Plan–Execute loop. A caller supplies an Agent state, and the Agent produces a terminal result containing the final Agent state. It derives a Plan when necessary, executes its Plan steps serially, records every Step execution, replans after failure, and returns a Blocked result when the Plan-revision budget is exhausted. The Agent owns the Executor context for the goal run and can safely continue from previously recorded Agent state.

## User Stories

1. As a host application developer, I want to run a goal through one Agent entry point, so that I do not coordinate Planner and Executor calls myself.
2. As a host application developer, I want the Agent to return the final Agent state, so that I can inspect the complete planning and execution history.
3. As a host application developer, I want the Agent to obtain an initial Plan from the Planner, so that a goal begins with a machine-validatable execution contract.
4. As a host application developer, I want Plan steps to execute in their declared order, so that later steps can rely on the outcomes of earlier steps.
5. As a host application developer, I want each Step execution recorded immediately, so that progress and failure are durable and auditable.
6. As a host application developer, I want completed Plan steps not to run again when Agent state is resumed, so that side-effecting Tool executions are not repeated.
7. As a host application developer, I want interrupted `running` and `skipped` Step executions retried, so that only a verified completed completion criterion is treated as done.
8. As a host application developer, I want a failed Step execution to stop the current Plan revision, so that the Planner can reconsider the whole remaining goal with the failure history available.
9. As a host application developer, I want each replan appended as a new immutable Plan revision, so that prior Plans and their outcomes remain reviewable.
10. As a host application developer, I want the Agent to attempt at most three Plan revisions, so that a failing goal has bounded cost and predictable termination.
11. As a host application developer, I want a Blocked result after failure exhausts the revision budget, so that I can distinguish a bounded unresolved goal from a completed one.
12. As a host application developer, I want invalid Planner output to surface as a planning error, so that an invalid Plan is not misrepresented as an execution failure.
13. As a host application developer, I want the Agent to manage the Executor context for the whole goal run, so that MCP resources are opened once and reliably released.
14. As a host application developer, I want a resumed Agent state whose current Plan is already complete to finish without a new Plan or Tool execution, so that completed work remains stable.
15. As a host application developer, I want the Executor to remain responsible only for one Step execution, so that planning and tool authority boundaries remain explicit.

## Implementation Decisions

- Introduce an Agent coordinator in the existing agent package. It accepts a Planner and Executor collaborator and exposes an asynchronous `run` operation over Agent state.
- The terminal Agent result has an explicit status distinguishing successful completion from a Blocked result and always carries the final Agent state.
- The Agent owns one Executor asynchronous context for each `run` call. Callers provide an inactive Executor rather than entering its context themselves.
- With no Plan history, the Agent requests the initial Plan, appends it immutably to Agent state, and begins serial execution.
- For the latest Plan revision, completed Step executions are skipped. Pending, running, and skipped entries are executed or re-executed in declared step order. A failed entry prevents further steps in that revision and causes replanning.
- A completed Step execution is appended to Agent state before the next Plan step is considered. A failed Step execution is appended before the Planner is called again.
- The Planner receives the full current Agent state for every Plan revision. Plans are never mutated or replaced.
- A current Plan whose steps are all completed returns a successful terminal result, including when reached by resuming Agent state.
- Once a failure occurs in revision three, the Agent returns a Blocked result with the accumulated state rather than requesting a fourth Plan revision.
- `PlanningValidationError` and other Planner contract failures propagate to the caller. They do not consume an additional Plan revision and do not become a Blocked result.
- The existing Planner and Executor remain the seams for model and MCP behavior; the Agent introduces no tool selection, tool retry, or Plan mutation policy.
- This preserves ADR-0001's immutable serial Plan–Execute model and completes the coordination boundary explicitly deferred by ADR-0002.

## Testing Decisions

- Test observable Agent behavior at the single high-level `run` seam using controlled Planner and Executor doubles; tests should assert terminal status, collaborator inputs, and resulting Agent state rather than internal loop mechanics.
- Cover initial planning followed by ordered successful Step executions and successful terminal completion.
- Cover a failure in an early Plan step, verify later steps in that revision do not run, then verify a new revision is planned from state containing the failure.
- Cover exhaustion after the third failed Plan revision and assert a Blocked result with all immutable Plan revisions and failed Step executions retained.
- Cover resumed state: completed steps are not invoked again; pending, running, and skipped steps are invoked in Plan order; a pre-existing failed current revision triggers replanning.
- Cover an already completed current Plan returning successfully without Planner or Executor work.
- Cover Planner validation failure propagating unchanged and ensure it is not converted into a Step execution or Blocked result.
- Cover Executor context entry and exit once per Agent run, including error cleanup where practical.
- Follow the existing Planner tests for strict revision and Agent state expectations, and the existing Executor tests for asynchronous controlled collaborators and result assertions.

## Out of Scope

- Changing the Planner's strict JSON Plan contract or its three-revision validation.
- Changing Executor MCP tool selection, tool-call round limits, tool error handling, or completion-result validation.
- Persisting Agent state to a database, providing process-level crash recovery, or adding a user interface.
- Parallel Plan-step execution, automatic retries within a Step execution, or Executor-initiated replanning.
- Memory-summary generation or mutation.

## Further Notes

The agreed test seam is the Agent's high-level asynchronous `run` operation with injected Planner and Executor collaborators. This is the highest existing architectural seam that exercises the complete coordination behavior while avoiding live model and MCP dependencies.

