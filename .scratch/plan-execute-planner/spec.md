# Structured Plan–Execute Planner

Status: ready-for-agent

## Problem Statement

The Codex Python agent has an asynchronous LLM adapter but no planning layer. An agent developer cannot turn a goal and its current Agent state into a validated, auditable Plan before later execution. Planning, execution authority, retries, and recovery therefore have no explicit boundary.

## Solution

Provide a Planner that derives a complete, strictly serial, structured Plan from a goal and Agent state through the existing Text stream interface. The Planner requires strict JSON from the model and validates it into immutable Plan and Plan step values. The surrounding state model retains immutable Plan revisions and per-step execution state so a future Agent can replan safely without allowing an Executor to mutate a Plan.

## User Stories

1. As an agent developer, I want a Planner that turns a goal and Agent state into a complete Plan, so that execution starts from an explicit contract.
2. As an agent developer, I want the Planner to use the existing asynchronous Text stream, so that planning shares the project's established LLM integration seam.
3. As an Agent, I want an initial Plan revision to be numbered predictably, so that later execution records can identify the Plan they belong to.
4. As an Agent, I want a replanned goal to produce a new immutable Plan revision, so that earlier decisions and their outcomes remain auditable.
5. As an Agent, I want a Plan to describe the whole goal, so that the intended path is visible before execution begins.
6. As an Agent, I want Plan steps to be strictly ordered, so that the initial Executor has unambiguous serial behavior.
7. As a Planner consumer, I want every Plan step to have a stable unique ID, so that its Step execution can be associated with precisely one task item.
8. As a Planner consumer, I want every Plan step to state its intended outcome, so that an Executor can choose suitable tools at execution time.
9. As a Planner consumer, I want every Plan step to state a completion criterion, so that execution can be evaluated against the planned result.
10. As a tool owner, I want Plan steps to omit prescribed tool calls, so that the Executor retains tool-choice and side-effect authority.
11. As an Agent, I want Agent state to include the goal, Plan history, Step executions, and optional memory summary, so that initial planning and replanning use the same input boundary.
12. As an Agent, I want Step execution status to be separate from the Plan, so that planning remains immutable while a specific task item's live status is visible.
13. As an Agent, I want a Step execution to be identified by its Plan revision and step ID, so that state from prior revisions cannot be confused with current work.
14. As an Agent, I want Step execution statuses to distinguish pending, running, completed, failed, and skipped work, so that later orchestration can make explicit decisions.
15. As an Agent, I want Planner output to be strict JSON that is validated before becoming a Plan, so that prose or malformed model output is never executed accidentally.
16. As an agent developer, I want invalid JSON and invalid Plan schemas to fail explicitly, so that the Agent can treat planning failure as a controlled outcome.
17. As an agent developer, I want the Planner to receive no direct tool or environment access, so that it cannot bypass execution permissions.
18. As an Agent, I want Plan revisions to be capped at three per goal, including the initial revision, so that future orchestration has a defined termination budget.
19. As an Agent, I want a failed step to be recorded rather than retried within its revision, so that repeated side effects are not hidden.
20. As an application developer, I want the initial Agent state model to be in-memory immutable data, so that planning can be adopted without introducing persistence or recovery infrastructure.
21. As a maintainer, I want Plans and Step executions to contain serializable data, so that durable storage can be introduced later without replacing the domain contract.
22. As a maintainer, I want the Planner contract testable with a controlled Text stream, so that tests need no provider credentials or live model.

## Implementation Decisions

- Introduce an `agent` package containing the public Planner abstraction. It receives the existing LLM abstraction and exposes an asynchronous operation that accepts Agent state and returns a validated Plan.
- Introduce an in-memory state model containing immutable Plan, Plan step, Agent state, and Step execution values. The model uses only serializable data fields; persistence is deliberately deferred.
- A Plan has exactly `revision`, `goal`, and an ordered sequence of `steps`. It has no mutable status, timestamp, or model rationale.
- A Plan step has exactly `id`, `intent`, and `completion_criterion`. Its position in the Plan defines order. It contains neither tool calls nor execution state.
- Step execution is held outside the Plan and is keyed by Plan revision plus Plan-step ID. It carries one of `pending`, `running`, `completed`, `failed`, or `skipped`, plus the recorded result or error where applicable.
- Plan revisions are immutable and append to history; they are never overwritten. The maximum is three revisions per goal, including the initial Plan.
- The Planner's model request includes the goal, current Agent state, and optional memory summary. The Planner does not call tools or inspect the environment.
- The Planner instructs the model to return only strict JSON for the full Plan. It parses and structurally validates the response, including revision sequencing, required fields, non-empty steps, and unique Plan-step IDs.
- Model output that is malformed JSON, has unknown or missing fields, has an invalid revision, or violates Plan invariants produces an explicit planning validation failure. It does not yield a partially usable Plan.
- The future Agent owns the loop Planner → Executor → result evaluation → optional replanning. The Executor chooses tools at runtime and makes only one attempt for a Plan step in a revision; it never changes the Plan or invokes the Planner itself.
- This specification implements only the Planner and its required state model. It preserves the architecture recorded in ADR-0001.

## Testing Decisions

- Test behavior through the single high-level `Planner.plan(AgentState)` seam, using a controlled Text stream double rather than a live OpenAI-compatible provider.
- A good test verifies the observable Planner contract: its use of Agent state in the Model request, conversion of valid strict JSON to an immutable Plan, and explicit failure for malformed or structurally invalid output. It must not assert private prompt-construction details.
- Verify initial and subsequent revision behavior, full-goal ordered step output, stable and unique step IDs, and rejection of schemas that violate these invariants.
- Verify state-model invariants independently where they are externally observable: immutable Plan data, separate Step execution state, revision/step association, permitted execution statuses, and the three-revision bound.
- Use the existing LLM tests as prior art: they test the public async stream seam with controlled doubles and make no live provider request.

## Out of Scope

- Implementing the top-level Agent loop, result evaluation, or `blocked` result production.
- Implementing the Executor, tool selection, tool permissions, actual Tool execution, or tool-result submission.
- Retries within a Plan revision, automatic recovery, parallel execution, dependency graphs, or one-step-at-a-time planning.
- Persistent Agent state, process recovery, database storage, migrations, or concurrent state updates.
- Live-provider integration tests, prompt templating beyond the Planner's contract instructions, and changes to the LLM adapter.

## Further Notes

The domain terms Plan, Plan revision, Plan step, Step execution, Agent state, Planner, Agent, and Blocked result use the root glossary. The boundary between immutable planning and runtime execution follows ADR-0001, Structured, versioned Plan–Execute execution.
