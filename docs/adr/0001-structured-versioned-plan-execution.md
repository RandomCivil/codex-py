# Structured, versioned Plan–Execute execution

The Agent coordinates a Planner and Executor through immutable, machine-validatable Plan revisions. The Planner derives a whole-goal, strictly serial Plan from the goal and Agent state, emitting strict JSON that is validated into Plan and PlanStep values; the Executor chooses tools at runtime to pursue each step's intent but never modifies the Plan. Failed execution is recorded and returned to the Agent, which may request a new Plan revision up to a three-revision goal budget before returning a blocked result. This preserves explicit tool authority while making planning and execution history auditable.

## Considered Options

- Bind tool invocations in Plan steps — rejected because tool choice belongs to the Executor at execution time.
- Let the Executor replan autonomously — rejected to preserve a clear planning/execution boundary.
- Overwrite Plans or plan one step at a time — rejected because immutable whole-goal revisions provide a reviewable path and failure history.
