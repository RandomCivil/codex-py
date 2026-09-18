# Deterministic task routing after descriptive analysis

**Status: accepted.** A Task Router analyzes a goal before creating the selected
whole-task Execution mode. The Task Analyzer remains descriptive and cannot
recommend an architecture. Routing is an ordered policy: no tools selects
`direct`; a tool-requiring task with at most two expected steps and path
uncertainty below `0.3` selects `tool_agent`; a long-horizon task, a task with
at least three open subgoals, or a task with need for replanning above `0.7`
selects `plan_execute`; every other tool-requiring task selects `react`.
Analyzer invocation or validation failure conservatively selects
`plan_execute`, retains only a safe diagnostic, and does not retry or escalate
after the selected mode starts. The router returns the selected mode, optional
analysis evidence, and the unchanged Execution answer.

This preserves ADR-0007: the existing Execution-mode factory remains the
selection seam, the Plan-step Executor remains exclusive to `plan_execute`,
and a failed mode never triggers automatic fallback.
