# Observation decision impact controls context replacement

**Status: accepted.** ADR-0018's static Tool-call evidence classification remains the entry point: only successful observation-class calls request an Observation, while native reads and failed calls retain their existing Raw-evidence lifecycles. Each such Observation also receives a deterministic decision summary—Goal, execution mode, applicable Plan step and completion criterion, and its triggering tool name and arguments—and must return `affects_current_decision` plus `affected_targets` drawn from `confirmed_facts`, `summary`, and `durable_state`. A valid positive Observation replaces its Raw fallback in later Runtime contexts; a valid negative result leaves Raw evidence until its existing window expires and then omits it from Runtime context, while complete provenance remains diagnostic-only.

Observation generation remains asynchronous: a pending, failed, or invalid response retains Raw fallback and never suppresses evidence. A positive impact must declare one or more targets, a negative impact none; Observation merging keeps positive impact and unions targets. `durable_state` is explanatory metadata only and cannot mutate ReAct or Agent state. This applies identically to ReAct and the Plan-step Executor, so the two loops do not develop divergent evidence semantics.

## Considered Options

- Always replace successful observation-class Raw evidence with an Observation — rejected because semantically irrelevant results consume context after their corrective Raw window.
- Let the Observation model determine evidence class for every tool call — rejected because exact native reads and failed-call correction evidence must remain lossless under ADR-0018.
- Block the tool loop until impact classification completes — rejected because it violates the nonblocking Observation lifecycle.
