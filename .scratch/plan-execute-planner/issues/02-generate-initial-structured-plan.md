# 02 — Generate an initial structured Plan

**What to build:** Given an initial Agent state, an agent developer can call the Planner and receive a complete, strictly serial, validated first Plan revision derived through the existing Text stream, ready for a future Executor to act on.

**Blocked by:** 01 — Unify Plan and Agent state contract.

**Status:** resolved

- [x] The Planner accepts Agent state and obtains planning output solely through the existing asynchronous Text stream boundary.
- [x] The Planner provides the goal, state, and optional memory summary to the model and requests strict JSON for the complete Plan.
- [x] Valid output returns revision one with the whole goal and ordered, uniquely identified Plan steps.
- [x] Tests verify the public planning operation with a controlled Text stream and no live provider.

## Comments

- Added the public `Planner.plan(AgentState)` async seam in `agent`, backed only by the existing `stream_text` boundary.
- Strictly validates the complete JSON document and converts it to the immutable `Plan`/`PlanStep` state model; tools are never supplied to the model.
- Added controlled-stream tests for state propagation, initial ordered plan generation, malformed JSON, and unknown schema fields.
- Verification: `python -m pytest -q` — 19 passed.
