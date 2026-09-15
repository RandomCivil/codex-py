# 01 — Unify Plan and Agent state contract

**What to build:** Agent developers can create a validated, immutable Plan for a whole goal and independently record the current status and outcome of each Step execution. The preliminary state model is aligned with the agreed Plan–Execute vocabulary, so planning and future execution cannot confuse intent with runtime state.

**Blocked by:** None — can start immediately.

**Status:** resolved

- [x] A Plan contains a sequential collection of uniquely identified Plan steps, each with an intent and completion criterion, without tool calls or mutable runtime status.
- [x] Agent state retains the goal, immutable Plan revision history, optional memory summary, and Step executions keyed by Plan revision and Plan-step ID.
- [x] Step execution exposes only the agreed pending, running, completed, failed, and skipped statuses, independently from the Plan.
- [x] The model enforces serializable, immutable data and rejects invalid revisions, duplicate step IDs, and more than three revisions for a goal.

## Comments

Implemented the immutable in-memory Plan–Execute state contract in `memory.state`:

- Added `PlanStep`, `Plan`, `StepExecution`, and `AgentState` frozen dataclasses.
- Added `AgentState.with_plan()` and `AgentState.with_step_execution()` for immutable history/state updates.
- Enforced non-empty fields, unique step IDs, sequential revisions, plan/execution associations, supported execution statuses, and the three-revision limit.
- Added focused public-interface tests in `tests/test_state.py` and included `memory` in the Poetry package list.

Verification: `python -m pytest -q` — 15 passed; `poetry check` — All set.
