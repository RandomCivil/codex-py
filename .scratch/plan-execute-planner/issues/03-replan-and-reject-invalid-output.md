# 03 — Replan safely and reject invalid output

**What to build:** Given a Plan history and recorded Step executions, an agent developer can request a new immutable Plan revision, while malformed or invalid model output is rejected explicitly and never becomes an executable Plan.

**Blocked by:** 02 — Generate an initial structured Plan.

**Status:** resolved

- [x] The Planner derives the next sequential Plan revision from Agent state that includes prior Plans and Step executions, without changing history.
- [x] The Planner rejects non-JSON output, missing or unknown schema fields, invalid revision sequencing, empty Plans, and invalid or duplicate Plan-step IDs.
- [x] Planning cannot exceed the three-revision goal budget and produces an explicit validation failure when the contract is violated.
- [x] Tests exercise successful replanning and each invalid-output category through the public Planner seam using a controlled Text stream.

## Comments

Implemented safe replanning through the public `Planner.plan(AgentState)` seam:

- The Planner derives the next revision from immutable Plan history and validates the model revision against it.
- Prior Plans and Step executions remain unchanged; the full state is supplied through the existing Text stream boundary.
- Strict JSON/schema validation now covers sequential revision, empty and duplicate steps, missing/unknown fields, and the three-revision budget.
- Fourth-revision requests fail before contacting the Text stream.

Verification: `python -m pytest -q` — 26 passed; `poetry check` — All set.
