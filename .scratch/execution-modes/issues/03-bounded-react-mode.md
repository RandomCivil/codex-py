# 03: Bounded ReAct mode

**What to build:** An application integrator can select `react` for an ephemeral, no-Planner/no-Plan tool loop that follows observations across multiple model and tool rounds and returns a completed answer only after explicitly proving the goal is satisfied.

**Blocked by:** 02: Shared Tool runtime and tool-agent mode.

**Status:** resolved

- [x] ReAct reuses the shared Tool runtime and preserves concurrent ordered tool-call batches, tool-error return-to-model behavior, allowlist enforcement, and forced working-directory constraints.
- [x] ReAct has a configurable positive model/tool-round budget with a default of fifty and returns failed when it is exhausted or an unrecoverable runtime error occurs.
- [x] ReAct completes only from a final structured response with a nonempty answer, `goal_satisfied: true`, and no further tool calls; unproven goal completion fails.
- [x] Public-seam tests cover corrective loops after tool errors, batch behavior, completion proof, invalid proof, and budget exhaustion without live provider or MCP dependencies.

## Comments

- Implemented `ReactMode` and factory wiring in `agent/execution.py`.
- Added public-seam tests for corrective tool-error loops, ordered concurrent batches, structured completion proof, invalid proof, and round-budget exhaustion.
- Verification: `poetry run pytest -q` — 145 passed, 5 skipped; `python -m compileall -q agent tests` passed.
