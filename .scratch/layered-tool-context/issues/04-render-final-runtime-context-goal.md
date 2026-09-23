# 04: Render the Goal as the final Runtime-context section

**What to build:** Every ReAct and Plan-step Executor model request ends with a distinct Goal section, while the durable Agent-state goal, Checkpoints, planning, and recovery behavior remain unchanged.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Runtime-context presentation omits the goal from its Durable State section and renders one Goal section after every other context section.
- [x] ReAct's direct goal and the Plan-step Executor's nested Agent-state goal produce the same final Goal presentation without changing their stored durable values.
- [x] Controlled RuntimeContextPolicy tests verify final-section placement and retain existing evidence, budget, persistence, and recovery contracts.

## Answer

Implemented the final `### Goal` Runtime-context section for direct ReAct durable state and nested Plan-step Executor Agent state. The displayed Durable State omits only the goal while the assembled durable payload remains unchanged. Added public prompt-rendering seam tests for placement and nested state preservation.

Verification: `.venv/bin/pytest -q` — 269 passed, 7 skipped.
