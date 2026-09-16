# 02: Replan failed Step executions

**What to build:** When a Step execution fails, the Agent records the outcome, does not execute subsequent Plan steps in that revision, and derives the next immutable Plan revision from the complete Agent state. After the third failed revision, it returns a Blocked result containing the full Plan and execution history.

**Blocked by:** 01 — Agent completes serial Plan execution.

**Status:** ready-for-agent

- [ ] A failed Step execution is recorded before replanning and prevents later steps in its Plan revision from running.
- [ ] Replanning appends a new Plan revision without changing the prior Plan or execution history.
- [ ] Failure in revision three produces a Blocked result and never requests a fourth Plan revision.
- [ ] Tests verify revision inputs, immutable history, skipped later work, and the bounded terminal outcome through the Agent entry point.

