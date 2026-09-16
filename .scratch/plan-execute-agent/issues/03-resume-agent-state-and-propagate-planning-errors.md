# 03: Resume Agent state and propagate planning errors

**What to build:** A host application can resume the Agent from recorded Agent state without repeating completed work. The Agent executes pending, running, and skipped Plan steps in serial order, replans after a recorded failure, and exposes Planner contract failures directly to the caller.

**Blocked by:** 02 — Replan failed Step executions.

**Status:** ready-for-agent

- [ ] Completed Step executions are not invoked again when their Agent state is resumed.
- [ ] Pending, running, and skipped entries are treated as incomplete and are executed in their Plan order.
- [ ] A pre-existing failed Step execution in the current revision triggers replanning rather than further execution in that revision.
- [ ] Planner validation or contract errors propagate unchanged and do not become Step executions or Blocked results.
- [ ] Tests cover resumed completed, incomplete, and failed Agent state through the public Agent entry point.

