# 04: Persist completed Step evidence across recovery

**What to build:** A completed Plan step retains the Completion judge's concise evidence separately from its handoff, so a durable Agent run can audit and restore the completed Step. An interrupted Step starts a fresh attempt that rechecks external state without trusting provisional judge progress, while a completed Step is not executed again.

**Blocked by:** 01: Judge a Tool-backed Plan step.

**Status:** completed

- [x] A completed Step execution carries its handoff result and final judge evidence as distinct values attributed to its Plan revision and step ID; pending or failed attempts do not persist provisional completion as success.
- [x] Durable Step records and recovery restore the final evidence of completed Steps, while previously stored records remain readable after the persistence change.
- [x] Resuming an interrupted Step starts with pending criterion state and rechecks actual external state; resuming a completed Step skips execution and retains its recorded evidence.
- [x] Controlled Agent and durable recovery behavior demonstrates completed evidence retention, interrupted-attempt reset, and unchanged serial Plan completion and replanning decisions.

## Comments

- Implemented with TDD. Completed evidence is retained separately from handoff, legacy recovery rows remain readable, interrupted attempts reset provisional evidence, and completed steps are skipped on recovery. Full suite: 337 passed, 7 skipped.
