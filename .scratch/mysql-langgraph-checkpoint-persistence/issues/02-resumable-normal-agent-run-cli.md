# 02: Resumable normal Agent run CLI

**What to build:** An operator can start a goal with `agent run --goal`, receive a UUIDv4 Agent-run ID in stable JSON, and resume unfinished normal work with `agent resume --run-id`. The complete top-level Agent Plan–Execute flow is checkpointed in MySQL at its decision boundaries, while the established Agent state invariants and immutable Plan revisions remain intact. Resuming a completed or Blocked result returns that saved terminal result without any new model, Planner, Executor, or Tool execution.

**Blocked by:** 01: Migratable MySQL Checkpoint run foundation.

**Status:** resolved

- [ ] A normal Agent run persists and restores its latest top-level state across a fresh application process, retains completed Step executions, and never re-executes them.
- [ ] The CLI exposes only the agreed run and resume lifecycle with one stable JSON result and the documented completed, blocked, configuration, persistence, and invalid-invocation outcomes.
- [ ] Controlled Planner and Executor tests retain the existing observable Agent behavior; local-MySQL integration tests demonstrate start, checkpoint, resume, and terminal idempotency.

## Answer

Implemented the normal durable Agent-run path with a JSON-compatible Agent-state adapter, a checkpointed LangGraph plan/execute/replan graph, UUIDv4 run IDs, `agent run --goal`, and `agent resume --run-id`. Added controlled tests for state round trips, terminal resume without duplicate Executor work, and immutable replanning after failure. The existing MySQL integration suite remains explicitly skipped unless `CODEX_TEST_MYSQL_URL` is configured.
