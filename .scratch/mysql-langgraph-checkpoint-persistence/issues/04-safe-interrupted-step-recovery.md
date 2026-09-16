# 04: Safe interrupted Step-execution recovery

**What to build:** An operator can safely recover an Agent run whose Step execution was interrupted. Each Executor graph receives a derived durable thread identity and checkpoints its progress; a Step execution is checkpointed as running before model or MCP work. Recovery explicitly turns uncertain work into interrupted, then permits fail-for-replanning or abort-as-Blocked, rejects retry because Atom MCP declares no Idempotent tools, and stops before further Tool execution when persistence fails.

**Blocked by:** 03: Exclusive Agent-run ownership and configuration compatibility.

**Status:** resolved

- [x] Executor checkpoints use the stable `{run_id}:r{revision}:s{step_id}` identity; `pending → running` is durable before any model or MCP invocation, and an expired lease or bounded signal shutdown yields an interrupted recovery state.
- [x] `agent resume` accepts `fail` and `abort` for an interrupted Step execution, rejects `retry`, sends fail through the existing immutable replanning behavior, and retains abort history in its Blocked result.
- [x] Persistence failures are fail-closed, and controlled tests plus local-MySQL integration tests prove no duplicate Tool execution occurs during interrupted recovery.

## Comments

Implemented the safe interrupted Step-execution recovery path. Executor graphs can use a Checkpointer with either an explicit thread ID or the stable `{run_id}:r{revision}:s{step_id}` identity, and persist `running` before model/tool work. Agent and durable lifecycle seams now require explicit recovery for stale `running`/`interrupted` work: `fail` records failure for immutable replanning, `abort` returns a Blocked result retaining history, and `retry` is rejected for unannotated MCP tools. CLI resume accepts `--recovery` and maps invalid recovery to exit code 5. Controlled coverage passes; MySQL integration tests remain explicitly skipped unless `CODEX_TEST_MYSQL_URL` is configured.
