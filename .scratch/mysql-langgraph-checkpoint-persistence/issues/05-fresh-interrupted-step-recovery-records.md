# 05: Fresh interrupted-step recovery records

**What to build:** `agent resume` automatically starts a fresh attempt for the one interrupted Plan step, without repeating completed steps. It restores the Agent's Step executions and trustworthy cumulative Context updates from project-owned MySQL recovery records, marks the Executor as a recovery attempt, and requires it to inspect and reconcile the Step's external state before pursuing its original completion criterion.

**Blocked by:** 04: Safe interrupted Step-execution recovery.

**Status:** ready-for-agent

- [ ] Add a versioned, idempotently migrated `step_recovery_attempts` project table. It stores one immutable record for each `(run_id, revision, step_id, attempt)`, including its Step execution, nullable completed Context update, recovery flag, common monotonic version, and timestamps.
- [ ] Preserve `AgentState.step_executions` as the latest execution for each `(revision, step_id)`, while recording every attempt durably. Completed steps never receive a new attempt; an interrupted Step receives the next number and a new Executor thread identity.
- [ ] Change ordinary `agent resume` to recover automatically. Remove the idempotency gate and redundant `retry` recovery option; retain `--recovery fail|abort` as explicit alternatives.
- [ ] Build recovery context from all completed attempts in the Agent run, including previous Plan revisions, and load all latest Step executions into restored Agent state. Do not use interrupted-attempt messages or tool output as factual context.
- [ ] Pass a recovery-attempt marker to the new Executor attempt. Its prompt directs it to verify the current external state and reconcile it before taking action; its original Plan intent and completion criterion remain unchanged.
- [ ] Associate Checkpoint state and recovery records with a common monotonic version. Resume accepts only the latest mutually confirmed version and fails closed, before model or MCP tool work, when that version cannot be established.
- [ ] Cover normal resume after graceful cancellation, lease-expiry recovery after hard process loss, completed-step non-reexecution, new-attempt identity, context restoration across Plan revisions, `fail`/`abort`, common-version mismatch failure, and MySQL migration/round-trip behavior. Assert no dependency on persisted per-tool invocation or idempotency snapshots.

## Answer

Design agreed through `$grill-with-docs` on 2026-09-18. ADR-0004 records the recovery-policy change and supersedes the relevant part of ADR-0003.
