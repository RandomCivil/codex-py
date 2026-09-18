# 07: Automatic fresh interrupted-Step recovery

**Parent:** 05: Fresh interrupted-step recovery records

**What to build:** An operator can run ordinary `agent resume` after an Agent exits during a Plan step and have the Agent start a fresh Execution attempt for that interrupted Step. Completed Steps remain complete; the recovery attempt receives the reconstructed, trustworthy Agent context and reconciles the external state before it pursues the original completion criterion.

**Blocked by:** 06: Durable Step recovery-record foundation.

**Status:** resolved

- [x] An interrupted Step automatically receives a next-numbered Execution attempt and fresh Executor identity on ordinary resume. Its earlier messages and tool output are not resumed or supplied as factual context.
- [x] Completed Steps never receive another attempt. The recovery Executor receives all latest Step executions plus cumulative Context updates from completed Steps in every prior Plan revision, along with a recovery marker that requires external-state inspection and reconciliation.
- [x] The CLI removes redundant `retry` recovery selection while preserving explicit `fail` for replanning and `abort` for a Blocked result; stable JSON and documented exit outcomes remain compatible for the retained commands.
- [x] Graceful cancellation and lease-expiry takeover both result in automatic fresh recovery; terminal runs still return their saved terminal result without Planner, Executor, model, or Tool work.
- [x] Controlled CLI and durable-Agent tests, plus optional local-MySQL integration coverage, prove automatic recovery, completed-Step non-reexecution, cross-revision context restoration, fresh attempt identity, recovery prompt contract, and retained `fail`/`abort` behavior.

## Answer

Implemented automatic fresh interrupted-Step recovery. Ordinary resume converts stale running work to an interrupted recovery record, creates the next immutable attempt, restores only completed-step Context across Plan revisions, and invokes a fresh Executor thread with an explicit reconciliation marker. Completed steps remain untouched. Removed CLI `--recovery retry`; explicit `fail` and `abort` remain supported. Added controlled and local-MySQL coverage for attempt identity, recovery prompts, terminal behavior, and automatic stale-work takeover.
