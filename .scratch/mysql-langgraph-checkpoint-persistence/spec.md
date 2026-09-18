Status: ready-for-agent

# MySQL-backed LangGraph Checkpoint Persistence

## Problem Statement

The Agent can resume only when a caller still holds an in-memory Agent state. A process restart, cancellation, crash, or deployment interruption loses the coordinating state and may cause a resumed Plan step to repeat an external Tool execution. Operators also lack a supported command-line way to start, inspect, migrate, or resume a durable Agent run.

## Solution

Persist both the top-level Agent coordination graph and every Step-execution graph to MySQL using LangGraph Checkpoints. A CLI creates an opaque Agent run, returns its UUIDv4 run ID, and later resumes that run from its latest mutually confirmed persistence version. A project-owned Run registry provides configuration compatibility checks and an exclusive renewable lease. Project-owned immutable Step recovery records make execution attempts and successful Context updates queryable and reconstruct the trustworthy Agent context. An interrupted Step resumes automatically as a fresh attempt that first reconciles its external state; completed Steps are never repeated.

## User Stories

1. As an operator, I want to start an Agent run from the CLI with a goal, so that the run receives a durable identity before work begins.
2. As an operator, I want `agent run` to return a UUIDv4 run ID in stable JSON, so that I can resume or script against the same Agent run.
3. As an operator, I want `agent resume` to continue an unfinished Agent run from its latest Checkpoint, so that process restarts do not discard completed Plan work.
4. As an operator, I want `agent migrate` to initialize and upgrade the persistence schema explicitly, so that execution commands do not make unreviewed schema changes.
5. As an operator, I want `agent run` and `agent resume` to reject an uninitialized schema, so that deployment permissions and schema ownership remain explicit.
6. As an operator, I want MySQL connection configuration to come only from `CODEX_MYSQL_URL`, so that credentials are not exposed in CLI arguments or project configuration files.
7. As an operator, I want unsupported database versions rejected by migration, so that checkpoint queries do not fail after execution has started.
8. As an operator, I want both top-level Agent coordination and in-progress Step execution checkpointed, so that recovery is possible between Plan transitions and during a bounded Executor graph.
9. As an operator, I want the top-level Agent graph to persist after plan, execute-step, replan, complete, and blocked transitions, so that the latest coordination decision is recoverable.
10. As an operator, I want each Executor attempt to use a stable identity derived from its Agent run, revision, Plan step, and attempt number, so that fresh recovery work cannot continue an unconfirmed transcript.
11. As an operator, I want recovery to use the latest mutually confirmed Checkpoint and Step recovery-record version, so that the CLI never replays an unconfirmed or split persistence state.
12. As an operator, I want completed Plan steps to remain completed after resume, so that their Tool executions are never repeated.
13. As an operator, I want a Step execution recorded as running before it invokes a model or tool, so that an interruption is distinguishable from work that never started.
14. As an operator, I want a stale running Step execution to become interrupted after its execution lease expires, so that the resumed Agent can create a deliberate recovery attempt.
15. As an operator, I want to record an interrupted Step execution as failed and trigger the established replanning behavior, so that the Planner can respond to uncertain work with its complete history.
16. As an operator, I want to abort an interrupted Agent run as blocked, so that I can stop uncertain work without executing another tool.
17. As an operator, I want ordinary `agent resume` to start a fresh attempt for an interrupted Step automatically, so that an ordinary process exit does not require an extra recovery command.
18. As an operator, I want a recovery attempt to inspect and reconcile current external state before acting, so that it can converge on its completion criterion despite an unknown prior side effect.
19. As an operator, I want only one process to own a run at a time, so that two `resume` commands cannot duplicate external tool side effects.
20. As an operator, I want a competing command to receive a machine-readable `run busy` result, so that it can retry later without guessing whether work is already executing.
21. As an operator, I want a crashed owner to lose its lease after 60 seconds, so that a later resumer can take over.
22. As an operator, I want the active owner to renew its lease every 15 seconds using MySQL server time, so that differing host clocks cannot produce incorrect takeover.
23. As an operator, I want cancellation signals to make a bounded attempt to checkpoint interruption and release the lease, so that ordinary shutdowns recover promptly while hard-kill recovery remains safe.
24. As an operator, I want a persistence failure to stop further Tool execution, so that the system never claims resumability after losing the record of a possible side effect.
25. As an operator, I want a configuration fingerprint recorded with the Agent run, so that resume rejects a changed model or MCP configuration rather than silently changing execution semantics.
26. As an operator, I want completed and blocked runs to return their previously saved terminal JSON on resume, so that CLI retries are idempotent.
27. As a shell-script author, I want documented JSON output fields and exit codes, so that automation can distinguish completed, blocked, busy, configuration, persistence, and invalid-invocation outcomes.
28. As a maintainer, I want project-owned run metadata isolated from LangGraph saver tables, so that third-party checkpointer upgrades do not overwrite project policy state.
29. As a maintainer, I want project-owned schema changes versioned and idempotent, so that repeated migration attempts and upgrades are predictable.
30. As a test author, I want MySQL integration tests against an explicitly configured local database, so that checkpoint serialization, schema setup, leases, and cross-process resume are verified against MySQL itself.
31. As a test author, I want integration tests skipped with a clear reason when `CODEX_TEST_MYSQL_URL` is absent, so that ordinary unit-test runs do not require a local database.
32. As an operator, I want completed Plan steps never to receive a new attempt during resume, so that known-complete work is not repeated.
33. As an operator, I want each Step recovery record to retain an immutable numbered attempt history, so that I can audit interruption and recovery without overwritten evidence.
34. As an operator, I want a resumed Agent to load every prior Step execution, so that its Planner retains failure and completion history across Plan revisions.
35. As an operator, I want a resumed Executor to receive cumulative Context updates only from completed Steps across all Plan revisions, so that it starts with trustworthy working context.
36. As an operator, I want interrupted-attempt messages and tool output excluded from the recovery prompt, so that unconfirmed information is not presented as fact.
37. As an operator, I want `--recovery fail` to retain its existing replanning behavior, so that I can choose not to reconcile uncertain work.
38. As an operator, I want `--recovery abort` to retain its Blocked result behavior, so that I can stop a run without more model or Tool execution.
39. As an operator, I want a persistence-version mismatch to fail closed before model or MCP work, so that a partial MySQL write cannot cause unsafe recovery.
40. As a maintainer, I want the recovery path not to persist per-tool invocation or idempotency snapshots, so that reconciliation remains a Step-level concern rather than a tool-history contract.

## Implementation Decisions

- Preserve the existing validated Agent state domain model and introduce an explicit serializable graph-state adapter. Do not replace domain values wholesale with unvalidated dictionaries.
- Replace the top-level Python Plan–Execute loop with a LangGraph state graph containing plan, execute-step, replan, complete, and blocked nodes. This extends ADR-0001 while preserving its immutable Plan revisions and serial execution semantics.
- Compile the top-level graph and each Executor `model → ToolNode → model` graph with the same MySQL-backed asynchronous checkpointer. The top-level thread ID is the Agent-run ID; an Executor attempt thread ID additionally includes its monotonically increasing attempt number.
- Use `langgraph-checkpoint-mysql[aiomysql]` and its asynchronous saver rather than a project-owned `BaseCheckpointSaver`. Support MySQL versions 8.0.19 through 9.5 only, and verify the version before migration.
- Make `agent migrate`, `agent run --goal <text>`, and `agent resume --run-id <id> [--recovery fail|abort]` the only public interface in this release. The CLI emits one stable JSON object per invocation; omitting `--recovery` selects automatic fresh recovery.
- Require `CODEX_MYSQL_URL` for migration and execution. The URL points to a dedicated MySQL database/schema for checkpoints and run metadata. Execution commands never run migrations implicitly.
- Keep project-owned Run registry and Step recovery-record tables separate from saver-owned tables. The registry records run identity, terminal status, configuration fingerprint, timestamps, and lease ownership/expiry; recovery records retain immutable Step-execution attempts and successful Context updates without duplicating Checkpoint scheduling state.
- Add a project-owned migration history and idempotent migrations for the Run registry, lease storage, and Step recovery records. `agent migrate` applies these before invoking the saver setup operation.
- Create run IDs as UUIDv4 strings. Never use a user identity, goal text, or incrementing identifier as a run ID.
- Implement the execution lease in MySQL with 60-second expiry and 15-second renewal. Every lease decision uses MySQL server time. A conflicting valid lease yields `run busy`; a new owner may take over after expiry.
- Persist `pending → running` before the Executor invokes a model or MCP tool. On graceful cancellation, lease takeover, or other detected exit, remaining running work becomes interrupted before recovery policy is applied.
- A normal resume creates a fresh, numbered attempt for the interrupted Step and starts its Executor graph from that Step's beginning. It never re-executes a completed Step and never continues the prior Executor graph's messages.
- A recovery Executor receives a recovery marker, the original Plan step, restored Agent state containing every latest Step execution, and cumulative Context updates from completed Steps in all Plan revisions. Its prompt requires external-state verification and reconciliation before it acts. Unconfirmed messages and tool output from the interrupted attempt are diagnostic only.
- Do not persist tool-invocation identities or idempotency snapshots for recovery. Tool idempotency does not gate a fresh recovery attempt.
- Persist an immutable Step recovery record for every attempt, keyed by run ID, revision, step ID, and attempt number. It contains the Step execution, nullable completed Context update, recovery marker, common monotonic version, and timestamps.
- Checkpoint state remains authoritative for graph scheduling. The Checkpoint and recovery record each carry the common monotonic version; resume uses their latest mutually confirmed version and raises the dedicated persistence outcome before model or Tool work when that cannot be established.
- Remove `retry` as a CLI recovery option because automatic fresh recovery is its replacement. A `fail` recovery records failure and reuses immutable replanning; an `abort` recovery returns a Blocked result while retaining state and attempt history.
- On SIGINT, SIGTERM, or task cancellation, attempt bounded checkpoint persistence and lease release. A hard process loss relies on lease expiry and interrupted-state conversion.
- Fail closed when MySQL cannot persist the next Checkpoint: do not begin or continue subsequent Tool execution, and return the dedicated persistence outcome.
- Store checkpoint state without field-level encryption or redaction. Do not introduce retention, pruning, deletion, or privacy lifecycle behavior in this feature.
- Persist only non-secret runtime configuration in the Run registry, including provider base URL, model name, MCP configuration, and a stable fingerprint. Resume requires an exact compatible fingerprint; credentials are never included in that snapshot.
- Resume of an already completed or blocked run returns the saved terminal result without Planner, Executor, model, or MCP work.
- Use exit codes 0 for completed, 1 for blocked, 2 for busy, 3 for configuration failures, 4 for persistence failures, and 5 for invalid CLI invocation or recovery decision.
- Add the required async MySQL saver dependency and a CLI entry point while keeping Planner and Executor collaborators injectable for non-CLI tests.

## Testing Decisions

- The primary new behavioral seam is the CLI: use it to assert migration, run creation, JSON result contracts, exit codes, configuration rejection, terminal resume, and busy responses. This is the highest public seam for persistence behavior.
- Exercise top-level Agent graph behavior using controlled Planner and Executor collaborators, following the existing Agent test style. Assert externally visible Agent state, terminal status, and invoked collaborator behavior rather than LangGraph node internals.
- Exercise recovery behavior with controlled Planner and Executor collaborators. Assert fresh attempt identity, restored Step context, recovery marker, completed-step non-reexecution, and `fail`/`abort` outcomes; do not inspect private LangGraph message storage directly.
- Run MySQL integration tests only with a dedicated `CODEX_TEST_MYSQL_URL`; never fall back to the runtime connection URL. Mark them `integration` and skip them with a clear reason when that variable is absent.
- Integration coverage must demonstrate schema initialization, supported-version validation, serial checkpoint persistence, fresh-process automatic resume, terminal resume without re-execution, configuration fingerprint rejection, lease contention, expired-lease takeover, interrupted conversion, immutable attempt persistence, cross-revision context restoration, and fail-closed common-version mismatch behavior.
- Unit coverage must demonstrate Agent-state serialization round trips, pending-to-running persistence, automatic fresh-attempt selection, completed-step non-reexecution, recovery-context construction, `fail`/`abort`, UUIDv4 generation, all project-migration idempotence, MySQL-server-time lease calculations through a repository double, and CLI JSON/exit-code mapping.
- Preserve existing Planner, Agent, and Executor tests as regression coverage. Update prior expectations that stale work requires explicit recovery or resumes the same Executor thread.
- Good tests assert observable CLI responses, persisted/recovered domain state, supplied recovery context, and absence of completed-step re-execution. They must not assert private SQL text, third-party saver table layout, LangGraph checkpoint IDs, message transcripts, or internal graph-edge ordering.

## Out of Scope

- Changes to Atom MCP or its tool metadata, including per-tool idempotency declarations.
- Persisting raw interrupted-attempt message transcripts, tool output, tool-invocation identities, or idempotency snapshots as recovery context.
- HTTP APIs, a graphical user interface, or a public Python API for run management; this release exposes only CLI commands.
- Historical checkpoint selection, replay, rollback, or arbitrary-time travel.
- Checkpoint retention, pruning, deletion, encryption, field redaction, privacy workflows, or backup policy.
- Supporting MySQL versions below 8.0.19, MySQL 9.6 or newer, or non-MySQL persistence providers.
- Automatic schema migration on `agent run` or `agent resume`.
- Parallel Plan-step execution, changes to the three-revision Plan budget, Planner contract changes, or Executor tool-selection policy beyond persistence and recovery integration.
- Using Testcontainers or provisioning a MySQL service. Integration tests use a locally available, explicitly configured MySQL database.

## Further Notes

This specification uses the project glossary's Agent, Agent state, Agent run, Checkpoint, Step execution, Execution attempt, Step recovery record, Context update, Step context, Recovery decision, Run lease, Run registry, and Blocked result terms. It follows ADR-0001 and ADR-0002; ADR-0003 establishes the MySQL persistence foundation, and ADR-0004 supersedes its interrupted-step recovery policy.

The agreed test seams are the CLI for end-to-end durable behavior, the Agent's high-level coordination seam with controlled collaborators, and the Executor's existing execution seam for nested graph behavior. No lower-level persistence seam is required as a public abstraction.
