# Fresh recovery attempts for interrupted Plan steps

**Status: accepted.** ADR-0004 supersedes the interrupted-step recovery policy in ADR-0003. A normal `agent resume` automatically starts a new, numbered attempt for the interrupted Plan step, while completed steps remain complete; the new attempt receives a recovery marker plus only trustworthy state: the Agent state and cumulative Context updates from completed steps across all Plan revisions. It starts from the Plan step's beginning, first verifies the external state and then converges on the completion criterion; unconfirmed messages and tool output from the interrupted attempt are retained only for diagnosis. This deliberately permits re-invocation even after an unknown side effect, because the recovery strategy is reconciliation rather than an idempotency gate. Operators retain `fail` and `abort`, but no longer need `retry`.

The project records immutable Step recovery records in its own MySQL table, one per `(run ID, Plan revision, Plan step ID, attempt number)`, containing the Step execution and its successful Context update. Checkpoint state remains the authority for graph scheduling; the recovery records are its queryable, recovery-facing projection. Both carry a common monotonic version, and resumption uses only their newest mutually confirmed version; an unverifiable mismatch fails closed before model or tool work. This keeps LangGraph's saver and schema ownership intact while ensuring resumption can reconstruct the Agent context without depending on saver-internal table layout.

## Considered Options

- Continue the interrupted Executor graph from its latest messages — rejected because unconfirmed tool output would become context.
- Require recorded tool-level idempotency evidence before resuming — rejected because recovery reconciles the Step's external outcome from its original intent instead.
- Re-run all prior Plan steps — rejected because completed work is already durable and must never be repeated.
- Make project records the graph authority — rejected because that would duplicate LangGraph scheduling and Checkpoint responsibilities.
