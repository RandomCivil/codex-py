# Per-call Tool-result Observation Policy

Status: ready-for-agent

## Problem Statement

The current Runtime context window treats every Tool-call batch as one indivisible unit: it retains the newest three Tool rounds as complete Raw tool results, then attempts one asynchronous Observation for every older batch. That loses the intended distinction between inexpensive directory discovery, durable source evidence, and write effects. A concurrent batch can contain calls needing different lifecycles, yet the current policy must retain, observe, or discard all of them together.

The tool loop needs per-call evidence retention without changing Tool-call batch settlement. `list_dir` and `glob` evidence should be short lived, `grep` and `read_file` evidence should remain exact for the invocation, and write evidence should be compacted by a nonblocking Observation while retaining lossless Raw fallback if compaction is not available. The same contract must apply to ReAct mode and the Plan-step Executor, including a conservative treatment of `exec` commands.

## Solution

Keep a Tool-call batch as the concurrent execution and settlement boundary, but classify and retain each settled Tool call independently under a Tool-call evidence policy. `list_dir` and `glob` retain Raw tool results only while their Tool round is one of the newest three. `grep` and `read_file` retain Raw tool results for the entire invocation. `apply_patch`, `write_file`, and write-class `exec` calls start an independent asynchronous Observation request after their batch settles: their Raw tool result is visible while the request is pending, a validated Observation replaces it after success, and failed, cancelled, or invalid Observation work leaves Raw tool-result fallback visible for the invocation.

The policy uses the same Runtime-context component and provider configuration in ReAct mode and the Plan-step Executor. ReAct and the Executor never wait for an Observation before issuing their next operational Model request. Required Raw tool results are lossless; if Durable State and required Raw evidence do not fit the configured Runtime context window budget, execution fails explicitly. Previously validated Observations retain the existing synchronous, strict-schema merge behavior when budget maintenance requires it.

## User Stories

1. As a ReAct caller, I want directory-discovery Raw tool results retained only for the newest three Tool rounds, so that stale listings do not dominate the Runtime context window.
2. As a Plan–execute caller, I want the Plan-step Executor to apply the same directory-discovery retention rule, so that Execution modes do not disagree about tool evidence.
3. As a tool-calling model, I want the newest three rounds' `list_dir` and `glob` results exactly preserved, so that I can correct a recent navigation decision from its full arguments and output.
4. As a tool-calling model, I want expired `list_dir` and `glob` results omitted rather than summarized, so that obsolete directory listings do not consume context.
5. As a tool-calling model, I want every `grep` result retained as its Raw tool result for the invocation, so that exact search evidence remains available when later reasoning revisits it.
6. As a tool-calling model, I want every `read_file` result retained as its Raw tool result for the invocation, so that later reasoning can cite the exact file evidence it previously inspected.
7. As an application integrator, I want `grep` and `read_file` never to invoke the Observation model, so that durable read evidence is not replaced by a lossy model interpretation.
8. As a tool-calling model, I want an `apply_patch` result represented by a concise Observation after validation, so that write history is useful without permanently retaining a large patch payload.
9. As a tool-calling model, I want a `write_file` result represented by a concise Observation after validation, so that completed write effects remain attributable without unnecessary Raw context growth.
10. As a ReAct caller, I want write-class Observation requests to run asynchronously, so that Observation latency does not delay the next tool-capable Model request.
11. As a Plan–execute caller, I want the Plan-step Executor not to wait for write-class Observation work, so that it shares ReAct's latency behavior.
12. As a tool-calling model, I want a just-settled write-class call's Raw tool result visible while its Observation is pending, so that nonblocking compaction never hides an effect I may need to verify or correct.
13. As a tool-calling model, I want a validated Observation to replace only its originating write-class call in a later Runtime context snapshot, so that unrelated calls in its concurrent batch keep their own retention policy.
14. As an operator, I want a provider failure for one Observation recorded diagnostically without failing the active execution, so that best-effort compaction cannot turn usable tool evidence into a task failure.
15. As a tool-calling model, I want the Raw tool result for a write-class call retained when its Observation is cancelled or invalid, so that the only authoritative evidence is never lost.
16. As an application owner, I want failed, cancelled, and invalid write-class Observations not retried, so that model-use accounting and failure behavior remain predictable.
17. As an application owner, I want each observation-class Tool call to launch exactly one Observation request after its entire Tool-call batch settles, so that concurrent execution continues to have one complete settlement boundary.
18. As a tool-calling model, I want Observation facts, reported errors, and inferences attributed to the originating tool-call ID and Tool round, so that compacted write evidence remains auditable.
19. As an `exec` user, I want a pure top-level `ls`, `find`, or `fd` command classified as directory-discovery evidence, so that shell navigation follows the `list_dir`/`glob` lifecycle.
20. As an `exec` user, I want a pure top-level `rg`, `grep`, `cat`, `sed`, `head`, or `tail` command classified as durable read evidence, so that shell search and file-reading follow the `grep`/`read_file` lifecycle.
21. As an `exec` user, I want a pure top-level `apply_patch`, `tee`, `cp`, `mv`, or `touch` command classified as write evidence, so that shell writes receive asynchronous Observation and Raw fallback.
22. As an application owner, I want a command containing redirection classified as write evidence, so that shell output directed to storage is not mistaken for a read.
23. As an application owner, I want pipelines, command lists, subshells, variable expansion, unknown commands, and otherwise unclassifiable commands classified as write evidence, so that ambiguous effects receive the conservative lifecycle.
24. As a maintainer, I want Tool-call batch concurrency, request-order results, per-call failures, and cancellation behavior unchanged, so that this context feature does not alter the execution contract in ADR-0005.
25. As an application owner, I want a Runtime context window that cannot fit its required Durable State and Raw tool results to fail explicitly, so that no permanent or fallback evidence is silently truncated, evicted, or re-summarized.
26. As an application owner, I want already validated Observations to remain eligible for synchronous budget-driven merging, so that compact evidence can still satisfy the configured budget when it is safe to merge.
27. As an operator, I want Raw tool results and runtime Observations to remain transient and available to diagnostic tracing, so that the new policy does not alter Durable State, Step context, checkpoints, recovery inputs, or later invocations.
28. As an application caller, I want terminal executions neither to wait for nor cancel outstanding Observation tasks, so that best-effort observation work does not extend terminal-answer latency.

## Implementation Decisions

- Amend the shared Runtime-context policy used by ReAct mode and the Plan-step Executor. It remains the sole owner of transient Tool-round history, per-call evidence classification, Observation lifecycle, context-window assembly, and budget maintenance.
- Preserve Tool-call batches as the concurrent settlement boundary. Construct the ordered settled calls after all sibling calls have settled, then evaluate every call independently; do not introduce a second tool scheduler or alter ToolNode behavior.
- Represent per-call evidence with stable tool-call ID, Tool round, tool name, arguments, result, error, classification, and lifecycle outcome. A batch is an ordered collection of these calls, not a single evidence-retention unit.
- Define three evidence lifecycles: recent raw (`list_dir`, `glob`), permanent raw (`grep`, `read_file`, and write-class fallback), and observation-class (`apply_patch`, `write_file`, and write-class `exec`).
- A recent-raw call is visible only while its Tool round is one of the three newest settled Tool rounds. It starts no Observation request and is omitted after expiry.
- A permanent-raw call is visible as an exact Raw tool result for the entire ReAct invocation or Plan-step Executor invocation. It starts no Observation request.
- An observation-class call starts exactly one no-tool, strict-schema Observation task after batch settlement. Its Raw tool result remains visible until the task completes successfully; a successful validated Observation replaces that call's Raw evidence in later context snapshots.
- Do not await an Observation task before subsequent ReAct or Plan-step Executor work. Preserve current no retry, no concurrency limit, process-local task ownership, diagnostic outcome recording, and terminal non-wait/non-cancel behavior.
- A provider error, cancellation, schema failure, evidence-validation failure, or other invalid Observation outcome permanently selects the call's Raw tool-result fallback and does not fail the active execution.
- Classify native tool names exactly: `list_dir` and `glob` are recent raw; `grep` and `read_file` are permanent raw; `apply_patch` and `write_file` are observation class.
- Add a conservative static classifier for `exec` command text. It recognizes only a single pure top-level command from the confirmed read and write command sets. It classifies redirection and every pipe, command-list operator, subshell, variable expansion, unknown command, or other syntactically uncertain command as write class. It must not execute, expand, or infer environment-dependent command behavior while classifying.
- Continue rendering context from Durable State plus per-call evidence in stable Tool-round and original request order. A concurrent batch may therefore contribute a mix of raw calls and Observation calls.
- Keep Observation provenance at the individual-call level. Merged Observations preserve the complete source Tool-round range and source tool-call IDs of their component Observations.
- Keep required Raw evidence lossless. When required Durable State plus selected Raw tool results exceed the configured budget and no safe merge of validated Observations can resolve it, raise the existing explicit context-maintenance failure.
- Preserve existing component-provider configuration, Line Protocol contracts, model-use accounting, tracing hooks, execution-mode round limits, and transient-state/recovery boundaries.
- ADR-0018 is authoritative for the per-call policy and amends the whole-round retention rules of ADR-0010 and ADR-0016.

## Testing Decisions

- Use the existing public RuntimeContextPolicy seam as the primary test boundary: settle calls, permit controlled asynchronous Observation outcomes, and inspect assembled Runtime context. Tests must assert externally observable evidence selection and request content, not private task collections, internal list layout, or event-loop scheduling details.
- Extend the existing controlled context-model doubles to deterministically produce valid, delayed, failed, cancelled, and invalid Observations without using a live provider.
- Test mixed concurrent batches at the policy seam. Verify that each call receives its own lifecycle: directory-discovery calls expire after three rounds, durable reads remain raw, and writes become Observations independently.
- Test the exact three-round boundary for `list_dir` and `glob`, including mixed batches where another call from the expired batch remains visible under permanent-raw or observation-class rules.
- Test that `grep` and `read_file` stay raw after more than three Tool rounds and never generate Observation-model requests.
- Test that `apply_patch`, `write_file`, and write-class `exec` begin one asynchronous Observation each, expose Raw evidence while pending, and replace only their own call with a validated Observation after completion.
- Test provider errors, cancellation, invalid protocol output, and invalid evidence attribution for observation-class calls. Each must preserve permanent Raw fallback, record its diagnostic outcome, avoid retry, and allow the active tool loop to continue.
- Test the `exec` classifier against every confirmed pure command, redirection, pipes, `&&`, `||`, semicolons, subshells, variable expansion, unknown commands, and malformed or otherwise uncertain input. Assert classification only; do not invoke a shell in classifier tests.
- Retain high-level ReAct and Plan-step Executor integration tests as confirmation that both modes share the policy, do not wait for a blocked Observation task, and supply the selected evidence to their next Model request.
- Retain existing concurrent-batch tests to verify batch completion barriers, request-order results, tool-call IDs, per-call errors, and cancellation are unchanged.
- Retain context-budget tests and adapt them to prove that required permanent Raw evidence and observation fallback cannot be truncated, evicted, or re-summarized; verify that safe merging remains restricted to validated Observations.
- Retain persistence and recovery tests to prove Raw tool results, per-call classifications, Observation outcomes, and runtime Observations remain excluded from Durable State, Step context, checkpoints, and recovery inputs.

## Out of Scope

- Changing concurrent Tool-call batch scheduling, ToolNode ownership, request-order result delivery, cancellation propagation, or serial Plan-step execution.
- Persisting Raw tool results, per-call evidence classifications, Observation tasks, Observation outcomes, or runtime Observations.
- Adding a worker queue, persistent task runner, retry policy, timeout policy, task concurrency limit, or cross-execution handoff for Observation work.
- Changing the context budget default, component provider configuration, Line Protocol, model-use accounting, Tool-round limit, ReAct completion contract, or Plan-step recovery contract.
- Classifying arbitrary shell language semantically, executing a command to discover its class, supporting environment-specific aliases/functions, or treating compound commands as independently classifiable fragments.
- Truncating, evicting, re-summarizing, or otherwise weakening required Raw evidence to satisfy a context budget.

## Further Notes

This specification uses the glossary terms Tool-call batch, Tool round, Raw tool result, Observation, Observation outcome, Runtime context window, Tool-call evidence policy, and Exec command class. It follows ADR-0005 for batch concurrency and ADR-0018 for per-call evidence retention. It supersedes the round-wide retention expectations in the asynchronous-observation-fallback and layered-tool-context specifications; those prior feature directories remain historical context rather than implementation authority.
