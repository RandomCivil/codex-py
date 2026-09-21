# Asynchronous Observation fallback

Status: ready-for-agent

## Problem Statement

Tool-capable execution currently waits for the Runtime-context component to generate and validate an Observation after every Tool-call batch. That extra language-model request stalls ReAct and the Plan-step Executor even though the complete Raw tool result is already available. A failed or invalid Observation also fails the active execution and discards the usable in-memory context path, turning a nonessential compacting operation into a task failure.

## Solution

After each Tool-call batch settles, record its Raw tool result immediately and launch one no-tool, structured Observation request as an independent asynchronous task. Do not delay a subsequent ReAct or Plan-step Executor request for that task. Assemble each Runtime context window from an instantaneous snapshot: always include complete Raw tool results for the newest three Tool rounds, use a validated Observation for an older round when one is ready, and otherwise use that older round's Raw tool result.

An Observation request that fails, is cancelled by process/event-loop shutdown, or produces invalid structured output is not retried and does not fail the active execution. Its Raw tool result remains the authoritative fallback. Required Raw fallback evidence must never be truncated or re-summarized; if Durable State plus required Raw evidence cannot fit the configured context budget, fail the execution. Budget-driven merging of already validated Observations remains synchronous and keeps its existing failure behavior.

## User Stories

1. As an application integrator, I want a settled Tool-call batch recorded as a Raw tool result before Observation work starts, so that the next tool-capable Model request has complete evidence without waiting for a second model call.
2. As an application integrator, I want each Tool round to start one independent Observation request, so that compact historical context can become available while the tool loop continues.
3. As a ReAct caller, I want Observation generation not to increase latency between Tool rounds, so that the loop can continue acting from settled tool evidence.
4. As a Plan–execute caller, I want the Plan-step Executor not to wait for Observation generation, so that the same latency behavior applies to both tool-capable execution modes.
5. As a model, I want the three newest Tool rounds supplied as complete Raw tool results, so that I can inspect exact arguments, results, and errors while correcting recent work.
6. As a model, I want an older Tool round supplied as a validated Observation once its asynchronous request completes, so that I receive compact, attributable historical evidence.
7. As a model, I want an older Tool round whose Observation is pending supplied as its Raw tool result, so that no historical evidence disappears merely because its compact form is late.
8. As a model, I want an older Tool round whose Observation request fails supplied as its Raw tool result, so that a context-maintenance failure does not hide tool evidence.
9. As a model, I want an older Tool round whose Observation is structurally invalid supplied as its Raw tool result, so that unvalidated model output is never treated as evidence.
10. As an application integrator, I want the fifth Tool-round request to contain Raw results for rounds 2–4 and the ready Observation for round 1, so that the three-round Raw window remains exact.
11. As an application integrator, I want the fifth Tool-round request to contain Raw results for rounds 1–4 when round 1's Observation is unavailable, so that fallback behavior is explicit and lossless.
12. As an application integrator, I want a successful Observation to retain its round number and tool-call-ID-attributed confirmed facts, reported errors, and model inferences, so that compact evidence remains auditable.
13. As an application integrator, I want only one Observation request per settled Tool round, so that failed work is not silently retried and model-use accounting remains predictable.
14. As an operator, I want an Observation failure or validation failure recorded for diagnostics without failing the active execution, so that degraded context remains observable.
15. As an application owner, I want outstanding Observation tasks to have no concurrency limit, so that every settled Tool round gets its best-effort compact representation.
16. As an application caller, I want a terminal execution neither to cancel nor wait for outstanding Observation tasks, so that terminal answer latency is not extended by best-effort context work.
17. As an application operator, I want outstanding tasks to remain process- and event-loop-local, so that the feature introduces no background worker, persistent queue, cross-execution handoff, or durable state.
18. As an application owner, I want an Observation task to be harmless if process or event-loop shutdown cancels it, so that shutdown does not alter an already terminal Execution answer.
19. As an application owner, I want every Runtime context to respect its configured token budget, so that tool loops retain bounded context behavior.
20. As an application owner, I want an oversized window caused by required Durable State and Raw fallback evidence to fail explicitly, so that no evidence is truncated or re-summarized without validation.
21. As an application owner, I want merging of already validated Observations to remain synchronous when needed for the next request to fit, so that a requested model call never starts with an over-budget context.
22. As an application owner, I want failed Observation merging to fail the execution, so that the strict compaction contract for validated evidence remains intact.
23. As a ReAct caller, I want Observation calls to count as model use but not Tool rounds, so that the Tool-round limit still bounds only Tool-call batches.
24. As a Plan–execute caller, I want each new Step execution to retain the existing transient-state and recovery boundaries, so that Raw tool results and runtime Observations never become Durable State or Step context.
25. As an operator, I want the complete Raw tool result—including ordered calls, arguments, results, and errors—to remain available to diagnostic tracing, so that asynchronous fallback does not weaken investigations.

## Implementation Decisions

- Amend the shared Runtime-context policy used by ReAct mode and the Plan-step Executor; it remains the sole owner of transient Tool-round history, Observation lifecycle, window assembly, and budget maintenance.
- On each settled Tool-call batch, construct and retain the Raw tool result immediately, increment the Tool-round count, and launch exactly one no-tool structured Observation task without awaiting it.
- An Observation contains the existing three evidence categories: tool-call-ID-attributed confirmed facts, reported errors, and model inferences. “Fact” is not a separate lifecycle or persistent object; it is the confirmed-facts category inside an Observation.
- Context assembly is snapshot-based. The newest three rounds are always Raw. For every older round, use its validated Observation if successful; otherwise use its full Raw tool result, including while the task is pending or after it failed validation or provider invocation.
- Observation request failures and invalid responses become recorded task outcomes. They neither retry nor fail the active execution; they only select Raw fallback for that round.
- Do not set an Observation-task concurrency limit. Do not cancel or await outstanding tasks when an execution becomes terminal. They have no persistent owner beyond the current process/event loop and cannot amend a terminal Execution answer.
- Keep Raw tool results and runtime Observations transient. They remain excluded from Durable State, Step context, checkpoints, recovery inputs, and later execution invocations.
- Continue counting Observation requests as model use. Observation work does not consume the Tool-round budget.
- Preserve existing synchronous, strict-schema merging for already validated Observations when context-budget maintenance requires it. Merging does not operate on Raw fallback evidence.
- Fail the execution when required Durable State plus required Raw evidence cannot fit the configured budget, or when synchronous Observation merging fails. Do not truncate, discard, or model-compress Raw fallback evidence.
- Preserve existing component provider configuration, structured-output mode, trace hooks, and provenance requirements for Observation and merge requests.

## Testing Decisions

- Test externally observable policy behavior through the shared RuntimeContextPolicy seam, rather than task scheduling internals or private message-list topology. Confirm ReAct and Plan-step Executor integration through their existing public execution seams.
- Extend the existing runtime-context policy test style, which uses controlled asynchronous model doubles that capture requests and deterministically return valid, invalid, delayed, or failed responses without live providers.
- Verify that recording a settled Tool-call batch returns without waiting for a deliberately blocked Observation request, while Raw evidence is immediately visible to the next assembled context.
- Verify the round-five window: Raw rounds 2–4 plus a ready Observation for round 1; separately verify Raw rounds 1–4 when round 1 is pending, fails, or is invalid.
- Verify that the newest three rounds remain Raw even if their Observations have already completed.
- Verify that a late successful Observation replaces only its eligible older round in a later context snapshot and preserves round and tool-call-ID provenance.
- Verify that provider errors and invalid Observation payloads do not fail ReAct or the Plan-step Executor, do not trigger another request for the same round, and retain the corresponding Raw fallback.
- Verify that tool errors and nonzero tool results remain visible in Raw fallback and are classified as reported errors in a successful Observation.
- Verify model-use accounting includes each started Observation request without increasing the Tool-round count.
- Verify that terminal execution neither waits for nor cancels pending Observation tasks, and that later task completion cannot alter the terminal Execution answer.
- Verify no Observation task state, Raw tool result, or runtime Observation is serialized into Durable State, Step context, checkpoints, or recovery inputs.
- Retain existing budget tests: required Raw evidence cannot be merged away, an over-budget window with insufficient mergeable validated Observations fails, and an invalid or failed synchronous merge fails.
- Retain existing trace tests and verify that asynchronous Observation requests and their outcomes remain diagnosable while Raw results remain the complete diagnostic evidence.

## Out of Scope

- Persisting Observation tasks, results, Raw tool results, or runtime Observations.
- Introducing a worker queue, background service, concurrency limit, retry policy, timeout policy, or cross-execution result handoff for Observation tasks.
- Changing the three-round Raw window, Tool-call batch ordering/concurrency semantics, Tool-round limit, ReAct completion contract, Plan-step recovery model, or component-provider configuration model.
- Compacting, truncating, or re-summarizing Raw fallback evidence to avoid a context-budget failure.
- Making budget-driven merging asynchronous or changing its existing strict failure behavior.

## Further Notes

This specification uses the glossary terms Observation, Raw tool result, Runtime context window, Tool round, Durable State, Step context, ReAct mode, and Plan-step Executor. It implements the accepted asynchronous fallback decision in ADR-0016, which amends the synchronous Observation failure semantics in ADR-0010. The previously published layered-tool-context specification and implementation tickets describe the superseded “Observation failure fails execution” behavior and must be reconciled before implementation planning begins.
