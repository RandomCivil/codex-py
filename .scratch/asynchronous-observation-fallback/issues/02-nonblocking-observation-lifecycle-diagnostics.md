# 02: Non-blocking Observation lifecycle and diagnostics

**What to build:** Observation work is best-effort transient context maintenance: every settled Tool round gets one unbounded asynchronous attempt, while failures and invalid output fall back to Raw evidence without failing the active execution or weakening diagnostic visibility.

**Blocked by:** 01: Asynchronous Raw fallback in Runtime context policy.

**Status:** resolved

- [x] Every started Observation request counts as model use but never consumes a Tool round, and no failed or invalid request is retried.
- [x] Provider, cancellation, and validation outcomes are diagnosable while failed or invalid outcomes leave the active execution usable through Raw fallback.
- [x] Observation tasks have no concurrency cap, and a terminal execution neither cancels nor waits for outstanding tasks.
- [x] Outstanding work remains process- and event-loop-local, cannot alter a terminal Execution answer, and never becomes Durable State or recovery input.

## Answer

Implemented transient `ObservationOutcome` diagnostics on `RuntimeContextPolicy` for pending, succeeded, provider-error, invalid, and cancelled asynchronous Observation attempts. Observation requests increment model-use accounting without affecting the Tool-round count, are started once with no concurrency cap, and failures remain usable through complete Raw fallback. Outcomes are exposed through the policy and forwarded to the trace hook; they are not part of `RuntimeContext`, Durable State, checkpoints, Step context, or recovery inputs.

Verification: `poetry run pytest -q` — 257 passed, 7 skipped.
