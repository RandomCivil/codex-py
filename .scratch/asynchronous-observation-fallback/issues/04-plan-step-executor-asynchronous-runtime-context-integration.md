# 04: Plan-step Executor asynchronous Runtime context integration

**What to build:** A Plan-step Executor continues from settled Raw tool evidence without waiting for Observation generation, while its operational and completion-receipt requests receive the shared instantaneous Runtime context window and preserve established durability boundaries.

**Blocked by:** 01: Asynchronous Raw fallback in Runtime context policy; 02: Non-blocking Observation lifecycle and diagnostics.

**Status:** ready-for-agent

- [ ] The Plan-step Executor advances from a settled Tool-call batch without awaiting its Observation task.
- [ ] Operational and structured completion-receipt requests receive the newest-three-Raw window plus ready Observations or Raw fallbacks for earlier rounds.
- [ ] Observation failure or invalid output does not fail the Step execution; existing completion, Tool-round, batch, and tool-error semantics remain unchanged.
- [ ] Runtime Raw evidence, Observations, and asynchronous task outcomes remain excluded from checkpoints, Durable State, Step context, and recovery inputs.
