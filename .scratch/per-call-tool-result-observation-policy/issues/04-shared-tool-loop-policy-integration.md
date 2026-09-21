# 04: Shared tool-loop policy integration

**What to build:** Deliver the complete per-call Tool-call evidence policy through both ReAct mode and the Plan-step Executor, so callers receive the same nonblocking Observation behavior and selected Runtime context regardless of execution mode, while existing concurrent batch and recovery contracts remain intact.

**Blocked by:** 03: Per-call write Observations.

**Status:** ready-for-agent

- [ ] ReAct continues to the next operational Model request without waiting for a deliberately blocked write Observation and receives pending Raw, completed Observation, or fallback Raw evidence as applicable.
- [ ] The Plan-step Executor demonstrates the same policy and nonblocking behavior in a fresh invocation-local Runtime context.
- [ ] Tool-call batch concurrency, completion barriers, request-order results, individual errors, and cancellation remain unchanged in both integration paths.
- [ ] Runtime Raw tool results, per-call classifications, Observation outcomes, and runtime Observations remain excluded from Durable State, Step context, checkpoints, recovery inputs, and terminal Execution answers.
- [ ] Existing provider configuration, Line Protocol, model-use accounting, trace diagnostics, and context-budget error behavior remain compatible with the shared policy.
