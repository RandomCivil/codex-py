# 03: ReAct asynchronous Runtime context integration

**What to build:** A ReAct invocation continues from settled Raw tool evidence without waiting for Observation generation, while every operational and final goal-satisfaction request receives the shared instantaneous Runtime context window.

**Blocked by:** 01: Asynchronous Raw fallback in Runtime context policy; 02: Non-blocking Observation lifecycle and diagnostics.

**Status:** ready-for-agent

- [ ] ReAct advances from a settled Tool-call batch without waiting for its Observation task.
- [ ] ReAct operational and final structured completion requests receive the newest-three-Raw window plus ready Observations or Raw fallbacks for earlier rounds.
- [ ] Observation failure or invalid output does not fail ReAct; the original Tool-round limit, concurrent request-order batch behavior, tool-error handling, and goal-satisfaction contract remain unchanged.
- [ ] ReAct retains no Raw tool result, runtime Observation, or asynchronous task outcome beyond its ephemeral invocation.
